"""Model loading + batched chat generation (HF transformers / PEFT / bitsandbytes).

Kept independent of Unsloth so the same code runs for every system under comparison, on Kaggle/Colab
GPUs, Apple-Silicon (MPS) or CPU.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class ModelSpec:
    name: str
    base: str
    adapter: Optional[str] = None
    load_in_4bit: bool = False
    prompt: str = "zeroshot"  # zeroshot | fewshot
    guard: bool = False  # solve-then-guard decoding (see eval_generate.py)
    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: Dict) -> "ModelSpec":
        known = {k: d[k] for k in ("base", "adapter", "load_in_4bit", "prompt", "guard") if k in d}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(name=name, extra=extra, **known)

    def available(self) -> bool:
        if self.adapter and not Path(self.adapter).exists():
            return False
        if self.base.startswith(("./", "/", "outputs")) and not Path(self.base).exists():
            return False
        return True


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(spec: ModelSpec, device: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    is_cuda = device.startswith("cuda")
    dtype = torch.float16 if (is_cuda or device == "mps") else torch.float32
    tok_src = spec.adapter if spec.adapter and Path(spec.adapter, "tokenizer_config.json").exists() else spec.base
    tok = AutoTokenizer.from_pretrained(tok_src)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"low_cpu_mem_usage": True}
    if spec.load_in_4bit and is_cuda:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": int(device.split(":")[1]) if ":" in device else 0}
    else:
        kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.base, **kwargs)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(spec.base, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    if spec.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.adapter)
    model.eval()
    return model, tok


def unload(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _render(tok, messages: List[Dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def chat_generate(
    model,
    tok,
    conversations: List[List[Dict]],
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 16,
    seed: int = 0,
    disable_adapter: bool = False,
    return_stats: bool = False,
):
    """Generate one assistant reply per conversation. Batches are length-sorted for throughput."""
    texts = [_render(tok, c) for c in conversations]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs: List[Optional[str]] = [None] * len(texts)
    n_new_tokens = 0
    t0 = time.time()
    device = next(model.parameters()).device
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + b)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        if disable_adapter and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                gen = model.generate(**enc, **gen_kwargs)
        else:
            gen = model.generate(**enc, **gen_kwargs)
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            row = new[j]
            n_new_tokens += int((row != tok.pad_token_id).sum())
            outs[i] = tok.decode(row, skip_special_tokens=True).strip()
    elapsed = time.time() - t0
    if return_stats:
        return outs, {"seconds": elapsed, "new_tokens": n_new_tokens,
                      "tokens_per_sec": n_new_tokens / max(elapsed, 1e-9),
                      "sec_per_response": elapsed / max(len(texts), 1)}
    return outs
