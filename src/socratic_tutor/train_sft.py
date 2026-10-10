"""QLoRA supervised fine-tuning of the Socratic tutor (Unsloth + HF Trainer).

* 4-bit NF4 base model (Unsloth pre-quantized), LoRA on all attention + MLP projections.
* Completion-only loss: the loss is computed ONLY on the target tutor turn, never on the problem,
  the student's text or earlier turns (toggle `completion_only_loss: false` for the ablation).
* Examples that do not fit `max_seq_length` are dropped (count is logged) rather than silently
  truncated, so the model never learns from a cut-off context.
* Writes train_metrics.json (trainable params, wall-clock, peak VRAM, loss curves, adapter size)
  which report.py turns into the efficiency table.

Usage: python -m socratic_tutor.train_sft --config configs/sft_socratic.yaml
"""
from __future__ import annotations

# Unsloth must be imported before transformers/peft to apply its patches.
try:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError:  # pragma: no cover - only used on GPU machines
    FastLanguageModel = None

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml


# ----------------------------------------------------------------------------- tokenization
def tokenize_example(tok, prompt: List[Dict], completion: List[Dict], completion_only: bool) -> Dict:
    prompt_text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(prompt + completion, tokenize=False)
    if not full_text.startswith(prompt_text):  # should never happen with ChatML templates
        raise ValueError("chat template prefix mismatch")
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    input_ids = p_ids + c_ids
    labels = ([-100] * len(p_ids) + c_ids) if completion_only else list(input_ids)
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            att[i, :L] = 1
            lab[i, :L] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": att, "labels": lab}


def build_dataset(path, tok, max_len: int, completion_only: bool):
    from datasets import Dataset

    from .data import read_jsonl

    rows, dropped = [], 0
    for ex in read_jsonl(path):
        t = tokenize_example(tok, ex["prompt"], ex["completion"], completion_only)
        if len(t["input_ids"]) > max_len:
            dropped += 1
            continue
        rows.append(t)
    return Dataset.from_list(rows), dropped


def filter_kwargs(cls, kwargs: Dict) -> Dict:
    """Keep only kwargs the installed version of `cls` accepts (APIs drift between releases)."""
    names = {f.name for f in dataclasses.fields(cls)}
    if "eval_strategy" in kwargs and "eval_strategy" not in names and "evaluation_strategy" in names:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in names]
    if dropped:
        print("note: ignoring unsupported args:", dropped)
    return {k: v for k, v in kwargs.items() if k in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="debug: cap optimizer steps")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tc, lc = cfg["train"], cfg["lora"]

    from transformers import Trainer, TrainingArguments

    if FastLanguageModel is None:
        raise SystemExit("Unsloth is required for training (pip install unsloth) and a CUDA GPU.")
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"], max_seq_length=cfg["max_seq_length"], load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc.get("dropout", 0.0), bias="none",
        target_modules=lc["target_modules"], use_gradient_checkpointing="unsloth", random_state=tc["seed"],
        use_rslora=lc.get("use_rslora", False),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # bitsandbytes packs two 4-bit weights per byte -> count them as 2 params each
    total = sum(p.numel() * (2 if hasattr(p, "quant_state") else 1) for p in model.parameters())

    co = cfg.get("completion_only_loss", True)
    train_ds, drop_tr = build_dataset(cfg["train_file"], tok, cfg["max_seq_length"], co)
    val_ds, drop_va = build_dataset(cfg["val_file"], tok, cfg["max_seq_length"], True)  # val loss always on tutor turns
    print(f"train={len(train_ds)} (dropped {drop_tr} too long) | val={len(val_ds)} (dropped {drop_va})")

    bf16 = is_bfloat16_supported()
    steps_per_epoch = math.ceil(len(train_ds) / (tc["batch_size"] * tc["grad_accum"]))
    warmup_steps = math.ceil(tc.get("warmup_ratio", 0.03) * steps_per_epoch * tc["epochs"])
    targs = TrainingArguments(**filter_kwargs(TrainingArguments, dict(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["grad_accum"],
        num_train_epochs=tc["epochs"],
        max_steps=args.max_steps or -1,
        learning_rate=tc["lr"],
        lr_scheduler_type=tc.get("scheduler", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=tc.get("weight_decay", 0.0),
        optim="adamw_8bit",
        fp16=not bf16, bf16=bf16,
        logging_steps=tc.get("logging_steps", 10),
        eval_strategy="steps", eval_steps=tc.get("eval_steps", 50),
        save_strategy="steps", save_steps=tc.get("eval_steps", 50), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        group_by_length=True, report_to="none", seed=tc["seed"], remove_unused_columns=False,
    )))
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=PadCollator(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    hours = (time.time() - t0) / 3600
    final_eval = trainer.evaluate()

    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    adapter_mb = sum(f.stat().st_size for f in adapter_dir.glob("*.safetensors")) / 2**20
    if cfg.get("save_merged", False):  # needed as the starting point for DPO
        model.save_pretrained_merged(str(out / "merged"), tok, save_method="merged_16bit")

    metrics = {
        "run": out.name, "model_name": cfg["model_name"], "lora_r": lc["r"], "lora_alpha": lc["alpha"],
        "completion_only_loss": co, "train_file": cfg["train_file"],
        "n_train": len(train_ds), "n_val": len(val_ds), "dropped_too_long": drop_tr,
        "trainable_params": trainable, "total_params": total, "trainable_frac": trainable / total,
        "train_hours": hours, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0), "final_eval_loss": final_eval.get("eval_loss"),
        "final_eval_ppl": float(torch.exp(torch.tensor(final_eval["eval_loss"]))) if "eval_loss" in final_eval else None,
        "adapter_mb": adapter_mb, "log_history": trainer.state.log_history,
    }
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2, default=str))


if __name__ == "__main__":
    main()
