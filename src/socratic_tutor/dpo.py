"""Optional stage: Direct Preference Optimization on top of the SFT tutor (QLoRA again).

Preference pairs (built automatically, no human labelling):
  chosen   = the human teacher's Socratic turn (from the curated training data)
  rejected = a model-sampled turn for the same context that is NON-Socratic, i.e. it
             (a) states the final answer before the student did, or
             (b) is a long lecture (> `lecture_words` words) that asks the student nothing.
Samples come from the SFT model itself (on-policy) and from the prompted base model.

Reference model = the SFT model: we merge the SFT adapter into the base weights (train_sft.py with
`save_merged: true`) and train a fresh LoRA on top, so TRL's "adapter disabled" reference is exactly SFT.

Usage:
  python -m socratic_tutor.dpo pairs --config configs/dpo.yaml
  python -m socratic_tutor.dpo train --config configs/dpo.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def build_pairs(cfg):
    import gc

    import torch

    from .answer_utils import is_leak
    from .data import read_jsonl, write_jsonl
    from .generation import ModelSpec, chat_generate, load_model

    pc = cfg["pairs"]
    rows = read_jsonl(pc["source_file"])
    random.Random(pc.get("seed", 0)).shuffle(rows)
    rows = [r for r in rows if r.get("answer") is not None][: pc["n_contexts"]]
    candidates = {r["id"]: [] for r in rows}
    for gname, gspec in pc["generators"].items():
        spec = ModelSpec.from_dict(gname, gspec)
        if not spec.available():
            print(f"[pairs] skip generator {gname}")
            continue
        model, tok = load_model(spec)
        for k in range(pc.get("samples_per_context", 2)):
            outs = chat_generate(model, tok, [r["prompt"] for r in rows], max_new_tokens=160,
                                 temperature=pc.get("temperature", 1.0), top_p=0.95,
                                 batch_size=pc.get("batch_size", 16), seed=1000 * k)
            for r, o in zip(rows, outs):
                candidates[r["id"]].append(o)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    pairs = []
    reasons = {"leak": 0, "lecture": 0}
    for r in rows:
        rejected = None
        for o in candidates[r["id"]]:
            if is_leak(o, r["answer"], r["student_texts"], r["problem"]):
                rejected, why = o, "leak"
                break
            if len(o.split()) > pc.get("lecture_words", 70) and "?" not in o:
                rejected, why = o, "lecture"
        if rejected is None:
            continue
        reasons[why] += 1
        pairs.append({"prompt": r["prompt"], "chosen": r["completion"],
                      "rejected": [{"role": "assistant", "content": rejected}], "reason": why})
    random.Random(1).shuffle(pairs)
    n_val = max(int(len(pairs) * 0.05), 1)
    out = Path(pc["out_dir"])
    write_jsonl(out / "dpo_val.jsonl", pairs[:n_val])
    write_jsonl(out / "dpo_train.jsonl", pairs[n_val:])
    stats = {"contexts": len(rows), "pairs": len(pairs), **reasons}
    (out / "dpo_pairs_stats.json").write_text(json.dumps(stats, indent=2))
    print(stats)


def train(cfg):
    from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

    PatchDPOTrainer()
    import time

    import torch
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    from .train_sft import filter_kwargs

    tc, lc = cfg["train"], cfg["lora"]
    out = Path(cfg["output_dir"])
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(model_name=cfg["sft_merged_model"],
                                                   max_seq_length=cfg["max_seq_length"], load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=0.0, bias="none",
        target_modules=lc["target_modules"], use_gradient_checkpointing="unsloth", random_state=tc["seed"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pdir = Path(cfg["pairs"]["out_dir"])
    ds = load_dataset("json", data_files={"train": str(pdir / "dpo_train.jsonl"), "val": str(pdir / "dpo_val.jsonl")})
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "chosen", "rejected")])
    bf16 = is_bfloat16_supported()
    args = DPOConfig(**filter_kwargs(DPOConfig, dict(
        output_dir=str(out / "checkpoints"), beta=tc["beta"], learning_rate=tc["lr"],
        per_device_train_batch_size=tc["batch_size"], per_device_eval_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["grad_accum"], num_train_epochs=tc["epochs"],
        lr_scheduler_type="cosine",
        warmup_steps=max(1, len(ds["train"]) // (tc["batch_size"] * tc["grad_accum"]) // 10), optim="adamw_8bit", fp16=not bf16, bf16=bf16,
        max_length=cfg["max_seq_length"], max_prompt_length=cfg["max_seq_length"] - 192,
        max_completion_length=192, logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="no", report_to="none", seed=tc["seed"],
    )))
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds["train"],
                         eval_dataset=ds["val"], processing_class=tok)
    t0 = time.time()
    trainer.train()
    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    metrics = {"run": out.name, "lora_r": lc["r"], "n_train": len(ds["train"]), "train_hours": (time.time() - t0) / 3600,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, "beta": tc["beta"],
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "adapter_mb": sum(f.stat().st_size for f in adapter_dir.glob("*.safetensors")) / 2**20,
               "log_history": trainer.state.log_history}
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pairs", "train"])
    ap.add_argument("--config", default="configs/dpo.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    {"pairs": build_pairs, "train": train}[a.stage](cfg)


if __name__ == "__main__":
    sys.exit(main())
