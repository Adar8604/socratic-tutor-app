"""Capability-retention check: does Socratic fine-tuning hurt the model's own math solving?

Each system answers GSM8K test questions under a neutral "solve it" system prompt (greedy decoding);
accuracy is exact match of the final number. A tutor that can no longer solve problems is a bad tutor.

Usage: python -m socratic_tutor.gsm8k_eval --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
import yaml

from .answer_utils import extract_predicted_answer, final_answer_from_solution, numbers_equal
from .data import write_jsonl
from .generation import ModelSpec, chat_generate, load_model
from .prompts import SOLVER_SYSTEM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    gcfg = cfg["gsm8k"]
    out_dir = Path(cfg["output_dir"]) / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    src = gcfg.get("source", "openai/gsm8k")
    if Path(src).exists():  # local jsonl with {question, answer}
        ds = list(load_dataset("json", data_files=src, split="train"))
    else:
        ds = list(load_dataset(src, "main", split="test"))
    random.Random(gcfg.get("seed", 0)).shuffle(ds)
    ds = ds[: gcfg["n"]]
    gold = [final_answer_from_solution(r["answer"]) for r in ds]
    convs = [[{"role": "system", "content": SOLVER_SYSTEM}, {"role": "user", "content": r["question"]}] for r in ds]

    for name in gcfg["systems"]:
        path = out_dir / f"{name}.jsonl"
        if path.exists() and not args.overwrite:
            continue
        spec = ModelSpec.from_dict(name, cfg["systems"][name])
        if not spec.available():
            print(f"[gsm8k] skip {name}")
            continue
        model, tok = load_model(spec)
        outs = chat_generate(model, tok, convs, max_new_tokens=gcfg.get("max_new_tokens", 512), temperature=0.0,
                             batch_size=cfg["generation"]["batch_size"])
        rows = []
        for r, o, g in zip(ds, outs, gold):
            p = extract_predicted_answer(o)
            rows.append({"question": r["question"], "gold": g, "pred": p, "correct": p is not None and numbers_equal(p, g),
                         "output": o})
        write_jsonl(path, rows)
        print(f"[gsm8k] {name}: acc={sum(r['correct'] for r in rows)/len(rows):.2%}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
