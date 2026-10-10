"""Stage 1 of evaluation: generate the next tutor turn for every held-out item, for every system.

Systems (see configs/eval.yaml): prompted base model (zero-/few-shot), SFT adapters (raw vs. Socratic
curated data), SFT + solve-then-guard decoding, optional DPO adapter, optional larger prompted model,
and the human teacher's gold turn as a reference row.

Solve-then-guard decoding: the system first solves the problem privately (with LoRA disabled, i.e. the
base model's math ability), then re-samples any tutor turn that contains its own predicted answer
(up to `guard_max_resamples` times). It never sees the gold answer, so the comparison is fair.

Usage: python -m socratic_tutor.eval_generate --config configs/eval.yaml [--systems a,b]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from .answer_utils import numbers_equal
from .data import read_jsonl, write_jsonl
from .generation import ModelSpec, chat_generate, load_model
from .guard import guard_resample, solve_problems
from .prompts import tutor_system


def item_messages(item, fewshot: bool):
    msgs = [dict(m) for m in item["messages"]]
    msgs[0] = {"role": "system", "content": tutor_system(item["problem"], fewshot=fewshot)}
    return msgs


def run_system(spec: ModelSpec, items, gcfg, out_dir: Path):
    model, tok = load_model(spec)
    convs = [item_messages(it, spec.prompt == "fewshot") for it in items]
    responses, stats = chat_generate(
        model, tok, convs, max_new_tokens=gcfg["max_new_tokens"], temperature=gcfg["temperature"],
        top_p=gcfg["top_p"], batch_size=gcfg["batch_size"], seed=gcfg["seed"], return_stats=True,
    )
    resamples = [0] * len(items)
    solved = [None] * len(items)
    if spec.guard:
        pred = solve_problems(model, tok, [it["problem"] for it in items], gcfg)
        solved = [pred[it["problem"]] for it in items]
        responses, resamples = guard_resample(
            model, tok, convs, responses, solved, [it["student_texts"] for it in items],
            [it["problem"] for it in items], gcfg, seed=gcfg["seed"])
        # how often the private solve was right (guard can only catch leaks it can predict)
        stats["guard_solver_accuracy"] = sum(
            s is not None and numbers_equal(s, it["answer"]) for s, it in zip(solved, items)) / len(items)
    rows = [{"id": it["id"], "system": spec.name, "response": r, "guard_resamples": n,
             "guard_solved_answer": s}
            for it, r, n, s in zip(items, responses, resamples, solved)]
    write_jsonl(out_dir / f"{spec.name}.jsonl", rows)
    meta = {"system": spec.name, "base": spec.base, "adapter": spec.adapter, "prompt": spec.prompt,
            "guard": spec.guard, **stats, "guard_resample_rate": sum(1 for n in resamples if n) / len(items)}
    (out_dir / f"{spec.name}.meta.json").write_text(json.dumps(meta, indent=2))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{spec.name}] done: {meta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--systems", default=None, help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    items = read_jsonl(cfg["eval_items"])[: args.limit or cfg.get("n_items")]
    out_dir = Path(cfg["output_dir"]) / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Human reference row (the MathDial teacher's actual next turn)
    write_jsonl(out_dir / "human_teacher.jsonl",
                [{"id": it["id"], "system": "human_teacher", "response": it["gold_response"]} for it in items])

    wanted = set(args.systems.split(",")) if args.systems else None
    for name, d in cfg["systems"].items():
        if wanted and name not in wanted:
            continue
        spec = ModelSpec.from_dict(name, d)
        if not spec.available():
            print(f"[{name}] skipped: missing {spec.adapter or spec.base}")
            continue
        if (out_dir / f"{name}.jsonl").exists() and not args.overwrite:
            print(f"[{name}] exists, skipping (use --overwrite)")
            continue
        run_system(spec, items, cfg["generation"], out_dir)


if __name__ == "__main__":
    main()
