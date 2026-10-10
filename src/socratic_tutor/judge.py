"""Stage 2 of evaluation: LLM-as-judge.

* Rubric scoring of every generated tutor turn: tutoring move (MathDial taxonomy), answer reveal,
  pedagogy / correctness / error-targeting on 1-5 scales.
* Pairwise preference vs. a reference system, judged in BOTH orders to cancel position bias
  (a win only counts when the verdict is consistent across orders, otherwise it is a tie).
* Judge validation: the judge's move label on the human teacher turns is compared with MathDial's
  human move annotations (accuracy + Cohen's kappa) -- reported in the results.

Backends: "hf" (local model, default Qwen2.5-7B-Instruct in 4-bit -- fits a free T4) or "openai"
(any OpenAI-compatible endpoint; set JUDGE_API_KEY / JUDGE_BASE_URL / judge.model in the config).

Usage: python -m socratic_tutor.judge --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .data import read_jsonl, write_jsonl
from .prompts import JUDGE_PAIRWISE, JUDGE_RUBRIC, JUDGE_SYSTEM, format_history

MOVES = ("focus", "probing", "telling", "generic")


# ----------------------------------------------------------------------------- parsing
def parse_rubric(text: str) -> Dict:
    out = {"move": None, "reveals_answer": None, "pedagogy": None, "correctness": None,
           "error_targeting": None, "parse_ok": False}
    m = re.search(r"\{.*\}", text or "", re.S)
    obj = None
    if m:
        blob = m.group(0)
        pyfix = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", blob))
        for candidate in (blob, pyfix, pyfix.replace("'", '"')):
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
    if obj is None:  # regex fallback for slightly malformed JSON
        obj = {}
        for k in ("pedagogy", "correctness", "error_targeting"):
            mm = re.search(rf'["\']?{k}["\']?\s*:\s*(\d)', text or "")
            if mm:
                obj[k] = int(mm.group(1))
        mm = re.search(r'["\']?move["\']?\s*:\s*["\']?(\w+)', text or "")
        if mm:
            obj["move"] = mm.group(1)
        mm = re.search(r'["\']?reveals_answer["\']?\s*:\s*["\']?(true|false)', text or "", re.I)
        if mm:
            obj["reveals_answer"] = mm.group(1).lower() == "true"
    move = str(obj.get("move", "")).lower().strip()
    out["move"] = move if move in MOVES else None
    ra = obj.get("reveals_answer")
    out["reveals_answer"] = ra if isinstance(ra, bool) else (str(ra).lower() == "true" if ra is not None else None)
    for k in ("pedagogy", "correctness", "error_targeting"):
        try:
            v = int(obj.get(k))
            out[k] = v if 1 <= v <= 5 else None
        except (TypeError, ValueError):
            pass
    out["parse_ok"] = out["move"] is not None and all(out[k] is not None for k in ("pedagogy", "correctness", "error_targeting"))
    return out


def parse_pairwise(text: str) -> Optional[str]:
    t = (text or "").strip().strip('".* ').lower()
    if t.startswith("tie") or "tie" in t[:20]:
        return "tie"
    m = re.search(r"\b(a|b)\b", t)
    if m:
        return m.group(1).upper()
    return None


# ----------------------------------------------------------------------------- backends
class HFJudge:
    def __init__(self, cfg):
        from .generation import ModelSpec, load_model

        spec = ModelSpec(name="judge", base=cfg["model"], load_in_4bit=cfg.get("load_in_4bit", True))
        self.model, self.tok = load_model(spec)
        self.bs = cfg.get("batch_size", 8)

    def __call__(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        from .generation import chat_generate

        convs = [[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": p}] for p in prompts]
        return chat_generate(self.model, self.tok, convs, max_new_tokens=max_new_tokens, temperature=0.0,
                             batch_size=self.bs)


class OpenAIJudge:
    def __init__(self, cfg):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ.get("JUDGE_API_KEY"), base_url=os.environ.get("JUDGE_BASE_URL"))
        self.model = cfg["model"]

    def __call__(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        outs = []
        for p in prompts:
            for attempt in range(5):
                try:
                    r = self.client.chat.completions.create(
                        model=self.model, temperature=0, max_tokens=max_new_tokens,
                        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": p}])
                    outs.append(r.choices[0].message.content or "")
                    break
                except Exception as e:  # rate limits etc.
                    if attempt == 4:
                        print("judge error:", e)
                        outs.append("")
                    time.sleep(2 ** attempt)
        return outs


def make_judge(cfg):
    return OpenAIJudge(cfg) if cfg.get("backend") == "openai" else HFJudge(cfg)


# ----------------------------------------------------------------------------- prompts
def history_for(item) -> str:
    return format_history(item["messages"])


def rubric_prompt(item, response: str) -> str:
    return JUDGE_RUBRIC.format(problem=item["problem"], solution=item["solution"], answer=_fmt(item["answer"]),
                               student_solution=item["student_solution"], history=history_for(item),
                               response=response)


def pairwise_prompt(item, a: str, b: str) -> str:
    return JUDGE_PAIRWISE.format(problem=item["problem"], answer=_fmt(item["answer"]),
                                 student_solution=item["student_solution"], history=history_for(item), a=a, b=b)


def _fmt(x):
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


def combine_orders(v_ab: Optional[str], v_ba: Optional[str]) -> str:
    """v_ab: verdict with system as A; v_ba: verdict with system as B. Returns win/loss/tie for the system."""
    s1 = {"A": "win", "B": "loss"}.get(v_ab, "tie")
    s2 = {"B": "win", "A": "loss"}.get(v_ba, "tie")
    return s1 if s1 == s2 else "tie"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--systems", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    jcfg = cfg["judge"]
    root = Path(cfg["output_dir"])
    items = {it["id"]: it for it in read_jsonl(cfg["eval_items"])}
    gen_dir, out_dir = root / "generations", root / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = sorted(p.stem for p in gen_dir.glob("*.jsonl"))
    if args.systems:
        systems = [s for s in systems if s in set(args.systems.split(","))]
    judge = None

    for s in systems:
        path = out_dir / f"{s}.rubric.jsonl"
        if path.exists() and not args.overwrite:
            continue
        judge = judge or make_judge(jcfg)
        gens = [g for g in read_jsonl(gen_dir / f"{s}.jsonl") if g["id"] in items]
        raw = judge([rubric_prompt(items[g["id"]], g["response"]) for g in gens], jcfg.get("max_new_tokens", 120))
        rows = [{"id": g["id"], "system": s, **parse_rubric(r), "raw": r} for g, r in zip(gens, raw)]
        write_jsonl(path, rows)
        ok = sum(r["parse_ok"] for r in rows) / max(len(rows), 1)
        print(f"[judge rubric] {s}: parse_ok={ok:.2%}")

    ref = cfg.get("pairwise_reference", "base_zeroshot")
    if (gen_dir / f"{ref}.jsonl").exists():
        ref_gen = {g["id"]: g["response"] for g in read_jsonl(gen_dir / f"{ref}.jsonl")}
        for s in systems:
            if s == ref:
                continue
            path = out_dir / f"{s}.pairwise_vs_{ref}.jsonl"
            if path.exists() and not args.overwrite:
                continue
            judge = judge or make_judge(jcfg)
            gens = [g for g in read_jsonl(gen_dir / f"{s}.jsonl") if g["id"] in ref_gen and g["id"] in items]
            p_ab = [pairwise_prompt(items[g["id"]], g["response"], ref_gen[g["id"]]) for g in gens]
            p_ba = [pairwise_prompt(items[g["id"]], ref_gen[g["id"]], g["response"]) for g in gens]
            v_ab = judge(p_ab, 8)
            v_ba = judge(p_ba, 8)
            rows = [{"id": g["id"], "system": s, "reference": ref,
                     "outcome": combine_orders(parse_pairwise(a), parse_pairwise(b)), "raw_ab": a, "raw_ba": b}
                    for g, a, b in zip(gens, v_ab, v_ba)]
            write_jsonl(path, rows)
            print(f"[judge pairwise] {s} vs {ref} done")


if __name__ == "__main__":
    main()
