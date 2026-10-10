"""Stage 3 of evaluation: multi-turn tutoring with an LLM-simulated student (MathDial-style).

A student simulator (default Qwen2.5-7B-Instruct, 4-bit) is given a persona, the problem and its
*incorrect* solution. Each tutor system then converses with it for up to `max_tutor_turns` turns.

Per dialogue we record:
  success        - the student's last stated FINAL ANSWER equals the gold answer
  leaked         - the tutor stated the gold answer before the student did (at any turn)
  success_no_leak- success AND no leak  (the student actually solved it themselves)
  turns          - tutor turns used until success (or the cap)

A "control_recheck" tutor that only says "Can you double-check your work?" measures how often the
student fixes its own error without real tutoring, so gains are attributable to the tutor.

Usage: python -m socratic_tutor.simulate --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from .answer_utils import FINAL_ANSWER_PATTERN, extract_numbers, is_leak, numbers_equal
from .data import read_jsonl, write_jsonl
from .generation import ModelSpec, chat_generate, load_model
from .guard import guard_resample, solve_problems
from .prompts import CONTROL_TUTOR_MESSAGE, STUDENT_OPENING, STUDENT_SIM_SYSTEM, tutor_system


def student_final_answer(text: str):
    m = list(FINAL_ANSWER_PATTERN.finditer(text or ""))
    if not m:
        return None
    nums = extract_numbers(m[-1].group(1))
    return nums[0] if nums else None


def run_dialogues(tutor, student, problems, scfg, gcfg, control=False, guard=False):
    """tutor/student: (model, tok) tuples. Runs all dialogues in lock-step batches."""
    solved = solve_problems(tutor[0], tutor[1], [p["problem"] for p in problems], gcfg) if guard else {}
    states = []
    for p in problems:
        opening = STUDENT_OPENING.format(solution=p["student_solution"])
        states.append({
            "id": p["qid"], "answer": p["answer"], "problem": p["problem"],
            "tutor_msgs": [{"role": "system", "content": tutor_system(p["problem"])}, {"role": "user", "content": opening}],
            "student_msgs": [{"role": "system", "content": STUDENT_SIM_SYSTEM.format(
                profile=p["student_profile"] or "a middle-school student", problem=p["problem"],
                solution=p["student_solution"])}],
            "student_said": [p["student_solution"]], "done": False, "success": False, "leaked": False,
            "turns": 0, "last_answer": None,
        })
    for turn in range(scfg["max_tutor_turns"]):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        if control:
            tutor_out = [CONTROL_TUTOR_MESSAGE] * len(active)
        else:
            tutor_out = chat_generate(tutor[0], tutor[1], [s["tutor_msgs"] for s in active],
                                      max_new_tokens=gcfg["max_new_tokens"], temperature=gcfg["temperature"],
                                      top_p=gcfg["top_p"], batch_size=gcfg["batch_size"], seed=gcfg["seed"] + turn)
            if guard:
                tutor_out, _ = guard_resample(
                    tutor[0], tutor[1], [s["tutor_msgs"] for s in active], tutor_out,
                    [solved.get(s["problem"]) for s in active], [s["student_said"] for s in active],
                    [s["problem"] for s in active], gcfg, seed=gcfg["seed"] + 10 * turn)
        for s, t in zip(active, tutor_out):
            s["turns"] += 1
            s["tutor_msgs"].append({"role": "assistant", "content": t})
            s["student_msgs"].append({"role": "user", "content": t})
            if is_leak(t, s["answer"], s["student_said"], s["problem"]):
                s["leaked"] = True
        stud_out = chat_generate(student[0], student[1], [s["student_msgs"] for s in active],
                                 max_new_tokens=scfg.get("student_max_new_tokens", 160), temperature=0.7,
                                 top_p=0.9, batch_size=gcfg["batch_size"], seed=gcfg["seed"] + 100 + turn)
        for s, r in zip(active, stud_out):
            s["tutor_msgs"].append({"role": "user", "content": r})
            s["student_msgs"].append({"role": "assistant", "content": r})
            s["student_said"].append(r)
            a = student_final_answer(r)
            if a is not None:
                s["last_answer"] = a
                if numbers_equal(a, s["answer"]):
                    s["success"], s["done"] = True, True
    return [{
        "qid": s["id"], "success": s["success"], "leaked": s["leaked"],
        "success_no_leak": s["success"] and not s["leaked"], "turns": s["turns"],
        "last_student_answer": s["last_answer"], "gold_answer": s["answer"], "transcript": s["tutor_msgs"][1:],
    } for s in states]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--systems", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    scfg, gcfg = cfg["simulation"], cfg["generation"]
    out_dir = Path(cfg["output_dir"]) / "simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    seen, problems = set(), []
    for it in read_jsonl(cfg["eval_items"]):
        if it["qid"] not in seen and not it.get("answer_in_problem"):
            seen.add(it["qid"])
            problems.append(it)
    problems = problems[: scfg["n_dialogues"]]

    student = load_model(ModelSpec(name="student", base=scfg["student_model"],
                                   load_in_4bit=scfg.get("student_load_in_4bit", True)))
    names = scfg["tutor_systems"]
    if args.systems:
        names = [n for n in names if n in set(args.systems.split(","))]
    for name in names:
        path = out_dir / f"{name}.jsonl"
        if path.exists() and not args.overwrite:
            continue
        if name == "control_recheck":
            rows = run_dialogues(None, student, problems, scfg, gcfg, control=True)
        else:
            spec = ModelSpec.from_dict(name, cfg["systems"][name])
            if not spec.available():
                print(f"[sim] skip {name}: not available")
                continue
            tutor = load_model(spec)
            rows = run_dialogues(tutor, student, problems, scfg, gcfg, guard=spec.guard)
            del tutor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_jsonl(path, rows)
        n = len(rows)
        print(f"[sim] {name}: success={sum(r['success'] for r in rows)/n:.2%} "
              f"leak={sum(r['leaked'] for r in rows)/n:.2%} success_no_leak={sum(r['success_no_leak'] for r in rows)/n:.2%}")


if __name__ == "__main__":
    main()
