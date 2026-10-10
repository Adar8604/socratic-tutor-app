"""MathDial -> Socratic SFT data + held-out evaluation items.

MathDial (Macina et al., EMNLP Findings 2023; HF: eth-nlped/mathdial) contains ~2.9k teacher-student
dialogues grounded in GSM8K problems, where an LLM-simulated student holds a real misconception and
a human teacher tutors them. Every teacher turn is labelled with a pedagogical move:
focus | probing | telling | generic.

Pipeline
  1. parse each dialogue into chat turns (+ teacher move labels)
  2. one training example per teacher turn: prompt = system(problem) + student's incorrect solution
     + dialogue so far, completion = that teacher turn
  3. Socratic curation ("socratic" variant): drop teacher turns that are labelled `telling` or that
     state the final answer before the student did. The "raw" variant keeps every turn (ablation).
  4. splits are made by problem id (qid) so no test problem is ever seen in training.

Usage:  python -m socratic_tutor.data --out data --n-eval 400
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .answer_utils import contains_number, final_answer_from_solution, is_leak
from .prompts import STUDENT_OPENING, tutor_system

EOM = "|EOM|"
MOVES = ("focus", "probing", "telling", "generic")
_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .'\-]{0,40}?)\s*:\s*(.*)$", re.S)
_MOVE_RE = re.compile(r"^\s*\((focus|probing|telling|generic)\)\s*", re.I)


@dataclass
class Turn:
    role: str  # "teacher" | "student"
    text: str
    move: Optional[str] = None


@dataclass
class Dialogue:
    qid: str
    problem: str
    solution: str
    answer: Optional[float]
    student_solution: str
    student_profile: str
    turns: List[Turn] = field(default_factory=list)


# ----------------------------------------------------------------------------- parsing
def parse_conversation(conv: str) -> List[Turn]:
    turns: List[Turn] = []
    for chunk in conv.split(EOM):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _SPEAKER_RE.match(chunk)
        if m:
            speaker, text = m.group(1).strip(), m.group(2).strip()
            role = "teacher" if speaker.lower() == "teacher" else "student"
        elif turns:  # continuation without a speaker prefix
            turns[-1].text = (turns[-1].text + " " + chunk).strip()
            continue
        else:
            role, text = "student", chunk
        move = None
        if role == "teacher":
            mm = _MOVE_RE.match(text)
            if mm:
                move = mm.group(1).lower()
                text = text[mm.end():].strip()
        if not text:
            continue
        # merge consecutive same-speaker turns
        if turns and turns[-1].role == role:
            turns[-1].text = (turns[-1].text + "\n" + text).strip()
            turns[-1].move = turns[-1].move or move
        else:
            turns.append(Turn(role=role, text=text, move=move))
    return turns


def row_to_dialogue(row: Dict) -> Dialogue:
    return Dialogue(
        qid=str(row["qid"]),
        problem=row["question"].strip(),
        solution=row["ground_truth"].strip(),
        answer=final_answer_from_solution(row["ground_truth"]),
        student_solution=row["student_incorrect_solution"].strip(),
        student_profile=(row.get("student_profile") or "").strip(),
        turns=parse_conversation(row["conversation"]),
    )


def dialogue_prefix_messages(d: Dialogue, upto: int, fewshot: bool = False) -> List[Dict]:
    """Chat messages (tutor = assistant) for everything before turn index `upto`."""
    msgs = [
        {"role": "system", "content": tutor_system(d.problem, fewshot=fewshot)},
        {"role": "user", "content": STUDENT_OPENING.format(solution=d.student_solution)},
    ]
    for t in d.turns[:upto]:
        role = "assistant" if t.role == "teacher" else "user"
        if msgs[-1]["role"] == role:  # keep strict alternation
            msgs[-1]["content"] += "\n" + t.text
        else:
            msgs.append({"role": role, "content": t.text})
    return msgs


def student_texts_before(d: Dialogue, upto: int) -> List[str]:
    """Everything the student has said before turn `upto` (incl. their opening solution)."""
    return [d.student_solution] + [t.text for t in d.turns[:upto] if t.role == "student"]


def turn_leaks(d: Dialogue, i: int) -> Optional[bool]:
    return is_leak(d.turns[i].text, d.answer, student_texts_before(d, i), d.problem)


def student_reached_answer(d: Dialogue, upto: int) -> bool:
    return any(contains_number(s, d.answer) for s in student_texts_before(d, upto))


# ----------------------------------------------------------------------------- examples
def keep_turn(d: Dialogue, i: int, variant: str) -> bool:
    t = d.turns[i]
    if t.role != "teacher":
        return False
    if variant == "raw":
        return True
    if variant == "socratic":
        if t.move == "telling":
            return False
        if turn_leaks(d, i):
            return False
        return len(t.text.split()) >= 2
    raise ValueError(f"unknown variant {variant}")


def build_sft_examples(dialogues: List[Dialogue], variant: str) -> List[Dict]:
    out = []
    for d in dialogues:
        for i, t in enumerate(d.turns):
            if not keep_turn(d, i, variant):
                continue
            out.append({
                "id": f"{d.qid}_{i}",
                "qid": d.qid,
                "move": t.move,
                "answer": d.answer,
                "problem": d.problem,
                "student_texts": student_texts_before(d, i),
                "prompt": dialogue_prefix_messages(d, i),
                "completion": [{"role": "assistant", "content": t.text}],
            })
    return out


def build_eval_items(dialogues: List[Dialogue], n_max: int, seed: int) -> List[Dict]:
    """One held-out tutoring context per test dialogue: a teacher turn before the student solved it."""
    rng = random.Random(seed)
    items = []
    for d in dialogues:
        cands = [i for i, t in enumerate(d.turns) if t.role == "teacher" and not student_reached_answer(d, i)]
        if not cands or d.answer is None:
            continue
        i = rng.choice(cands)
        items.append({
            "id": f"{d.qid}_{i}",
            "qid": d.qid,
            "problem": d.problem,
            "solution": d.solution,
            "answer": d.answer,
            "student_solution": d.student_solution,
            "student_profile": d.student_profile,
            "turn_index": i,
            "messages": dialogue_prefix_messages(d, i),
            "student_texts": student_texts_before(d, i),
            "gold_response": d.turns[i].text,
            "gold_move": d.turns[i].move,
            "answer_in_problem": contains_number(d.problem, d.answer),
        })
    rng.shuffle(items)
    return items[:n_max]


def dataset_stats(dialogues: List[Dialogue]) -> Dict:
    moves = Counter()
    n_teacher = n_leak = 0
    for d in dialogues:
        for i, t in enumerate(d.turns):
            if t.role != "teacher":
                continue
            n_teacher += 1
            moves[t.move or "unlabelled"] += 1
            n_leak += bool(turn_leaks(d, i))
    return {
        "dialogues": len(dialogues),
        "teacher_turns": n_teacher,
        "move_distribution": {k: round(v / max(n_teacher, 1), 4) for k, v in moves.most_common()},
        "gold_turn_answer_leak_rate": round(n_leak / max(n_teacher, 1), 4),
    }


# ----------------------------------------------------------------------------- io
def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_mathdial(source: str = "eth-nlped/mathdial"):
    """Returns {'train': [rows], 'test': [rows]} from the HF hub or a local dir of jsonl/csv."""
    from datasets import load_dataset

    p = Path(source)
    if p.exists():
        files = {s: str(next(p.glob(f"{s}.*"))) for s in ("train", "test")}
        fmt = "csv" if files["train"].endswith(".csv") else "json"
        ds = load_dataset(fmt, data_files=files)
    else:
        ds = load_dataset(source)
    return {s: [dict(r) for r in ds[s]] for s in ("train", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="eth-nlped/mathdial")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    raw = load_mathdial(args.source)
    train_d = [row_to_dialogue(r) for r in raw["train"]]
    test_d = [row_to_dialogue(r) for r in raw["test"]]
    test_qids = {d.qid for d in test_d}
    train_d = [d for d in train_d if d.qid not in test_qids]  # paranoia: no problem overlap

    qids = sorted({d.qid for d in train_d})
    random.Random(args.seed).shuffle(qids)
    val_qids = set(qids[: int(len(qids) * args.val_frac)])
    tr = [d for d in train_d if d.qid not in val_qids]
    va = [d for d in train_d if d.qid in val_qids]

    out = Path(args.out)
    counts = {}
    for variant in ("socratic", "raw"):
        for split, ds in (("train", tr), ("val", va)):
            ex = build_sft_examples(ds, variant)
            write_jsonl(out / f"sft_{variant}_{split}.jsonl", ex)
            counts[f"sft_{variant}_{split}"] = len(ex)
    items = build_eval_items(test_d, args.n_eval, args.seed)
    write_jsonl(out / "eval_items.jsonl", items)
    counts["eval_items"] = len(items)

    stats = {"counts": counts, "train": dataset_stats(tr), "test": dataset_stats(test_d)}
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
