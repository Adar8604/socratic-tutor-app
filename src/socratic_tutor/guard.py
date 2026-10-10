"""Solve-then-guard decoding: an inference-time answer-leak filter that never uses the gold answer.

1. The model privately solves each problem with its LoRA adapter disabled (base-model math skill).
2. Any tutor turn that contains that privately-solved answer (before the student said it) is
   re-sampled, up to `guard_max_resamples` times.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .answer_utils import extract_predicted_answer, is_leak
from .generation import chat_generate
from .prompts import SOLVER_SYSTEM


def solve_problems(model, tok, problems: Sequence[str], gcfg: Dict) -> Dict[str, Optional[float]]:
    uniq = sorted(set(problems))
    outs = chat_generate(
        model, tok,
        [[{"role": "system", "content": SOLVER_SYSTEM}, {"role": "user", "content": p}] for p in uniq],
        max_new_tokens=gcfg.get("solver_max_new_tokens", 384), temperature=0.0,
        batch_size=gcfg["batch_size"], disable_adapter=True,
    )
    return {p: extract_predicted_answer(o) for p, o in zip(uniq, outs)}


def guard_resample(model, tok, convs: List[List[Dict]], responses: List[str], solved: List[Optional[float]],
                   student_texts: List[List[str]], problems: List[str], gcfg: Dict, seed: int = 0):
    """Re-sample responses that contain the privately-solved answer. Returns (responses, n_resamples)."""
    responses = list(responses)
    resamples = [0] * len(responses)
    for attempt in range(gcfg.get("guard_max_resamples", 3)):
        bad = [i for i in range(len(responses))
               if is_leak(responses[i], solved[i], student_texts[i], problems[i])]
        if not bad:
            break
        new = chat_generate(
            model, tok, [convs[i] for i in bad], max_new_tokens=gcfg["max_new_tokens"],
            temperature=max(gcfg["temperature"], 0.8), top_p=gcfg["top_p"],
            batch_size=gcfg["batch_size"], seed=seed + 1000 * (attempt + 1),
        )
        for i, r in zip(bad, new):
            responses[i] = r
            resamples[i] += 1
    return responses, resamples
