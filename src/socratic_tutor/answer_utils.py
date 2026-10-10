"""Numeric answer extraction and answer-leakage detection.

Leakage is the core safety metric of a Socratic tutor: a tutor turn "leaks" when it states the
final numeric answer of the problem *before the student has produced it themselves*.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional

# Matches 1,234.5 | -3 | .5 | 3/4 ; currency/percent signs are stripped before matching.
_NUM_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:/\d+)?(?![\w])")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?[\d,]*\.?\d+(?:/\d+)?)", re.I
)
_FINAL_ANSWER_TAG_RE = re.compile(r"FINAL\s*ANSWER\s*[:=]?\s*\$?\s*(-?[\d,]*\.?\d+(?:/\d+)?)", re.I)


def _to_float(tok: str) -> Optional[float]:
    tok = tok.replace(",", "").strip()
    if not tok or tok in {"-", ".", "-."}:
        return None
    try:
        if "/" in tok:
            num, den = tok.split("/", 1)
            if not num or not den or float(den) == 0:
                return None
            return float(num) / float(den)
        return float(tok)
    except ValueError:
        return None


def extract_numbers(text: str) -> List[float]:
    """All numbers that appear in `text`, in order (handles 1,000 / $5 / 3.5 / 3/4 / 40%)."""
    if not text:
        return []
    cleaned = text.replace("$", " ").replace("%", " ")
    out = []
    for m in _NUM_RE.finditer(cleaned):
        tok = m.group(0)
        if not re.search(r"\d", tok):
            continue
        val = _to_float(tok)
        if val is not None and math.isfinite(val):
            out.append(val)
    return out


def numbers_equal(a: float, b: float, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def final_answer_from_solution(solution: str) -> Optional[float]:
    """Gold answer from a GSM8K/MathDial worked solution ('#### 42' if present, else last number)."""
    if not solution:
        return None
    if "####" in solution:
        tail = solution.split("####")[-1]
        nums = extract_numbers(tail)
        if nums:
            return nums[0]
    nums = extract_numbers(solution)
    return nums[-1] if nums else None


def extract_predicted_answer(text: str) -> Optional[float]:
    """Model's final answer: 'FINAL ANSWER: x' > 'the answer is x' > '#### x' > last number."""
    if not text:
        return None
    for regex in (_FINAL_ANSWER_TAG_RE, _ANSWER_IS_RE):
        ms = list(regex.finditer(text))
        if ms:
            val = _to_float(ms[-1].group(1))
            if val is not None:
                return val
    if "####" in text:
        nums = extract_numbers(text.split("####")[-1])
        if nums:
            return nums[0]
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def contains_number(text: str, value: Optional[float]) -> bool:
    if value is None:
        return False
    return any(numbers_equal(n, value) for n in extract_numbers(text))


def is_leak(
    tutor_text: str,
    answer: Optional[float],
    student_texts: Iterable[str] = (),
    problem_text: str = "",
) -> Optional[bool]:
    """True if the tutor states the final answer before the student did.

    Returns None (undefined) when the answer already appears in the problem statement, because a
    number match is then uninformative. Returns False when the student already said the answer
    (confirming a correct student answer is good tutoring, not leakage).
    """
    if answer is None:
        return None
    if contains_number(problem_text, answer):
        return None
    if any(contains_number(s, answer) for s in student_texts):
        return False
    return contains_number(tutor_text, answer)


FINAL_ANSWER_PATTERN = _FINAL_ANSWER_TAG_RE
