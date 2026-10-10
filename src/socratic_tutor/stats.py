"""Small, dependency-free statistics helpers: bootstrap CIs, paired bootstrap tests, Cohen's kappa, ROUGE-L."""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


def _clean(xs):
    return [float(x) for x in xs if x is not None]


def mean(xs) -> Optional[float]:
    xs = _clean(xs)
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(xs: Sequence, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    xs = _clean(xs)
    if not xs:
        return None, None, None
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return sum(xs) / n, lo, hi


def paired_bootstrap_pvalue(a: Sequence, b: Sequence, n_boot: int = 5000, seed: int = 0) -> Optional[float]:
    """Two-sided p-value for H0: mean(a) == mean(b), on paired observations (None pairs dropped)."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    diffs = [x - y for x, y in pairs]
    obs = sum(diffs) / len(diffs)
    centered = [d - obs for d in diffs]
    rng = random.Random(seed)
    n = len(diffs)
    extreme = 0
    for _ in range(n_boot):
        m = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        if abs(m) >= abs(obs) - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_boot + 1)


def cohen_kappa(y1: Sequence, y2: Sequence) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(y1, y2) if a is not None and b is not None]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(c1[k] * c2.get(k, 0) for k in c1) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _lcs(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    l = _lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def summarize(xs, seed: int = 0) -> Dict:
    m, lo, hi = bootstrap_ci(xs, seed=seed)
    return {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(_clean(xs))}
