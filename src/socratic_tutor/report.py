"""Stage 4: aggregate everything into results/metrics.json, results/RESULTS.md, figures, and
auto-filled resume bullets (results/resume_bullets.md). Every number in the bullets is computed here
from saved outputs -- nothing is typed by hand.

Usage: python -m socratic_tutor.report --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import yaml

from .answer_utils import is_leak
from .data import read_jsonl
from .stats import cohen_kappa, mean, paired_bootstrap_pvalue, rouge_l, summarize

PRETTY = {
    "human_teacher": "Human teacher (MathDial gold)",
    "base_zeroshot": "Qwen2.5-1.5B-Instruct, zero-shot prompt",
    "base_fewshot": "Qwen2.5-1.5B-Instruct, few-shot prompt",
    "qwen7b_zeroshot": "Qwen2.5-7B-Instruct, zero-shot prompt (5x larger)",
    "sft_raw": "QLoRA SFT, all teacher turns (ablation)",
    "sft_full_loss": "QLoRA SFT, Socratic data, full-sequence loss (ablation)",
    "sft_socratic": "QLoRA SFT, Socratic-curated data (ours)",
    "sft_socratic_guard": "QLoRA SFT + solve-then-guard decoding (ours)",
    "dpo": "QLoRA SFT -> DPO (ours)",
    "control_recheck": "No tutor (\"please double-check\" control)",
}
ORDER = list(PRETTY)


def _order(names):
    return sorted(names, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def ci_pct(s: Optional[Dict]):
    if not s or s.get("mean") is None:
        return "–"
    return f"{100 * s['mean']:.1f}% [{100 * s['ci_low']:.1f}, {100 * s['ci_high']:.1f}]"


def ci_num(s: Optional[Dict]):
    if not s or s.get("mean") is None:
        return "–"
    return f"{s['mean']:.2f} [{s['ci_low']:.2f}, {s['ci_high']:.2f}]"


# ----------------------------------------------------------------------------- aggregation
def single_turn_metrics(root: Path, items: Dict, ref: str) -> Dict:
    gen_dir, judge_dir = root / "generations", root / "judge"
    out = {}
    per_item_leak = {}
    for path in sorted(gen_dir.glob("*.jsonl")):
        s = path.stem
        gens = [g for g in read_jsonl(path) if g["id"] in items]
        if not gens:
            continue
        leaks = [is_leak(g["response"], items[g["id"]]["answer"], items[g["id"]]["student_texts"], items[g["id"]]["problem"]) for g in gens]
        leaks = [None if l is None else float(l) for l in leaks]
        per_item_leak[s] = dict(zip([g["id"] for g in gens], leaks))
        m = {
            "n": len(gens),
            "leak_rate": summarize(leaks),
            "question_rate": summarize([float("?" in g["response"]) for g in gens]),
            "words": summarize([len(g["response"].split()) for g in gens]),
            "rougeL_vs_human": summarize([rouge_l(g["response"], items[g["id"]]["gold_response"]) for g in gens]),
        }
        meta_p = gen_dir / f"{s}.meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            m["sec_per_response"] = meta.get("sec_per_response")
            m["tokens_per_sec"] = meta.get("tokens_per_sec")
            m["guard_resample_rate"] = meta.get("guard_resample_rate")
            m["guard_solver_accuracy"] = meta.get("guard_solver_accuracy")
        rub_p = judge_dir / f"{s}.rubric.jsonl"
        if rub_p.exists():
            rub = [r for r in read_jsonl(rub_p) if r["id"] in items]
            for k in ("pedagogy", "correctness", "error_targeting"):
                m[f"judge_{k}"] = summarize([r[k] for r in rub])
            moves = [r["move"] for r in rub if r["move"]]
            m["judge_move_dist"] = {k: v / max(len(moves), 1) for k, v in Counter(moves).items()}
            m["judge_telling_rate"] = summarize([None if r["move"] is None else float(r["move"] == "telling") for r in rub])
            m["judge_reveal_rate"] = summarize([None if r["reveals_answer"] is None else float(r["reveals_answer"]) for r in rub])
            m["judge_parse_ok"] = mean([float(r["parse_ok"]) for r in rub])
        pw_p = judge_dir / f"{s}.pairwise_vs_{ref}.jsonl"
        if pw_p.exists():
            pw = read_jsonl(pw_p)
            c = Counter(r["outcome"] for r in pw)
            n = max(len(pw), 1)
            m["pairwise_vs_ref"] = {"reference": ref, "win": c["win"] / n, "loss": c["loss"] / n, "tie": c["tie"] / n,
                                    "n": len(pw)}
        out[s] = m
    # significance of leak-rate difference vs reference (paired bootstrap)
    if ref in per_item_leak:
        for s, d in per_item_leak.items():
            if s == ref:
                continue
            ids = [i for i in d if i in per_item_leak[ref]]
            out[s]["leak_p_vs_ref"] = paired_bootstrap_pvalue([d[i] for i in ids], [per_item_leak[ref][i] for i in ids])
    return out


def judge_validation(root: Path, items: Dict) -> Optional[Dict]:
    p = root / "judge" / "human_teacher.rubric.jsonl"
    if not p.exists():
        return None
    rub = {r["id"]: r for r in read_jsonl(p)}
    pairs = [(items[i]["gold_move"], rub[i]["move"]) for i in rub if i in items and items[i].get("gold_move")]
    pairs = [(h, j) for h, j in pairs if j is not None]
    if not pairs:
        return None
    # binary "telling vs not" is what our leak story depends on
    tb = [(h == "telling", j == "telling") for h, j in pairs]
    return {
        "n": len(pairs),
        "move_accuracy_4way": sum(h == j for h, j in pairs) / len(pairs),
        "move_kappa_4way": cohen_kappa([h for h, _ in pairs], [j for _, j in pairs]),
        "telling_kappa_binary": cohen_kappa([h for h, _ in tb], [j for _, j in tb]),
        "confusion": {f"{h}->{j}": c for (h, j), c in Counter(pairs).most_common()},
    }


def simulation_metrics(root: Path) -> Dict:
    out = {}
    for p in sorted((root / "simulation").glob("*.jsonl")):
        rows = read_jsonl(p)
        out[p.stem] = {
            "n": len(rows),
            "success": summarize([float(r["success"]) for r in rows]),
            "leak": summarize([float(r["leaked"]) for r in rows]),
            "success_no_leak": summarize([float(r["success_no_leak"]) for r in rows]),
            "turns_to_success": mean([r["turns"] for r in rows if r["success"]]),
        }
    return out


def gsm8k_metrics(root: Path) -> Dict:
    return {p.stem: summarize([float(r["correct"]) for r in read_jsonl(p)]) for p in sorted((root / "gsm8k").glob("*.jsonl"))}


def training_metrics(train_root: Path) -> Dict:
    out = {}
    for p in sorted(train_root.glob("*/train_metrics.json")):
        out[p.parent.name] = json.loads(p.read_text())
    return out


# ----------------------------------------------------------------------------- rendering
def render_markdown(M: Dict) -> str:
    L = ["# Results\n", "_Auto-generated by `python -m socratic_tutor.report`. 95% bootstrap CIs in brackets._\n"]
    d = M.get("data")
    if d:
        c = d["counts"]
        L += ["## Data\n",
              f"- MathDial train dialogues: {d['train']['dialogues']} | teacher turns: {d['train']['teacher_turns']}",
              f"- Human teacher move distribution (train): " + ", ".join(f"{k} {pct(v)}" for k, v in d["train"]["move_distribution"].items()),
              f"- Human teacher turns that state the final answer early: {pct(d['train']['gold_turn_answer_leak_rate'])}",
              f"- SFT examples — raw: {c.get('sft_raw_train')} | Socratic-curated: {c.get('sft_socratic_train')} "
              f"({pct(1 - c.get('sft_socratic_train', 0) / max(c.get('sft_raw_train', 1), 1))} removed)",
              f"- Held-out eval items (unseen problems): {c.get('eval_items')}\n"]
    T = M.get("training") or {}
    if T:
        L += ["## Training efficiency\n", "| Run | Trainable params | Train time | Peak VRAM | Final eval loss | Adapter size |",
              "|---|---|---|---|---|---|"]
        for k, t in T.items():
            L.append(f"| {k} | {t.get('trainable_params', 0) / 1e6:.1f}M ({pct(t.get('trainable_frac'), 2)}) | "
                     f"{num(t.get('train_hours'))} h | {num(t.get('peak_vram_gb'))} GB | {num(t.get('final_eval_loss'), 3)} | "
                     f"{num(t.get('adapter_mb'), 1)} MB |")
        L.append("")
    S = M.get("single_turn") or {}
    if S:
        L += ["## Single-turn tutoring quality (held-out MathDial test problems)\n",
              "Leak = tutor states the final answer before the student did (deterministic). Telling = judge labels the "
              "turn as giving away the solution. Pedagogy / Correctness / Error-targeting = LLM-judge 1–5. "
              f"Win/Loss = position-debiased pairwise preference vs `{M.get('reference')}`.\n",
              "| System | Leak ↓ | Telling ↓ | Pedagogy ↑ | Correctness ↑ | Error-targeting ↑ | Asks question | Words | Win / Loss | ROUGE-L |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in _order(S):
            m = S[s]
            pw = m.get("pairwise_vs_ref")
            L.append(f"| {PRETTY.get(s, s)} | {ci_pct(m['leak_rate'])} | {ci_pct(m.get('judge_telling_rate'))} | "
                     f"{ci_num(m.get('judge_pedagogy'))} | {ci_num(m.get('judge_correctness'))} | {ci_num(m.get('judge_error_targeting'))} | "
                     f"{pct(m['question_rate']['mean'])} | {num(m['words']['mean'], 0)} | "
                     f"{(pct(pw['win'], 0) + ' / ' + pct(pw['loss'], 0)) if pw else '–'} | {num(m['rougeL_vs_human']['mean'])} |")
        L.append("")
        sig = [f"`{s}`: p={m['leak_p_vs_ref']:.4f}" for s, m in S.items() if m.get("leak_p_vs_ref") is not None]
        if sig:
            L += ["Paired bootstrap test of leak-rate difference vs reference: " + "; ".join(sig) + "\n"]
    J = M.get("judge_validation")
    if J:
        L += ["## Judge validation (vs. MathDial human move labels)\n",
              f"- n = {J['n']} human teacher turns; 4-way move accuracy {pct(J['move_accuracy_4way'])}, "
              f"Cohen's κ = {num(J['move_kappa_4way'])}; binary telling-vs-not κ = {num(J['telling_kappa_binary'])}\n"]
    SIM = M.get("simulation") or {}
    if SIM:
        L += ["## Multi-turn evaluation with a simulated student\n",
              "| Tutor | Student solves ↑ | Tutor leaked answer ↓ | Solves *without* leak ↑ | Avg. turns to solve |",
              "|---|---|---|---|---|"]
        for s in _order(SIM):
            m = SIM[s]
            L.append(f"| {PRETTY.get(s, s)} | {ci_pct(m['success'])} | {ci_pct(m['leak'])} | {ci_pct(m['success_no_leak'])} | {num(m['turns_to_success'], 1)} |")
        L.append("")
    G = M.get("gsm8k") or {}
    if G:
        L += ["## Capability retention (GSM8K, solve mode)\n", "| Model | GSM8K accuracy |", "|---|---|"]
        for s in _order(G):
            L.append(f"| {PRETTY.get(s, s)} | {ci_pct(G[s])} |")
        L.append("")
    if S:
        L += ["## Latency (single T4 GPU, batched)\n", "| System | sec / response | tokens / sec | guard re-sample rate |", "|---|---|---|---|"]
        for s in _order(S):
            m = S[s]
            if m.get("sec_per_response") is not None:
                L.append(f"| {PRETTY.get(s, s)} | {num(m['sec_per_response'], 3)} | {num(m.get('tokens_per_sec'), 0)} | {pct(m.get('guard_resample_rate'))} |")
        L.append("")
    return "\n".join(L)


def _g(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def resume_bullets(M: Dict, primary: str, guard: str, base: str) -> str:
    """Numbers AND verbs come from the data: a claim is only phrased as an improvement if it is one."""
    S, SIM, G, T, D, J = (M.get(k) or {} for k in ("single_turn", "simulation", "gsm8k", "training", "data", "judge_validation"))
    tr = T.get(primary) or {}
    P = lambda x, d=0: "[RUN]" if x is None else f"{100 * x:.{d}f}%"
    N = lambda x, d=1: "[RUN]" if x is None else f"{x:.{d}f}"
    leak = lambda s: _g(S, s, "leak_rate", "mean")
    best = guard if leak(guard) is not None else primary
    b_leak, p_leak, best_leak = leak(base), leak(primary), leak(best)
    rel = None if (b_leak in (None, 0) or best_leak is None) else (b_leak - best_leak) / b_leak
    removed = None
    if D.get("counts"):
        c = D["counts"]
        removed = 1 - c["sft_socratic_train"] / max(c["sft_raw_train"], 1)

    # bullet 1 - what was built
    n_train = tr.get("n_train")
    b1 = (f"Fine-tuned Qwen2.5-1.5B-Instruct into a Socratic math tutor with QLoRA (4-bit NF4, LoRA r={tr.get('lora_r', 16)} on all "
          f"attention/MLP projections, {P(tr.get('trainable_frac'), 2)} of params trainable) on "
          f"{f'{n_train:,}' if isinstance(n_train, int) else '[RUN]'} curated MathDial tutoring turns with completion-only loss; "
          f"trained in {N(tr.get('train_hours'))} h on one free T4 GPU ({N(tr.get('peak_vram_gb'))} GB peak VRAM) and deployed as a "
          f"Streamlit app with a {N(tr.get('adapter_mb'), 0)} MB adapter.")

    # bullet 2 - data curation + guard -> leakage
    if rel is not None and rel > 0:
        verb = f"cutting answer-leak rate from {P(b_leak, 1)} (prompted base) to {P(best_leak, 1)} ({P(rel)} relative reduction"
    else:
        verb = f"changing answer-leak rate from {P(b_leak, 1)} (prompted base) to {P(best_leak, 1)} (no reduction"
    p = _g(S, best, "leak_p_vs_ref")
    sig = f", p={p:.3f}" if isinstance(p, float) and p < 0.05 else ""
    b2 = (f"Designed a pedagogy-aware data-curation filter (removing 'telling'/answer-revealing teacher turns, {P(removed)} of data) "
          f"and a solve-then-guard decoding step, {verb}{sig}; ablations: SFT alone {P(p_leak, 1)}, SFT on unfiltered data "
          f"{P(leak('sft_raw'), 1)}).")

    # bullet 3 - evaluation harness
    win, loss = _g(S, primary, "pairwise_vs_ref", "win"), _g(S, primary, "pairwise_vs_ref", "loss")
    ped_b, ped_p = _g(S, base, "judge_pedagogy", "mean"), _g(S, primary, "judge_pedagogy", "mean")
    ped_verb = "rose" if (ped_b is not None and ped_p is not None and ped_p > ped_b) else "went"
    big = "qwen7b_zeroshot"
    ped_big = _g(S, big, "judge_pedagogy", "mean")
    vs7b = ""
    if ped_big is not None and ped_p is not None:
        rel_word = "outperforming" if ped_p > ped_big else ("matching" if abs(ped_p - ped_big) < 0.1 else "approaching")
        vs7b = f", {rel_word} a 5x-larger prompted Qwen2.5-7B ({N(ped_big, 2)}/5, leak {P(leak(big), 1)})"
    n_items = _g(S, primary, "n") or _g(D, "counts", "eval_items")
    b3 = (f"Built a reproducible evaluation harness on {n_items or '[RUN]'} held-out problems: deterministic leak metric, LLM-as-judge "
          f"rubric validated against human move labels (Cohen's κ={N(_g(J, 'move_kappa_4way'), 2)}), position-debiased pairwise "
          f"preference and bootstrap CIs; judge pedagogy score {ped_verb} {N(ped_b, 2)}→{N(ped_p, 2)}/5 with a {P(win)}/{P(loss)} "
          f"win/loss rate vs the base model{vs7b}.")

    # bullet 4 - multi-turn simulation + capability retention
    sim_sys = best if _g(SIM, best) else primary
    sim_p = _g(SIM, sim_sys, "success_no_leak", "mean")
    sim_b, sim_c = _g(SIM, base, "success_no_leak", "mean"), _g(SIM, "control_recheck", "success_no_leak", "mean")
    g_b, g_p = _g(G, base, "mean"), _g(G, primary, "mean")
    if g_b is not None and g_p is not None:
        d = 100 * (g_p - g_b)
        retain = (f"while keeping GSM8K solve accuracy within {abs(d):.1f} pts of the base model ({P(g_p, 1)} vs {P(g_b, 1)})"
                  if d > -2 else f"at a {abs(d):.1f}-pt GSM8K solve-accuracy cost ({P(g_p, 1)} vs {P(g_b, 1)}), quantifying the pedagogy/skill trade-off")
    else:
        retain = "while tracking GSM8K solve accuracy for capability regression ([RUN])"
    n_dlg = _g(SIM, sim_sys, "n")
    b4 = (f"Ran {n_dlg or '[RUN]'} multi-turn dialogues against an LLM-simulated student holding real MathDial misconceptions: the "
          f"student fixed its own error without being told the answer in {P(sim_p)} of dialogues vs {P(sim_b)} with the prompted "
          f"base and {P(sim_c)} with a no-tutor control, {retain}.")
    return "# Resume bullets (auto-filled from results)\n\n" + "\n".join(f"- {b}" for b in (b1, b2, b3, b4)) + "\n"


def make_figures(M: Dict, out: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    S = M.get("single_turn") or {}
    pts = [(s, m["leak_rate"]["mean"], _g(m, "judge_pedagogy", "mean")) for s, m in S.items()]
    pts = [p for p in pts if p[1] is not None and p[2] is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for s, x, y in pts:
            ax.scatter(100 * x, y, s=60, color="#2a6fdb" if s.startswith(("sft", "dpo")) else "#888888")
            ax.annotate(s, (100 * x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Answer-leak rate (%)  ← better")
        ax.set_ylabel("Judge pedagogy score (1–5)  ↑ better")
        ax.set_title("Socratic quality vs. answer leakage")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_leak_vs_pedagogy.png", dpi=160)
        plt.close(fig)
    T = M.get("training") or {}
    if T:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for run, t in T.items():
            hist = t.get("log_history", [])
            tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
            ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
            if tr:
                ax.plot(*zip(*tr), label=f"{run} train", alpha=0.8)
            if ev:
                ax.plot(*zip(*ev), "o--", label=f"{run} val")
        ax.set_xlabel("step")
        ax.set_ylabel("loss (tutor-turn tokens)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_training_curves.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["output_dir"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = {it["id"]: it for it in read_jsonl(cfg["eval_items"])}
    ref = cfg.get("pairwise_reference", "base_zeroshot")

    M = {"reference": ref}
    stats_p = Path(cfg["eval_items"]).parent / "data_stats.json"
    if stats_p.exists():
        M["data"] = json.loads(stats_p.read_text())
    M["training"] = training_metrics(Path(cfg.get("train_root", "outputs")))
    M["single_turn"] = single_turn_metrics(root, items, ref) if (root / "generations").exists() else {}
    M["judge_validation"] = judge_validation(root, items)
    M["simulation"] = simulation_metrics(root) if (root / "simulation").exists() else {}
    M["gsm8k"] = gsm8k_metrics(root) if (root / "gsm8k").exists() else {}

    slim = json.loads(json.dumps(M, default=str))
    for t in slim.get("training", {}).values():
        t.pop("log_history", None)
    (out / "metrics.json").write_text(json.dumps(slim, indent=2))
    (out / "RESULTS.md").write_text(render_markdown(M))
    rep = cfg.get("report", {})
    (out / "resume_bullets.md").write_text(resume_bullets(M, rep.get("primary", "sft_socratic"),
                                                          rep.get("guard", "sft_socratic_guard"), ref))
    make_figures(M, out)
    print((out / "RESULTS.md").read_text())
    print((out / "resume_bullets.md").read_text())


if __name__ == "__main__":
    main()
