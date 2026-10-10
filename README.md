# 🎓 Socratic Math Tutor: QLoRA Fine-Tuning

**Teaching a 1.5B LLM to tutor instead of tell.** I fine-tuned Qwen2.5-1.5B-Instruct with QLoRA on real teacher–student
dialogues (MathDial). The goal is a tutor that helps a student find and fix *their own* mistake rather than handing over the
answer. I measured that goal directly with a leak-aware evaluation suite: single-turn quality, an LLM judge checked against
human labels, multi-turn conversations with a simulated student, and a capability-retention check.

> All numbers in this README come from `results/RESULTS.md`, which `python -m socratic_tutor.report` generates from saved
> model outputs. Run the pipeline (below) to fill them in.

---

## 1. Problem

A general LLM asked to "tutor" usually **solves the problem for the student**. That's useful for homework, but bad for learning.
A Socratic tutor has to do three things at once:

1. **Diagnose** the student's specific misconception (not just re-solve the problem),
2. **Guide** with one targeted hint or question at a time,
3. **Never leak** the final answer before the student reaches it.

Small models are cheap enough to deploy per student, but they are the worst at (3). This project asks a direct question:
*can parameter-efficient fine-tuning on a single free GPU make a 1.5B model a better Socratic tutor than prompting, and even
better than a 5× larger prompted model, without losing its own math ability?*

## 2. Data: MathDial + pedagogy-aware curation

[MathDial](https://huggingface.co/datasets/eth-nlped/mathdial) (Macina et al., EMNLP Findings 2023) has ~2.9k tutoring
dialogues grounded in GSM8K problems. In each one, a student with a real misconception (LLM-simulated) is tutored by a human
teacher, and every teacher turn is labelled with a move: `focus`, `probing`, `telling`, or `generic`.

| Step | Detail |
|---|---|
| Example construction | One example per teacher turn: *prompt* = system (problem) + student's incorrect solution + dialogue so far; *target* = that teacher turn |
| **Socratic curation** | Drop teacher turns labelled `telling`, and turns that state the final answer before the student did (deterministic numeric check) |
| Leak-free splits | Train/val split **by problem id**. Evaluation uses MathDial's *test* problems, so no test problem is ever seen in training |
| Ablation data | `raw` variant keeps every teacher turn, to measure what curation buys |

Even the human teachers "tell" fairly often (see the move distribution in `results/RESULTS.md`). Imitating them naively would
teach the model to give answers away. That is the motivation for curation.

## 3. Method

```
MathDial ──► parse + move labels ──► Socratic filter ──► QLoRA SFT (completion-only loss)
                                                           │
                                                           ├─► (optional) DPO: chosen = human Socratic turn,
                                                           │      rejected = model turn that leaks / lectures
                                                           ▼
                                        inference: solve-then-guard decoding (no gold answer used)
```

* **QLoRA SFT.** `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (NF4), LoRA r=16 / α=32 on all attention + MLP projections
  (~1.2% of params trainable), 2 epochs, effective batch 32, cosine LR 2e-4, 8-bit AdamW, gradient checkpointing.
  Fits a single 16 GB T4.
* **Completion-only loss.** Loss is computed only on the target tutor turn. The v1 model trained on full sequences and showed
  a large train/val loss gap (1.7 vs 3.5). Ablation `sft_full_loss` quantifies this choice.
* **Solve-then-guard decoding.** At inference the model first solves the problem privately with its LoRA adapter *disabled*
  (base-model math skill). It then re-samples (up to 3×) any tutor turn containing that privately-computed answer. It never sees
  the gold answer, so the comparison stays fair.
* **DPO (optional).** Preference pairs are built automatically. The SFT adapter is merged into the base first, so the DPO
  reference model is exactly the SFT policy.

## 4. Evaluation

| Axis | Metric | How |
|---|---|---|
| Answer leakage ↓ | % turns stating the final answer before the student | deterministic numeric match; undefined if the answer appears in the problem |
| Pedagogy / correctness / error-targeting ↑ | 1–5 | LLM-as-judge rubric (Qwen2.5-7B-Instruct 4-bit by default, or any OpenAI-compatible API) |
| Tutoring move mix | % focus / probing / telling / generic | judge, using MathDial's taxonomy |
| **Judge validity** | accuracy, Cohen's κ | judge move labels vs **human** MathDial labels on gold teacher turns |
| Preference | win / loss vs base | pairwise judge, run in **both orders**; inconsistent verdicts count as ties (removes position bias) |
| **Multi-turn outcome** ↑ | student self-corrects *without* being told | up to 6-turn dialogues with an LLM-simulated student holding the original misconception, plus a no-tutor "please double-check" control |
| Capability retention | GSM8K accuracy (solve mode) | does Socratic tuning damage the model's own math? |
| Efficiency | trainable params, train hours, peak VRAM, adapter MB, tokens/s | logged automatically |
| Statistics | 95% bootstrap CIs; paired bootstrap p-values | `stats.py` |

**Systems compared:** prompted base (zero-shot, few-shot) · Qwen2.5-7B-Instruct prompted (5× larger) · SFT on raw data
(ablation) · SFT with full-sequence loss (ablation) · **SFT on Socratic data** · **SFT + guard** · SFT→DPO · human teacher (reference).

## 5. Results

The full tables with CIs are in [`results/RESULTS.md`](results/RESULTS.md). Figures: `results/fig_leak_vs_pedagogy.png`,
`results/fig_training_curves.png`.

| System | Leak ↓ | Pedagogy ↑ | Win/Loss vs base | Sim. student self-corrects ↑ | GSM8K |
|---|---|---|---|---|---|
| Qwen2.5-1.5B prompted | _run_ | _run_ | — | _run_ | _run_ |
| Qwen2.5-7B prompted | _run_ | _run_ | _run_ | — | — |
| SFT, raw data (ablation) | _run_ | _run_ | _run_ | — | — |
| **SFT, Socratic data** | _run_ | _run_ | _run_ | _run_ | _run_ |
| **SFT + guard** | _run_ | _run_ | _run_ | _run_ | — |
| Human teacher | _run_ | _run_ | _run_ | — | — |

_(After running the pipeline, copy the key rows from `results/RESULTS.md` into this table.)_

**Known limitations (worth stating in interviews):** the judge is from the same model family as the tutor (self-preference
risk; mitigated by validating against human labels and by an optional external API judge). The student simulator is itself
an LLM. The leak metric is numeric and can miss verbal leaks or flag small coincidental numbers; the judge's `reveals_answer`
and `telling` rates complement it.

## 6. Reproduce

```bash
# GPU (Kaggle "T4 x2" free tier, or Colab): see notebooks/kaggle_pipeline.ipynb
pip install unsloth && pip install -r requirements-train.txt
bash scripts/run_pipeline.sh            # data → train → ablations → dpo → eval → judge → sim → gsm8k → report
# or stage by stage:  bash scripts/run_pipeline.sh data train eval judge report

# local demo (Mac MPS / CPU / CUDA)
pip install -r requirements.txt
streamlit run app.py
```

Tests (CPU, no model downloads): `python -m pytest -q`.

## 7. Repository layout

```
app.py                      Streamlit demo (problem-grounded tutoring, leak guard, base-model comparison)
configs/                    sft_socratic / sft_raw / sft_full_loss / dpo / eval  (YAML)
scripts/run_pipeline.sh     resumable end-to-end pipeline (uses both T4s for ablations)
notebooks/kaggle_pipeline.ipynb
src/socratic_tutor/
  data.py                   MathDial parsing, Socratic curation, leak-free splits, eval items
  answer_utils.py           numeric answer extraction + leak detection
  prompts.py                tutor / few-shot / solver / judge / student-simulator prompts
  train_sft.py              Unsloth QLoRA SFT, completion-only loss, efficiency logging
  dpo.py                    automatic preference pairs + DPO
  generation.py, guard.py   batched HF/PEFT generation; solve-then-guard decoding
  eval_generate.py          single-turn generation for all systems
  judge.py                  rubric, pairwise (order-debiased), judge validation
  simulate.py               multi-turn simulated-student evaluation (+ no-tutor control)
  gsm8k_eval.py             capability retention
  stats.py, report.py       bootstrap CIs, significance, tables, figures, auto-filled resume bullets
socratic_model/             v1 adapter (original run, kept for comparison in the app)
tests/                      unit tests
```
