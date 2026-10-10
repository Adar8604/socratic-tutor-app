# Resume: Socratic Math Tutor (QLoRA Fine-Tuning)

**Socratic Math Tutor: QLoRA Fine-Tuning** | Python, PyTorch, Hugging Face (Transformers, PEFT, TRL), Unsloth, bitsandbytes, Streamlit

- Fine-tuned Qwen2.5-1.5B-Instruct into a Socratic math tutor with QLoRA (4-bit NF4, LoRA r=16, ~1.2% of params trainable) on **[N]K** curated MathDial teacher–student tutoring turns with completion-only loss; trained on a single free T4 GPU (**[X] h, [Y] GB** peak VRAM) and deployed as a Streamlit app.
- Designed a pedagogy-aware data-curation filter (removing "telling"/answer-revealing teacher turns) and solve-then-guard decoding, cutting the tutor's answer-leak rate from **[A]%** (prompted base) to **[B]%** (**[R]%** relative reduction), with ablations isolating the effects of data curation and loss masking.
- Built a reproducible evaluation harness on **[n]** held-out problems: LLM-as-judge rubric validated against human pedagogical labels (Cohen's κ=**[k]**), position-debiased pairwise preference, and bootstrap CIs; pedagogy score rose **[p0]→[p1]/5** (**[W]%** win rate vs base), **[outperforming/matching]** a 5× larger prompted Qwen2.5-7B.
- Evaluated multi-turn tutoring with an LLM-simulated student holding real misconceptions: students self-corrected without being told the answer in **[S1]%** of dialogues vs **[S0]%** for the prompted base and **[C]%** for a no-tutor control, with GSM8K accuracy within **[d]** pts of the base model (no capability regression).

---

**How to fill this in:** after the pipeline runs, `results/resume_bullets.md` holds these same four bullets with every number
computed from your outputs. The verbs follow the data too: if a number got worse, the generated text says so instead of
claiming an improvement. Copy that file's text here. If a result comes out weak (for example the 7B baseline wins), drop that
clause rather than stretch it. Interviewers will ask about every number.

**Short version (if you only have space for 3 lines):** keep bullets 2, 3 and 4. Fold "QLoRA on a single T4" into bullet 2.
