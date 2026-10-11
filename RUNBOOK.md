# Runbook: everything you need to do, step by step

Total effort: ~1 h of your time plus ~12–14 h of unattended Kaggle GPU time (inside the free 30 h/week quota).

---

## Phase 0: One-time setup (≈20 min)

1. **Kaggle account.** Phone-verify it (Settings → Phone verification). Without this you get no GPU and no internet.
2. **Hugging Face token (optional, recommended).** Go to huggingface.co → Settings → Access Tokens → create a *read* token.
   In Kaggle: open the notebook → *Add-ons → Secrets* → add `HF_TOKEN` and switch it on for the notebook.
3. **Put the code on GitHub** (from the project folder on your Mac):
   ```bash
   cd ~/Desktop/Projects/socratic-tutor-app
   git init && git add . && git commit -m "Socratic Math Tutor v2: QLoRA + leak-aware evaluation"
   # create an EMPTY public repo named socratic-tutor-app on github.com, then:
   git remote add origin https://github.com/<YOUR_USERNAME>/socratic-tutor-app.git
   git branch -M main && git push -u origin main
   ```
   `.gitignore` already keeps out `.venv/`, the 1.8 GB checkpoint/cache folders, `data/` and `outputs/`.
   - Prefer a private repo? Upload the folder (without `.venv` and `socratic_model/`) as a Kaggle **Dataset** named
     `socratic-tutor-app-code` instead, and add that dataset as notebook input. The notebook detects it automatically.
4. **Sanity check locally (optional):** `python -m pytest -q` should report 14 passed.

---

## Phase 1: Kaggle Part 1: data + training (≈5–6 h GPU)

1. Kaggle → *Code → New Notebook → File → Import Notebook* → upload `notebooks/kaggle_pipeline.ipynb`.
2. Right panel → *Session options*: **Accelerator = GPU T4 x2**, **Internet = On**.
3. In the second cell, set:
   ```python
   STAGES = "data train ablations"
   RESUME_DIR = ""
   REPO_URL = "https://github.com/<YOUR_USERNAME>/socratic-tutor-app.git"
   ```
4. **Optional quick test (5 min, interactive):** run only the first 6 cells interactively, to confirm the install and tests pass,
   then stop the session.
5. Click **Save Version** → **Save & Run All (Commit)**. Under *Advanced settings* make sure output is saved
   ("Always save output"). Close the tab. It runs in the background.
6. When it finishes, open the version → **Output** tab. You should see `socratic-tutor-app/outputs/sft_socratic/adapter`,
   `outputs/sft_raw`, `outputs/sft_full_loss`, `outputs/sft_socratic/merged`, and `data/`.
   - Check `train_metrics.json`. The validation loss should be falling, and there should be no gap like v1's 1.7 vs 3.5.

## Phase 2: Kaggle Part 2: DPO + evaluation + judge (≈3–4 h GPU)

1. Open the same notebook (*Edit*). Right panel → **Add Input** → *Your Work / Notebooks* → choose this notebook's Part 1 version
   (the output appears under `/kaggle/input/<notebook-name>/`).
2. Set:
   ```python
   STAGES = "dpo eval judge"
   RESUME_DIR = "auto"   # finds the attached previous output automatically
   ```
3. **Save Version → Save & Run All (Commit)** again.
4. Check that `outputs/eval/generations/*.jsonl` and `outputs/eval/judge/*.jsonl` exist in the Output tab. The log prints
   `parse_ok=` per system. If it is below ~90%, tell me and I'll tighten the judge prompt.

## Phase 3: Kaggle Part 3: simulation + GSM8K + report (≈2–3 h GPU)

1. Remove the Part 1 input and add the **Part 2** version's output instead.
2. Set `STAGES = "sim gsm8k report"` and keep `RESUME_DIR = "auto"`.
3. **Save & Run All (Commit)**.
4. Output tab → download **`socratic_results.zip`** (~100 MB: adapters, metrics, eval outputs, results, figures).

> If any part fails or times out, just commit again with the same settings. Every stage skips work that's already
> finished, as long as `RESUME_DIR` points at the latest successful output.

---

## Phase 4: Back on your Mac (≈20 min)

1. Unzip `socratic_results.zip` into the project folder. This creates `results/` and `outputs/*/adapter`.
2. Try the demo: `source .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py`.
   Pick *"v2 – Socratic-curated SFT"*, paste a problem, and try to make it give the answer away. Toggle "Also show base model reply".
   (The DPO option in the app needs the 3 GB `outputs/sft_socratic/merged`. Download it from the Part 1 output only if you want it.)
3. Open `results/RESULTS.md` and `results/resume_bullets.md`.
   - Copy the bullets into `RESUME_BULLETS.md` and your resume.
   - Copy the key rows into the results table in `README.md`, and add the two figures from `results/`.
   - **Read every number.** If a comparison is weak (for example the 7B baseline wins on pedagogy, or a CI overlaps), drop or
     soften that clause.
4. Commit and push: `git add results README.md RESUME_BULLETS.md && git commit -m "Add results" && git push`.

## Phase 5: Make it look like an Applied Scientist project (optional, ≈30 min, high value)

1. **Publish the adapter** to the Hugging Face Hub and put the link on your resume:
   `huggingface-cli login && huggingface-cli upload <you>/socratic-qwen2.5-1.5b-lora outputs/sft_socratic/adapter`
2. **Read 20 transcripts yourself** (`outputs/eval/simulation/sft_socratic_guard.jsonl`) and write 3–4 sentences of
   error analysis in the README: where it still fails, and what the judge gets wrong. Interviewers love this.
3. **External judge (optional):** if you have an OpenAI/Groq/etc. key, set `judge.backend: openai` in `configs/eval.yaml`,
   export `JUDGE_API_KEY` (and `JUDGE_BASE_URL` if not OpenAI), and re-run `judge report`. This removes the
   "same-family judge" caveat.
4. Deploy the Streamlit app (Hugging Face Spaces, CPU) for a live demo link.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in training | In `configs/sft_*.yaml` set `batch_size: 2`, `grad_accum: 16` |
| Unsloth / TRL import or argument errors | Copy the error text to me. Library APIs change often; usually a one-line fix |
| `RESUME_DIR` assertion fails | Path is wrong. Copy it exactly from the Input panel (it ends in `/socratic-tutor-app`) |
| Part runs out of time | Commit again with the same `STAGES` and the latest output as input; it resumes |
| Very low judge `parse_ok` | Tell me. I'll adjust the prompt or switch to the API judge |
