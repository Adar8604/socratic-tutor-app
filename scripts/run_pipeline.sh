#!/usr/bin/env bash
# End-to-end pipeline. Designed for a Kaggle "GPU T4 x2" notebook (free tier, ~30 GPU-h/week).
# Usage:  bash scripts/run_pipeline.sh [stage ...]
#   stages: data train ablations dpo eval judge sim gsm8k report   (default: all, in order)
# Every stage is resumable: finished outputs are skipped on re-run.
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGES=${@:-data train ablations dpo eval judge sim gsm8k report}
NGPU=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
log() { echo -e "\n========== $* ==========\n"; }

for s in $STAGES; do case $s in
  data)  log "1/9 data: MathDial -> Socratic SFT data + held-out eval items"
         python -m socratic_tutor.data --source "${MATHDIAL_SOURCE:-eth-nlped/mathdial}" --out data --n-eval 400 ;;
  train) log "2/9 QLoRA SFT (main run, ~2-3 h on T4)"
         [ -f outputs/sft_socratic/train_metrics.json ] || python -m socratic_tutor.train_sft --config configs/sft_socratic.yaml ;;
  ablations)
         log "3/9 ablations: unfiltered data + full-sequence loss (2 GPUs -> run in parallel)"
         if [ "$NGPU" -ge 2 ]; then
           [ -f outputs/sft_raw/train_metrics.json ]       || CUDA_VISIBLE_DEVICES=0 python -m socratic_tutor.train_sft --config configs/sft_raw.yaml       > outputs_sft_raw.log 2>&1 &
           [ -f outputs/sft_full_loss/train_metrics.json ] || CUDA_VISIBLE_DEVICES=1 python -m socratic_tutor.train_sft --config configs/sft_full_loss.yaml > outputs_sft_full_loss.log 2>&1 &
           wait
         else
           [ -f outputs/sft_raw/train_metrics.json ]       || python -m socratic_tutor.train_sft --config configs/sft_raw.yaml
           [ -f outputs/sft_full_loss/train_metrics.json ] || python -m socratic_tutor.train_sft --config configs/sft_full_loss.yaml
         fi ;;
  dpo)   log "4/9 (optional) DPO on top of SFT"
         [ -f data/dpo/dpo_train.jsonl ] || CUDA_VISIBLE_DEVICES=0 python -m socratic_tutor.dpo pairs --config configs/dpo.yaml
         [ -f outputs/dpo/train_metrics.json ] || CUDA_VISIBLE_DEVICES=0 python -m socratic_tutor.dpo train --config configs/dpo.yaml ;;
  eval)  log "5/9 generate tutor turns for all systems"
         CUDA_VISIBLE_DEVICES=0 python -m socratic_tutor.eval_generate --config configs/eval.yaml ;;
  judge) log "6/9 LLM-as-judge (rubric + pairwise + judge validation)"
         CUDA_VISIBLE_DEVICES=0 python -m socratic_tutor.judge --config configs/eval.yaml ;;
  sim)   log "7/9 multi-turn simulated-student evaluation"
         python -m socratic_tutor.simulate --config configs/eval.yaml ;;   # uses both GPUs (student cuda:0, tutor cuda:1)
  gsm8k) log "8/9 GSM8K capability retention"
         CUDA_VISIBLE_DEVICES=${GSM8K_GPU:-0} python -m socratic_tutor.gsm8k_eval --config configs/eval.yaml ;;
  report) log "9/9 report"
         python -m socratic_tutor.report --config configs/eval.yaml --out results ;;
  *) echo "unknown stage $s"; exit 1 ;;
esac; done
