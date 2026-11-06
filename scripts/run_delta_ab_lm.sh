#!/usr/bin/env bash
# Engineered by uncoalesced
# Delta-rule A/B, LM arms (2026-11-04): slab-v4-ab-small (44.7M) from scratch on processed_v5,
# identical seed and batches; arms differ only by --delta-rule / --beta-mode.
# Expects out/locks/GPU.lock to be held by the caller. Logs: out/delta_ab/lm_<arm>.log
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
STEPS="${STEPS:-1000}"
common=(--preset slab-v4-ab-small --data data/processed_v5/train.bin --boundaries data/processed_v5/train_boundaries.npy
        --tokenizer data/tokenizer/v3_32k --batch-size 4 --grad-accum 8 --grad-checkpointing --ce-rows 2048
        --lr 1e-3 --warmup-steps 100
        --decay-start 700 --min-lr 1e-4 --differential-adamw --recurrent-lr-ratio 0.1 --seed 4242
        --max-steps "$STEPS" --log-every 10 --ckpt-every "$STEPS")
for arm in ${ARMS:-A0 A1 A2}; do
  case $arm in
    A0) extra=() ;;
    A1) extra=(--delta-rule) ;;
    A2) extra=(--delta-rule --beta-mode sig05) ;;
  esac
  echo "=== $arm $(date -Iseconds)" | tee -a out/delta_ab/runs.txt
  .venv/Scripts/python.exe train.py "${common[@]}" "${extra[@]}" --out-dir "checkpoints/delta_ab/$arm" \
      > "out/delta_ab/lm_$arm.log" 2>&1
  echo "=== $arm exit $? $(date -Iseconds)" | tee -a out/delta_ab/runs.txt
done
