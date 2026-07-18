#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/CARPK/train}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-weights/upcount_fsc147_best_epoch432.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/carpk-finetune}"

python train_carpk.py \
  --architecture_version v6 \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --data_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 1000 \
  --batch_size 8 \
  --blr 2e-4 \
  --min_lr 0 \
  --warmup_epochs 10 \
  --weight_decay 0.05 \
  --patience 0 \
  --train_backbone \
  "$@"

