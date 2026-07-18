#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/FSC147}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fsc147-mae-pretrain}"

python pretrain_mae_fsc147.py \
  --image_dir "${DATA_ROOT}/images_384_VarV2" \
  --data_split_file "${DATA_ROOT}/Train_Test_Val_FSC_147.json" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 500 \
  --batch_size 8 \
  --blr 1.5e-4 \
  --warmup_epochs 10 \
  --weight_decay 0.05 \
  "$@"

