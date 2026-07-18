#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/FSC147}"
CHECKPOINT="${CHECKPOINT:-weights/upcount_fsc147_best_epoch432.pth}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/fsc147-${SPLIT}}"

python test.py \
  --architecture_version v6 \
  --disable_text_conditioning \
  --resume "${CHECKPOINT}" \
  --data_split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --img_dir "${DATA_ROOT}/images_384_VarV2" \
  --FSC147_anno_file "${DATA_ROOT}/annotation_FSC147_384.json" \
  --FSC147_D_anno_file FSC-147-D.json \
  --data_split_file "${DATA_ROOT}/Train_Test_Val_FSC_147.json" \
  "$@"

