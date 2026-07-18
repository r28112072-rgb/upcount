#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/CARPK/test}"
CHECKPOINT="${CHECKPOINT:-weights/upcount_carpk_best_epoch816.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/carpk-test}"

python test_carpk.py \
  --architecture_version v6 \
  --resume "${CHECKPOINT}" \
  --data_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --protocol "CARPK fine-tuned" \
  "$@"

