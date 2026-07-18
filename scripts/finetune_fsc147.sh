#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/FSC147}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-weights/upcount_mae_fsc147_pretrain_epoch500.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fsc147-finetune}"

python train.py \
  --architecture_version v6 \
  --mae_checkpoint "${MAE_CHECKPOINT}" \
  --disable_text_conditioning \
  --train_backbone \
  --epochs 1000 \
  --batch_size 26 \
  --blr 2e-4 \
  --min_lr 0 \
  --warmup_epochs 10 \
  --weight_decay 0.05 \
  --validation_interval 1 \
  --save_interval 50 \
  --output_dir "${OUTPUT_DIR}" \
  --img_dir "${DATA_ROOT}/images_384_VarV2" \
  --gt_dir "${DATA_ROOT}/gt_density_map_adaptive_384_VarV2" \
  --class_file "${DATA_ROOT}/ImageClasses_FSC147.txt" \
  --FSC147_anno_file "${DATA_ROOT}/annotation_FSC147_384.json" \
  --FSC147_D_anno_file FSC-147-D.json \
  --data_split_file "${DATA_ROOT}/Train_Test_Val_FSC_147.json" \
  "$@"

