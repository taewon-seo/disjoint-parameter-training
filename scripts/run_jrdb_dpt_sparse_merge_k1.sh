#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${1:-dpt_sparse_merge_k1}"

cd "$(dirname "$0")/.."

python sparse_merging.py \
  --preset dpt \
  --base_model checkpoints/pretrained_model.pth.tar \
  --plan_model checkpoints/dpt_plan_model.pth.tar \
  --pred_model checkpoints/dpt_pred_model.pth.tar \
  --base_output_dir outputs/merged \
  --exp_name "${EXP_NAME}" \
  --k_values 1
