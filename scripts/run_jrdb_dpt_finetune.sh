#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${1:-dpt_finetune}"

cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train_dpt.py \
  --cfg configs/dpt.yaml \
  --exp_name "${EXP_NAME}"
