#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/evaluation

eval_one() {
  local name="$1"
  local ckpt="$2"
  python evaluate_jrdb.py \
    --ckpt "${ckpt}" \
    --split test \
    --modality traj+2dbox \
    --log_file "results/evaluation/${name}_traj_2dbox.log"
}

eval_one dpt_plan_model checkpoints/dpt_plan_model.pth.tar
eval_one dpt_pred_model checkpoints/dpt_pred_model.pth.tar
eval_one dpt_sparse_merged_model_k1 checkpoints/dpt_sparse_merged_model_k1.pt
