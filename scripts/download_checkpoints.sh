#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

base_url="${DPT_CHECKPOINT_BASE_URL:-https://github.com/taewon-seo/disjoint-parameter-training/releases/latest/download}"
checkpoint_dir="checkpoints"
checkpoint_files=(
  "pretrained_model.pth.tar"
  "dpt_plan_model.pth.tar"
  "dpt_pred_model.pth.tar"
  "dpt_sparse_merged_model_k1.pt"
  "plan_finetuned_model.pth.tar"
  "pred_finetuned_model.pth.tar"
)

mkdir -p "$checkpoint_dir"

for filename in "${checkpoint_files[@]}"; do
  target="$checkpoint_dir/$filename"
  if [[ -f "$target" ]]; then
    echo "Found $target"
    continue
  fi

  echo "Downloading $filename"
  curl --fail --location --retry 3 \
    --output "$target.part" "$base_url/$filename"
  mv "$target.part" "$target"
done

echo "Checkpoints are ready in $checkpoint_dir/"
