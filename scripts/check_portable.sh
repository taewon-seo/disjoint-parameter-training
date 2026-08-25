#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import copy
from pathlib import Path
import torch

required = [
    "dpt/backbone.py",
    "dpt/jrdb.py",
    "dpt/objectives.py",
    "dpt/runtime.py",
    "dpt/training.py",
    "dpt/merging.py",
    "train_backbone.py",
    "train_dpt.py",
    "evaluate_jrdb.py",
    "sparse_merging.py",
    "assets/teaser.png",
    "assets/method.png",
    "configs/pretrain.yaml",
    "configs/plan_finetune.yaml",
    "configs/pred_finetune.yaml",
    "configs/dpt.yaml",
    "data/jrdb_2dbox/train",
    "data/jrdb_2dbox/val",
    "data/jrdb_2dbox/test",
    "checkpoints/pretrained_model.pth.tar",
    "checkpoints/dpt_plan_model.pth.tar",
    "checkpoints/dpt_pred_model.pth.tar",
    "checkpoints/plan_finetuned_model.pth.tar",
    "checkpoints/pred_finetuned_model.pth.tar",
    "checkpoints/dpt_sparse_merged_model_k1.pt",
]

missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))

from dpt.backbone import create_model
from dpt.jrdb import create_dataset
from dpt.runtime import create_logger

logger = create_logger("")
ds = create_dataset("jrdb_2dbox", logger, split="val", track_size=21)
checkpoint_paths = sorted(Path("checkpoints").glob("*.pt*"))
checkpoints = {
    path.name: torch.load(path, map_location="cpu", weights_only=True)
    for path in checkpoint_paths
}

for name, checkpoint in checkpoints.items():
    config = copy.deepcopy(checkpoint["config"])
    config["DEVICE"] = "cpu"
    model = create_model(config, logger)
    model.load_state_dict(checkpoint["model"], strict=True)

base_checkpoint = checkpoints["pretrained_model.pth.tar"]
config = copy.deepcopy(base_checkpoint["config"])
config["DEVICE"] = "cpu"
model = create_model(config, logger)
model.load_state_dict(base_checkpoint["model"], strict=True)
model.eval()
with torch.no_grad():
    output = model(torch.zeros(1, 9, 18, 4), torch.zeros(1, 9))
assert output.shape == (1, 21, 9, 2), output.shape

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("JRDB val samples:", len(ds))
print("strictly loaded checkpoints:", len(checkpoints))
print("model forward shape:", tuple(output.shape))
print("portable check: OK")
PY
