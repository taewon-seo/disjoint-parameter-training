import argparse
import os
import random

import numpy as np
import torch

from dpt.runtime import create_logger, load_config
from dpt.training import run_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default="configs/dpt.yaml",
        help="Path to the DPT config file.",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        help="Experiment name for logs and checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one planner and one forecaster batch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_name = args.exp_name
    if experiment_name is None:
        experiment_name = os.path.splitext(os.path.basename(args.cfg))[0]

    config = load_config(args.cfg, exp_name=experiment_name)
    config["dry_run"] = args.dry_run

    seed = config["SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        config["DEVICE"] = "cpu"

    logger = create_logger(config["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(config)
    run_training(config, logger, experiment_name)


if __name__ == "__main__":
    main()
