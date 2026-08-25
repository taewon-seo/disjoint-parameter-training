import argparse
import logging
import os

import torch

from dpt.merging import load_checkpoint_model, merge_task_vectors


SPARSE_K_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0]

PRESETS = {
    "dpt": {
        "exp_name": "dpt_sparse_k_sweep",
        "base_model": "checkpoints/pretrained_model.pth.tar",
        "plan_model": "checkpoints/dpt_plan_model.pth.tar",
        "pred_model": "checkpoints/dpt_pred_model.pth.tar",
    },
    "non_dpt": {
        "exp_name": "finetuned_models_sparse_k_sweep",
        "base_model": "checkpoints/pretrained_model.pth.tar",
        "plan_model": "checkpoints/plan_finetuned_model.pth.tar",
        "pred_model": "checkpoints/pred_finetuned_model.pth.tar",
    },
}


def parse_merge_combinations(k_values, k_pairs):
    if k_pairs:
        combinations = []
        for pair in k_pairs.split(","):
            pair = pair.strip()
            if not pair:
                continue
            try:
                planner_k, forecaster_k = pair.split(":")
            except ValueError as error:
                raise ValueError(
                    f"Invalid top-k pair '{pair}'; expected planner:forecaster"
                ) from error
            combinations.append((float(planner_k), float(forecaster_k)))
        return combinations
    if k_values:
        values = [
            float(value.strip())
            for value in k_values.split(",")
            if value.strip()
        ]
        return [(value, value) for value in values]
    return [(value, value) for value in SPARSE_K_VALUES]


def format_k_tag(value):
    value = float(value)
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="dpt",
        help="Checkpoint set to merge.",
    )
    parser.add_argument("--base_model", default=None)
    parser.add_argument(
        "--plan_model", "--ego_model", dest="plan_model", default=None
    )
    parser.add_argument(
        "--pred_model", "--neighbor_model", dest="pred_model", default=None
    )
    parser.add_argument("--base_output_dir", default="outputs/merged")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument(
        "--k_values", default=None,
        help="Comma-separated shared top-k percentages, for example 1,2,5.",
    )
    parser.add_argument(
        "--k_pairs", default=None,
        help="Comma-separated planner:forecaster pairs, for example 1:0,2:1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    base_path = args.base_model or preset["base_model"]
    planner_path = args.plan_model or preset["plan_model"]
    forecaster_path = args.pred_model or preset["pred_model"]
    combinations = parse_merge_combinations(args.k_values, args.k_pairs)
    if not combinations:
        raise ValueError("At least one top-k setting is required")

    device = (
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger = logging.getLogger("sparse_merging")
    logger.addHandler(logging.NullHandler())
    base_model, base_checkpoint = load_checkpoint_model(
        base_path, device, logger
    )
    planner_model, _ = load_checkpoint_model(planner_path, device, logger)
    forecaster_model, _ = load_checkpoint_model(
        forecaster_path, device, logger
    )

    experiment_name = args.exp_name or preset["exp_name"]
    output_directory = os.path.join(args.base_output_dir, experiment_name)
    os.makedirs(output_directory, exist_ok=True)
    log_entries = []

    for planner_k, forecaster_k in combinations:
        merged_state, analysis = merge_task_vectors(
            base_model,
            planner_model,
            forecaster_model,
            device,
            planner_k,
            forecaster_k,
        )
        prefix = "dpt" if args.preset == "dpt" else "finetuned_models"
        planner_tag = format_k_tag(planner_k)
        forecaster_tag = format_k_tag(forecaster_k)
        if planner_k == forecaster_k:
            filename = f"{prefix}_sparse_merged_model_k{planner_tag}.pt"
        else:
            filename = (
                f"{prefix}_sparse_merged_model_"
                f"plan{planner_tag}_pred{forecaster_tag}.pt"
            )
        output_path = os.path.join(output_directory, filename)
        torch.save(
            {
                "model": merged_state,
                "config": base_checkpoint["config"],
                "merging_info": {
                    "base_model": os.path.basename(base_path),
                    "planner_model": os.path.basename(planner_path),
                    "forecaster_model": os.path.basename(forecaster_path),
                    "planner_top_k": planner_k,
                    "forecaster_top_k": forecaster_k,
                    "merge_strategy": "average_overlap",
                    "preset": args.preset,
                    "total_params": analysis["total_params"],
                },
            },
            output_path,
        )
        print(
            f"P-Top-K: {planner_k:.1f}% | F-Top-K: {forecaster_k:.1f}% | "
            f"Overlap: {analysis['overlap_ratio']:.2f}% | Saved: {output_path}"
        )
        log_entries.append(
            f"| {planner_k:9.1f} | {forecaster_k:17.1f} | "
            f"{analysis['overlap_ratio']:15.2f} |\n"
        )

    report = (
        f"Sparse Merging Report ({experiment_name})\n"
        f"Preset: {args.preset}\n"
        f"Base Model: {os.path.basename(base_path)}\n"
        f"Planner Model: {os.path.basename(planner_path)}\n"
        f"Forecaster Model: {os.path.basename(forecaster_path)}\n"
        + "-" * 49 + "\n"
        + f"| {'P-Top-K (%)':9} | {'F-Top-K (%)':17} | "
          f"{'Overlap (%)':15} |\n"
        + "-" * 49 + "\n"
        + "".join(log_entries)
        + "-" * 49 + "\n"
    )
    with open(
        os.path.join(output_directory, "merging_log.txt"),
        "w",
        encoding="utf-8",
    ) as log_file:
        log_file.write(report)


if __name__ == "__main__":
    main()
