import argparse
import datetime
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dpt.backbone import create_model
from dpt.jrdb import batch_process_coords, collate_batch, create_dataset
from dpt.runtime import create_logger, create_progress_bar


DEFAULT_LOG_PATH = "jrdb_eval.log"


def evaluate_model(model, dataloader, config, modality_selection='traj+2dbox'):
    output_frames = config['TRAIN']['output_track_size']
    token_count = config['MODEL'].get('token_num', 2)
    collision_threshold = config['EVAL'].get('col_threshold', 0.6)
    miss_threshold = config['EVAL'].get('miss_threshold', 0.5)

    ego_ade_sum = 0.0
    ego_fde_sum = 0.0
    ego_count = 0
    neighbor_ade_sum = 0.0
    neighbor_fde_sum = 0.0
    neighbor_count = 0
    collision_count = 0
    collision_comparisons = 0
    miss_count = 0
    valid_ego_count = 0

    model.eval()
    bar = create_progress_bar("EVAL", fill="#", max=len(dataloader))
    with torch.inference_mode():
        for joints, masks, padding_mask in dataloader:
            padding_mask = padding_mask.to(config["DEVICE"])
            in_joints, _, out_joints, _, padding_mask = batch_process_coords(
                joints,
                masks,
                padding_mask,
                config,
                modality_selection,
            )
            pred_joints = model(in_joints, padding_mask)[:, -output_frames:]

            in_joints = in_joints.cpu()
            out_joints = out_joints.cpu()
            pred_joints = pred_joints.cpu()
            padding_mask = padding_mask.cpu()
            batch_size, _, num_people, _ = pred_joints.shape

            for batch_index in range(batch_size):
                gt_ego = out_joints[batch_index, :, 0, :2]
                pred_ego = pred_joints[batch_index, :, 0, :2]

                # Match the accumulation order used for the paper results.
                sample_ade = 0.0
                for timestep in range(output_frames):
                    sample_ade += np.linalg.norm(
                        gt_ego[timestep].numpy() - pred_ego[timestep].numpy()
                    )
                ego_ade_sum += sample_ade / output_frames
                ego_fde_sum += np.linalg.norm(
                    gt_ego[-1].numpy() - pred_ego[-1].numpy()
                )
                ego_count += 1

                ego_is_valid = padding_mask[batch_index, 0] < 0.5
                ego_has_nan = (
                    torch.isnan(gt_ego).any() or torch.isnan(pred_ego).any()
                )
                if ego_is_valid and not ego_has_nan:
                    ego_errors = torch.linalg.norm(pred_ego - gt_ego, dim=-1)
                    miss_count += (ego_errors[-1] > miss_threshold).item()
                    valid_ego_count += 1

                    neighbor_indices = [
                        person_index
                        for person_index in range(1, num_people)
                        if padding_mask[batch_index, person_index] < 0.5
                    ]
                    if neighbor_indices:
                        gt_neighbors = torch.stack(
                            [
                                out_joints[
                                    batch_index,
                                    :,
                                    person_index * token_count,
                                    :2,
                                ]
                                for person_index in neighbor_indices
                            ],
                            dim=0,
                        )
                        distances = torch.linalg.norm(
                            pred_ego.unsqueeze(0) - gt_neighbors, dim=-1
                        )
                        collision_count += (
                            distances < collision_threshold
                        ).sum().item()
                        collision_comparisons += distances.numel()

                for person_index in range(1, num_people):
                    if padding_mask[batch_index, person_index] >= 0.5:
                        continue
                    gt_neighbor = out_joints[
                        batch_index, :, person_index * token_count, :2
                    ]
                    pred_neighbor = pred_joints[
                        batch_index, :, person_index, :2
                    ]
                    if (
                        torch.isnan(gt_neighbor).any()
                        or torch.isnan(pred_neighbor).any()
                    ):
                        continue
                    errors = torch.linalg.norm(
                        gt_neighbor - pred_neighbor, dim=-1
                    )
                    neighbor_ade_sum += errors.mean().item()
                    neighbor_fde_sum += errors[-1].item()
                    neighbor_count += 1

            bar.next()
    bar.finish()

    return {
        'ego_ade': ego_ade_sum / ego_count if ego_count else 0.0,
        'ego_fde': ego_fde_sum / ego_count if ego_count else 0.0,
        'collision_rate': (
            collision_count / collision_comparisons
            if collision_comparisons else 0.0
        ),
        'miss_rate': miss_count / valid_ego_count if valid_ego_count else 0.0,
        'neighbor_ade': (
            neighbor_ade_sum / neighbor_count if neighbor_count else 0.0
        ),
        'neighbor_fde': (
            neighbor_fde_sum / neighbor_count if neighbor_count else 0.0
        ),
    }


def log_and_print(log_file, message):
    print(message)
    log_file.write(message + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--split", default="test", choices=["train", "test", "val"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--modality",
        default="traj+2dbox",
        choices=["traj", "traj+2dbox"],
        help="Input modalities.",
    )
    parser.add_argument(
        "--log_file", default=DEFAULT_LOG_PATH,
        help="Path for the evaluation report.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        parser.error(f"Checkpoint not found: {args.ckpt}")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logger = create_logger('')
    logger.info(f'Loading checkpoint from {args.ckpt}')
    checkpoint = torch.load(
        args.ckpt, map_location='cpu', weights_only=True
    )
    config = checkpoint['config']
    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed_all(0)
    else:
        config["DEVICE"] = "cpu"

    model = create_model(config, logger)
    model.load_state_dict(checkpoint['model'])
    model.to(config["DEVICE"])

    input_frames = config['TRAIN']['input_track_size']
    output_frames = config['TRAIN']['output_track_size']
    if (input_frames, output_frames) != (9, 12):
        logger.warning(
            "Paper evaluation used 9 observation and 12 prediction frames; "
            f"checkpoint uses {input_frames} and {output_frames}."
        )

    dataset_name = config['DATA']['train_datasets'][0]
    dataset = create_dataset(
        dataset_name,
        logger,
        split=args.split,
        track_size=input_frames + output_frames,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['TRAIN']['batch_size'],
        num_workers=config['TRAIN']['num_workers'],
        shuffle=False,
        collate_fn=collate_batch,
    )
    metrics = evaluate_model(model, dataloader, config, args.modality)

    log_directory = os.path.dirname(os.path.abspath(args.log_file))
    os.makedirs(log_directory, exist_ok=True)
    with open(args.log_file, 'w', encoding='utf-8') as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_and_print(log_file, f"--- Evaluation started at {timestamp} ---")
        log_and_print(log_file, "==========================================")
        log_and_print(log_file, f"Model: {args.ckpt}")
        log_and_print(
            log_file,
            f"Loaded model saved at epoch: {checkpoint.get('epoch', 'N/A')}",
        )
        log_and_print(log_file, f"Ego ADE: {metrics['ego_ade']:.4f}")
        log_and_print(
            log_file,
            f"Ego Collision Rate: {metrics['collision_rate']:.4f}",
        )
        log_and_print(log_file, f"Ego FDE: {metrics['ego_fde']:.4f}")
        log_and_print(log_file, f"Ego Miss Rate: {metrics['miss_rate']:.4f}")
        log_and_print(
            log_file, f"Neighbor ADE: {metrics['neighbor_ade']:.4f}"
        )
        log_and_print(
            log_file, f"Neighbor FDE: {metrics['neighbor_fde']:.4f}"
        )
        log_and_print(log_file, "==========================================")

    print(f"Evaluation finished. Results saved to {args.log_file}")


if __name__ == "__main__":
    main()
