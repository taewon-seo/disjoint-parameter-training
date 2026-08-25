import argparse
from datetime import datetime
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from dpt.backbone import clip_embedding_norms, create_model
from dpt.jrdb import (
    batch_process_coords,
    create_dataset,
    dataloader_for,
    get_datasets,
    validation_dataloader_for,
)
from dpt.objectives import (
    collision_loss,
    evaluate_task_loss,
    split_targets,
    task_loss,
    trajectory_loss,
)
from dpt.runtime import (
    RunningAverage,
    compact_progress_suffix,
    create_progress_bar,
    estimate_remaining_time,
    load_config,
    save_checkpoint,
)


def split_task_parameters(model):
    planner_parameters = []
    forecaster_parameters = []
    shared_seen = ego_seen = neighbor_seen = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "fc_out_traj_ego" in name:
            ego_seen += 1
            planner_parameters.append(parameter)
        elif "fc_out_traj_neigh" in name:
            neighbor_seen += 1
            forecaster_parameters.append(parameter)
        else:
            shared_seen += 1
            planner_parameters.append(parameter)
            forecaster_parameters.append(parameter)

    assert ego_seen > 0 and neighbor_seen > 0 and shared_seen > 0
    assert planner_parameters and forecaster_parameters
    return planner_parameters, forecaster_parameters


def train_joint_tasks(
    model,
    config,
    in_joints,
    out_joints,
    out_masks,
    padding_mask,
    planner_optimizer,
    forecaster_optimizer,
):
    if planner_optimizer is None or forecaster_optimizer is None:
        raise ValueError("Both planner and forecaster optimizers are required")

    input_frames = in_joints.shape[1]
    gt_ego, gt_neighbors, ego_mask, neighbor_mask = split_targets(
        out_joints, out_masks, model
    )
    collision_weight = float(config["TRAIN"].get("col_weight", 1.0))
    collision_threshold = config["EVAL"].get("col_threshold", 0.6)
    metrics = {}
    total_loss = torch.tensor(0.0, device=in_joints.device)

    model.train()
    planner_optimizer.zero_grad(set_to_none=True)
    predictions = model(in_joints, padding_mask)
    pred_ego = predictions[:, input_frames:, 0:1, :2]
    pred_neighbors = predictions[:, input_frames:, 1:, :2].detach()

    plan_cost = collision_loss(
        pred_ego.squeeze(-2),
        pred_neighbors,
        mask_neigh=neighbor_mask,
        threshold=collision_threshold,
    )
    gt_cost = collision_loss(
        gt_ego.squeeze(-2),
        pred_neighbors,
        mask_neigh=neighbor_mask,
        threshold=collision_threshold,
    )
    collision_difference = plan_cost - gt_cost
    loss_ego = trajectory_loss(
        pred_ego, gt_ego, mask=ego_mask, ego=True
    )
    planner_loss = loss_ego + collision_weight * collision_difference

    planner_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["TRAIN"]["max_grad_norm"]
    )
    planner_optimizer.step()
    clip_embedding_norms(model)

    metrics.update(
        {
            "loss_ego": loss_ego.detach(),
            "plan_cost": plan_cost.detach(),
            "gt_cost": gt_cost.detach(),
            "col_plan_diff": collision_difference.detach(),
        }
    )
    total_loss += planner_loss.detach()

    forecaster_optimizer.zero_grad(set_to_none=True)
    predictions = model(in_joints, padding_mask)
    pred_ego = predictions[:, input_frames:, 0:1, :2].detach()
    pred_neighbors = predictions[:, input_frames:, 1:, :2]

    plan_cost = collision_loss(
        pred_ego.squeeze(-2),
        pred_neighbors,
        mask_neigh=neighbor_mask,
        threshold=collision_threshold,
    )
    gt_cost = collision_loss(
        gt_ego.squeeze(-2),
        pred_neighbors,
        mask_neigh=neighbor_mask,
        threshold=collision_threshold,
    )
    collision_difference = plan_cost - gt_cost
    loss_neighbors = trajectory_loss(
        pred_neighbors,
        gt_neighbors,
        mask=neighbor_mask,
        ego=False,
    )
    forecaster_loss = (
        loss_neighbors + collision_weight * collision_difference
    )

    forecaster_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["TRAIN"]["max_grad_norm"]
    )
    forecaster_optimizer.step()
    clip_embedding_norms(model)

    metrics.update(
        {
            "loss_neigh": loss_neighbors.detach(),
            "plan_cost": plan_cost.detach(),
            "gt_cost": gt_cost.detach(),
            "col_fore_diff": collision_difference.detach(),
        }
    )
    total_loss += forecaster_loss.detach()
    metrics["loss_total"] = total_loss
    return total_loss, metrics


def adjust_learning_rate(epoch, config, *optimizers):
    is_pretraining = config["MODEL"].get("task") == "pretrain"
    decay = config["TRAIN"]["lr_decay"]
    drop = config["TRAIN"].get("lr_drop", False)
    epochs = config["TRAIN"]["epochs"]

    def scheduled_rate(base_rate):
        rate = float(base_rate) * decay**epoch
        if drop:
            rate *= 0.1 ** (epoch // (epochs * 4.0 / 5.0))
        return rate

    if is_pretraining:
        planner_optimizer, forecaster_optimizer = optimizers
        planner_rate = scheduled_rate(config["TRAIN"]["lr_planner"])
        forecaster_rate = scheduled_rate(config["TRAIN"]["lr_forecaster"])
        for group in planner_optimizer.param_groups:
            group["lr"] = planner_rate
        for group in forecaster_optimizer.param_groups:
            group["lr"] = forecaster_rate
        print(f"Adjusted planner lr: {planner_rate:.6f}")
        print(f"Adjusted forecaster lr: {forecaster_rate:.6f}")
        return

    optimizer = optimizers[0]
    if optimizer is None:
        return
    rate = scheduled_rate(config["TRAIN"]["lr"])
    for group in optimizer.param_groups:
        group["lr"] = rate
    print(f"Adjusted lr: {rate:.6f}")


def train(config, logger, experiment_name=""):
    from torch.utils.tensorboard import SummaryWriter

    input_frames = config["TRAIN"]["input_track_size"]
    output_frames = config["TRAIN"]["output_track_size"]
    dataset_train = ConcatDataset(
        get_datasets(config["DATA"]["train_datasets"], config, logger)
    )
    dataloader_train = dataloader_for(
        dataset_train, config, shuffle=False
    )
    logger.info(f"Training on a total of {len(dataset_train)} annotations.")

    dataset_val = create_dataset(
        config["DATA"]["train_datasets"][0],
        logger,
        split="val",
        track_size=input_frames + output_frames,
    )
    dataloader_val = validation_dataloader_for(
        dataset_val, config, shuffle=False
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer_name = f"{experiment_name}_{timestamp}"
    writer_train = SummaryWriter(
        os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN")
    )
    writer_valid = SummaryWriter(
        os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID")
    )

    model = create_model(config, logger)
    model.to(config["DEVICE"])
    task = config["MODEL"].get("task")
    optimizer = None
    planner_optimizer = forecaster_optimizer = None

    if task == "pretrain":
        planner_parameters, forecaster_parameters = split_task_parameters(model)
        planner_optimizer = torch.optim.Adam(
            planner_parameters, lr=float(config["TRAIN"]["lr_planner"])
        )
        forecaster_optimizer = torch.optim.Adam(
            forecaster_parameters,
            lr=float(config["TRAIN"]["lr_forecaster"]),
        )
    elif task == "ego":
        planner_parameters, _ = split_task_parameters(model)
        optimizer = torch.optim.Adam(
            planner_parameters, lr=float(config["TRAIN"]["lr"])
        )
    elif task == "neighbor":
        _, forecaster_parameters = split_task_parameters(model)
        optimizer = torch.optim.Adam(
            forecaster_parameters, lr=float(config["TRAIN"]["lr"])
        )
    else:
        raise ValueError("Unknown task type in config['MODEL']['task']")

    best_checkpoint_name = {
        "pretrain": "pretrained_model.pth.tar",
        "ego": "plan_finetuned_model.pth.tar",
        "neighbor": "pred_finetuned_model.pth.tar",
    }[task]
    is_pretraining = task == "pretrain"

    start_epoch = 0
    checkpoint_path = config["MODEL"].get("checkpoint", "")
    pretrained_path = config["MODEL"].get("pretrained", "")
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint.get("epoch", -1) + 1
            if is_pretraining:
                if "opt_planner" in checkpoint:
                    planner_optimizer.load_state_dict(
                        checkpoint["opt_planner"]
                    )
                if "opt_forecaster" in checkpoint:
                    forecaster_optimizer.load_state_dict(
                        checkpoint["opt_forecaster"]
                    )
            elif optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            logger.warning(
                f"Checkpoint file not found: {checkpoint_path}. "
                "Starting from scratch."
            )
    elif pretrained_path:
        if os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(
                pretrained_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model"], strict=False)
            logger.info("Pretrained weights loaded.")
        else:
            logger.warning(
                f"Pretrained file not found: {pretrained_path}. "
                "Starting from scratch."
            )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(f"Model has {num_parameters} parameters.")

    min_val_loss = 1e4
    train_start_time = time.time()
    total_train_epochs = config["TRAIN"]["epochs"] - start_epoch

    for epoch in range(start_epoch, config["TRAIN"]["epochs"]):
        epoch_start = time.time()
        loss_average = RunningAverage()
        ego_loss_average = RunningAverage()
        neighbor_loss_average = RunningAverage()
        plan_cost_average = RunningAverage()
        collision_difference_average = RunningAverage()

        if config["TRAIN"]["optimizer"] == "adam":
            if is_pretraining:
                adjust_learning_rate(
                    epoch,
                    config,
                    planner_optimizer,
                    forecaster_optimizer,
                )
            else:
                adjust_learning_rate(epoch, config, optimizer)

        train_steps = len(dataloader_train)
        bar = create_progress_bar(
            f"TRAIN {epoch}/{config['TRAIN']['epochs'] - 1}",
            fill="#",
            max=train_steps,
        )
        for step, (joints, masks, padding_mask) in enumerate(dataloader_train):
            model.train()
            in_joints, _, out_joints, out_masks, padding_mask = (
                batch_process_coords(
                    joints, masks, padding_mask, config, training=True
                )
            )
            device = config["DEVICE"]
            in_joints = in_joints.to(device, non_blocking=True)
            out_joints = out_joints.to(device, non_blocking=True)
            out_masks = out_masks.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)

            if is_pretraining:
                loss, metrics = train_joint_tasks(
                    model,
                    config,
                    in_joints,
                    out_joints,
                    out_masks,
                    padding_mask,
                    planner_optimizer,
                    forecaster_optimizer,
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = task_loss(
                    model,
                    config,
                    in_joints,
                    out_joints,
                    out_masks,
                    padding_mask,
                )
                loss.backward()
                planner_parameters, forecaster_parameters = (
                    split_task_parameters(model)
                )
                parameters = (
                    planner_parameters
                    if task == "ego"
                    else forecaster_parameters
                )
                torch.nn.utils.clip_grad_norm_(
                    parameters, config["TRAIN"]["max_grad_norm"]
                )
                optimizer.step()

            batch_size = len(joints)
            loss_average.add(loss.item(), batch_size)
            if "loss_ego" in metrics:
                ego_loss_average.add(metrics["loss_ego"].item(), batch_size)
            if "loss_neigh" in metrics:
                neighbor_loss_average.add(
                    metrics["loss_neigh"].item(), batch_size
                )
            if "plan_cost" in metrics:
                plan_cost_average.add(
                    metrics["plan_cost"].item(), batch_size
                )
            if "col_plan_diff" in metrics:
                collision_difference_average.add(
                    metrics["col_plan_diff"].item(), batch_size
                )

            now = time.time()
            completed = step + 1
            epoch_eta = estimate_remaining_time(
                epoch_start, completed, train_steps, now=now
            )
            total_steps = total_train_epochs * train_steps
            completed_steps = (
                (epoch - start_epoch) * train_steps + completed
            )
            total_eta = estimate_remaining_time(
                train_start_time, completed_steps, total_steps, now=now
            )
            bar.suffix = compact_progress_suffix(
                loss_average.mean, epoch_eta, total_eta
            )
            bar.next()
            if config.get("dry_run", False):
                break

        bar.finish()
        writer_train.add_scalar("loss/train_total", loss_average.mean, epoch)
        if ego_loss_average.samples:
            writer_train.add_scalar(
                "loss/train_ego", ego_loss_average.mean, epoch
            )
        if neighbor_loss_average.samples:
            writer_train.add_scalar(
                "loss/train_neigh", neighbor_loss_average.mean, epoch
            )
        if plan_cost_average.samples:
            writer_train.add_scalar(
                "cost/train_plan_cost", plan_cost_average.mean, epoch
            )
        if collision_difference_average.samples:
            writer_train.add_scalar(
                "cost/train_col_plan_diff",
                collision_difference_average.mean,
                epoch,
            )

        if is_pretraining:
            writer_train.add_scalar(
                "lr/planner", planner_optimizer.param_groups[0]["lr"], epoch
            )
            writer_train.add_scalar(
                "lr/forecaster",
                forecaster_optimizer.param_groups[0]["lr"],
                epoch,
            )
        elif optimizer is not None:
            writer_train.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if config.get("dry_run", False):
            logger.info("Dry run completed after one training batch.")
            break

        val_loss, val_metrics = evaluate_task_loss(
            model, dataloader_val, config
        )
        writer_valid.add_scalar("loss/val_total", val_loss, epoch)
        writer_valid.add_scalar(
            "loss/val_ego", val_metrics["loss_ego"], epoch
        )
        writer_valid.add_scalar(
            "loss/val_neigh", val_metrics["loss_neigh"], epoch
        )

        val_frequency = config["TRAIN"].get("val_frequency", 1)
        if (epoch + 1) % val_frequency == 0 and val_loss < min_val_loss:
            min_val_loss = val_loss
            print("\n---------------- BEST MODEL UPDATED ----------------")
            print(f"Best Loss at epoch {epoch}: {val_loss:.4f}")
            save_checkpoint(
                model,
                optimizer if not is_pretraining else None,
                planner_optimizer if is_pretraining else None,
                forecaster_optimizer if is_pretraining else None,
                epoch,
                config,
                best_checkpoint_name,
                logger,
            )

        checkpoint_frequency = int(
            config["TRAIN"].get("checkpoint_frequency", 5)
        )
        if checkpoint_frequency > 0 and (
            epoch + 1
        ) % checkpoint_frequency == 0:
            filename = f"checkpoint_epoch_{epoch + 1}.pth.tar"
            logger.info(f"Saving periodic checkpoint: {filename}")
            save_checkpoint(
                model,
                optimizer if not is_pretraining else None,
                planner_optimizer if is_pretraining else None,
                forecaster_optimizer if is_pretraining else None,
                epoch,
                config,
                filename,
                logger,
            )

        print(f"Time for epoch {epoch}: {time.time() - epoch_start:.2f}s")

    if not config.get("dry_run", False):
        save_checkpoint(
            model,
            optimizer if not is_pretraining else None,
            planner_optimizer if is_pretraining else None,
            forecaster_optimizer if is_pretraining else None,
            epoch,
            config,
            "checkpoint.pth.tar",
            logger,
        )

    writer_train.close()
    writer_valid.close()
    logger.info("All done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["ego", "pretrain", "neighbor"],
        help="Training task.",
    )
    parser.add_argument(
        "--exp_name", default="",
        help="Experiment name (defaults to the task or config name).",
    )
    parser.add_argument(
        "--cfg", default="",
        help="Config path (uses the task default when omitted).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run one training iteration.",
    )
    return parser.parse_args()


def resolve_config(args):
    if not args.cfg:
        defaults = {
            "pretrain": ("configs/pretrain.yaml", "pretrain"),
            "ego": ("configs/plan_finetune.yaml", "plan_finetune"),
            "neighbor": ("configs/pred_finetune.yaml", "pred_finetune"),
        }
        if args.task not in defaults:
            raise ValueError(
                "Please provide --task {ego, pretrain, neighbor} or --cfg"
            )
        args.cfg, default_name = defaults[args.task]
        if not args.exp_name:
            args.exp_name = default_name
    elif not args.exp_name:
        args.exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
    return load_config(args.cfg, exp_name=args.exp_name)


def configure_logger(log_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"train_log_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main():
    args = parse_args()
    config = resolve_config(args)
    config["dry_run"] = args.dry_run
    seed = config["SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        config["DEVICE"] = "cpu"

    logger = configure_logger(config["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(config)
    train(config, logger, experiment_name=args.exp_name)


if __name__ == "__main__":
    main()
