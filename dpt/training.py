import copy
from datetime import datetime
import os
import time

import torch
from torch.utils.data import ConcatDataset

from dpt.backbone import clip_embedding_norms, create_model
from dpt.jrdb import (
    batch_process_coords,
    create_dataset,
    dataloader_for,
    get_datasets,
)
from dpt.objectives import evaluate_task_loss, task_loss
from dpt.runtime import (
    RunningAverage,
    compact_progress_suffix,
    create_progress_bar,
    estimate_remaining_time,
    save_checkpoint,
)


class DisjointFinetuner:
    """Train planner and forecaster models on disjoint parameter masks."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config["DEVICE"]

        self.logger.info("Initializing DPT models.")
        self.model_planner = self._load_pretrained_model()
        self.model_forecaster = copy.deepcopy(self.model_planner)

        self.w_pretrain = {
            name: value.cpu().clone()
            for name, value in self.model_planner.state_dict().items()
        }
        self.param_names = [
            name
            for name, parameter in self.model_planner.named_parameters()
            if parameter.requires_grad
        ]
        self.num_total_params = sum(
            parameter.numel()
            for parameter in self.model_planner.parameters()
            if parameter.requires_grad
        )

        self.mask_planner = {
            name: torch.zeros_like(parameter, dtype=torch.bool, device=self.device)
            for name, parameter in self.model_planner.named_parameters()
        }
        self.mask_forecaster = {
            name: torch.zeros_like(parameter, dtype=torch.bool, device=self.device)
            for name, parameter in self.model_forecaster.named_parameters()
        }

        planner_lr = float(config['TRAIN']['lr_planner'])
        forecaster_lr = float(config['TRAIN']['lr_forecaster'])
        optimizer_name = config['TRAIN'].get('optimizer', 'adam').lower()
        self.logger.info(f"Using optimizer: {optimizer_name}")

        if optimizer_name == 'adam':
            self.opt_planner = torch.optim.Adam(
                self.model_planner.parameters(), lr=planner_lr
            )
            self.opt_forecaster = torch.optim.Adam(
                self.model_forecaster.parameters(), lr=forecaster_lr
            )
        elif optimizer_name == 'sgd':
            momentum = config['TRAIN'].get('momentum', 0.9)
            self.logger.info(f"Using SGD with momentum: {momentum}")
            self.opt_planner = torch.optim.SGD(
                self.model_planner.parameters(), lr=planner_lr, momentum=momentum
            )
            self.opt_forecaster = torch.optim.SGD(
                self.model_forecaster.parameters(), lr=forecaster_lr,
                momentum=momentum,
            )
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. Supported: adam, sgd"
            )


    def _load_pretrained_model(self):
        model_config = copy.deepcopy(self.config)
        model_config['MODEL']['task'] = 'ego'
        model_config['MODEL']['checkpoint'] = ""
        model = create_model(model_config, self.logger)

        pretrained_path = self.config["MODEL"]["pretrained"]
        self.logger.info(f"Loading pretrained weights: {pretrained_path}")
        checkpoint = torch.load(
            pretrained_path, map_location='cpu', weights_only=True
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(self.device)
        return model

    def _update_masks_and_zero_grads(
        self, model, cumulative_mask, opponent_mask, k_new_params
    ):
        """Claim the largest free gradients and enforce the task mask."""
        with torch.no_grad():
            grad_list = []
            cumulative_list = []
            opponent_list = []
            param_info = []

            for name, parameter in model.named_parameters():
                if name not in cumulative_mask or parameter.grad is None:
                    continue
                numel = parameter.numel()
                if numel == 0:
                    continue
                param_info.append((name, parameter.shape, numel))
                grad_list.append(parameter.grad.flatten())
                cumulative_list.append(cumulative_mask[name].flatten())
                opponent_list.append(opponent_mask[name].flatten())

            if not grad_list:
                return

            flat_grads = torch.cat(grad_list)
            flat_cumulative = torch.cat(cumulative_list)
            flat_opponent = torch.cat(opponent_list)

            unclaimed_indices = (
                ~flat_cumulative & ~flat_opponent
            ).nonzero(as_tuple=False).flatten()
            if unclaimed_indices.numel() > 0 and k_new_params > 0:
                unclaimed_grads = flat_grads.index_select(
                    0, unclaimed_indices
                ).abs()
                k_to_claim = min(k_new_params, unclaimed_grads.numel())
                _, relative_indices = torch.topk(unclaimed_grads, k_to_claim)
                claimed_indices = unclaimed_indices.index_select(
                    0, relative_indices
                )
                flat_cumulative.scatter_(0, claimed_indices, True)

            flat_grads.mul_(flat_cumulative)

            offset = 0
            for name, shape, numel in param_info:
                model.get_parameter(name).grad.copy_(
                    flat_grads.narrow(0, offset, numel).reshape(shape)
                )
                cumulative_mask[name].copy_(
                    flat_cumulative.narrow(0, offset, numel).reshape(shape)
                )
                offset += numel

    @staticmethod
    def _get_total_masked_params(mask):
        return sum(mask_tensor.sum().item() for mask_tensor in mask.values())

    def _get_effective_activation_ratio(self, model):
        total_values = 0
        changed_values = 0
        with torch.no_grad():
            current_state = {
                name: value.cpu()
                for name, value in model.state_dict().items()
            }
            for name, pretrained_value in self.w_pretrain.items():
                if name not in current_state:
                    continue
                delta = current_state[name] - pretrained_value
                total_values += delta.numel()
                changed_values += (delta.abs() > 1e-9).sum().item()

        if total_values == 0:
            return 0.0
        return changed_values / total_values * 100.0

    def _get_gradient_l2_norm(self, model):
        gradients = [
            parameter.grad.flatten()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return torch.tensor(0.0, device=self.device)
        return torch.norm(torch.cat(gradients), p=2)

    def _allocation_per_batch(self, ratio, allocation_epochs, num_batches):
        if ratio <= 0 or allocation_epochs <= 0 or num_batches <= 0:
            return 0
        per_epoch = self.num_total_params * ratio / allocation_epochs
        return max(1, int(per_epoch // num_batches))

    def train(self, dataloader_train, dataloader_val, writer_train, writer_valid):
        allocation_epochs = self.config['DISJOINT']['epoch_allocation']
        training_epochs = self.config['DISJOINT']['epoch_train']
        total_epochs = allocation_epochs + training_epochs
        if total_epochs <= 0:
            raise ValueError("DPT requires at least one training epoch")
        if len(dataloader_train) == 0:
            raise ValueError("Training dataloader contains no batches")

        planner_ratio = self.config['DISJOINT']['planner_allocation']
        forecaster_ratio = self.config['DISJOINT']['forecaster_allocation']
        planner_k_per_batch = self._allocation_per_batch(
            planner_ratio, allocation_epochs, len(dataloader_train)
        )
        forecaster_k_per_batch = self._allocation_per_batch(
            forecaster_ratio, allocation_epochs, len(dataloader_train)
        )

        self.logger.info(
            f"DPT training: {allocation_epochs} allocation epochs, "
            f"{training_epochs} mask-training epochs"
        )
        self.logger.info(
            f"Planner allocation: {planner_k_per_batch} params/batch "
            f"({planner_ratio:.2%} total)"
        )
        self.logger.info(
            f"Forecaster allocation: {forecaster_k_per_batch} params/batch "
            f"({forecaster_ratio:.2%} total)"
        )

        preemption = self.config['DISJOINT']['preemption']
        turn_order = (
            ['forecaster', 'planner']
            if preemption == 'forecaster'
            else ['planner', 'forecaster']
        )
        min_val_loss = {'planner': float('inf'), 'forecaster': float('inf')}
        train_start_time = time.time()
        total_train_steps = total_epochs * len(turn_order) * len(dataloader_train)

        for epoch in range(total_epochs):
            epoch_start_time = time.time()
            is_allocation_phase = epoch < allocation_epochs
            if is_allocation_phase:
                planner_k = planner_k_per_batch
                forecaster_k = forecaster_k_per_batch
                self.logger.info(
                    f"===== Epoch {epoch}/{total_epochs - 1}: allocation ====="
                )
            else:
                planner_k = forecaster_k = 0
                self.logger.info(
                    f"===== Epoch {epoch}/{total_epochs - 1}: mask training ====="
                )

            current_turn_order = turn_order
            if preemption == 'alternate' and epoch % 2 == 1:
                current_turn_order = turn_order[::-1]
            self.logger.info(
                f"Turn order: {current_turn_order[0]} -> {current_turn_order[1]}"
            )

            for turn_index, turn in enumerate(current_turn_order):
                original_task = self.config['MODEL'].get('task')
                if turn == 'planner':
                    model = self.model_planner
                    optimizer = self.opt_planner
                    task_mask = self.mask_planner
                    opponent_mask = self.mask_forecaster
                    k_for_turn = planner_k
                    self.config['MODEL']['task'] = 'ego'
                else:
                    model = self.model_forecaster
                    optimizer = self.opt_forecaster
                    task_mask = self.mask_forecaster
                    opponent_mask = self.mask_planner
                    k_for_turn = forecaster_k
                    self.config['MODEL']['task'] = 'neighbor'

                loss_average = RunningAverage()
                gradient_average = RunningAverage()
                ego_loss_average = RunningAverage()
                plan_cost_average = RunningAverage()
                collision_difference_average = RunningAverage()
                model.train()

                label = "Plan" if turn == "planner" else "Pred"
                bar = create_progress_bar(
                    f"TRAIN {label} {epoch}", fill="#", max=len(dataloader_train)
                )
                for batch_index, (joints, masks, padding_mask) in enumerate(
                    dataloader_train
                ):
                    in_joints, _, out_joints, out_masks, padding_mask = (
                        batch_process_coords(
                            joints, masks, padding_mask, self.config, training=True
                        )
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss, loss_parts = task_loss(
                        model,
                        self.config,
                        in_joints.to(self.device),
                        out_joints.to(self.device),
                        out_masks.to(self.device),
                        padding_mask.to(self.device),
                    )
                    loss.backward()

                    gradient_norm = self._get_gradient_l2_norm(model)
                    gradient_average.add(gradient_norm.item(), len(joints))
                    self._update_masks_and_zero_grads(
                        model, task_mask, opponent_mask, k_for_turn
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config['TRAIN']['max_grad_norm']
                    )
                    optimizer.step()
                    clip_embedding_norms(
                        model, max_norm=1.0, parameter_mask=task_mask
                    )

                    loss_average.add(loss.item(), len(joints))
                    if turn == 'planner':
                        if 'loss_ego' in loss_parts:
                            ego_loss_average.add(
                                loss_parts['loss_ego'].item(), len(joints)
                            )
                        if 'plan_cost' in loss_parts:
                            plan_cost_average.add(
                                loss_parts['plan_cost'].item(), len(joints)
                            )
                        if 'col_plan_diff' in loss_parts:
                            collision_difference_average.add(
                                loss_parts['col_plan_diff'].item(), len(joints)
                            )

                    now = time.time()
                    steps_in_epoch = (
                        turn_index * len(dataloader_train) + batch_index + 1
                    )
                    completed_steps = (
                        epoch * len(turn_order) * len(dataloader_train)
                        + steps_in_epoch
                    )
                    bar.suffix = compact_progress_suffix(
                        loss_average.mean,
                        estimate_remaining_time(
                            epoch_start_time,
                            steps_in_epoch,
                            len(current_turn_order) * len(dataloader_train),
                            now=now,
                        ),
                        estimate_remaining_time(
                            train_start_time,
                            completed_steps,
                            total_train_steps,
                            now=now,
                        ),
                    )
                    bar.next()
                    if self.config.get('dry_run', False):
                        break

                bar.finish()
                writer_train.add_scalar(
                    f"loss/train_{turn}", loss_average.mean, epoch
                )
                writer_train.add_scalar(
                    f"diagnostics/grad_L2_norm_avg_{turn}",
                    gradient_average.mean,
                    epoch,
                )
                if turn == 'planner':
                    if ego_loss_average.samples:
                        writer_train.add_scalar(
                            "loss/train_ego_mse", ego_loss_average.mean, epoch
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
                self.config['MODEL']['task'] = original_task

            planner_mask_size = self._get_total_masked_params(self.mask_planner)
            forecaster_mask_size = self._get_total_masked_params(
                self.mask_forecaster
            )
            overlap_count = sum(
                (self.mask_planner[name] & self.mask_forecaster[name]).sum().item()
                for name in self.param_names
            )
            planner_activation = self._get_effective_activation_ratio(
                self.model_planner
            )
            forecaster_activation = self._get_effective_activation_ratio(
                self.model_forecaster
            )

            writer_train.add_scalar(
                "disjoint/planner_mask_ratio_ALLOCATED",
                planner_mask_size / self.num_total_params,
                epoch,
            )
            writer_train.add_scalar(
                "disjoint/forecaster_mask_ratio_ALLOCATED",
                forecaster_mask_size / self.num_total_params,
                epoch,
            )
            writer_train.add_scalar(
                "disjoint/overlap_check", overlap_count, epoch
            )
            writer_train.add_scalar(
                "disjoint/planner_activation_ratio_ACTIVATED",
                planner_activation,
                epoch,
            )
            writer_train.add_scalar(
                "disjoint/forecaster_activation_ratio_ACTIVATED",
                forecaster_activation,
                epoch,
            )
            self.logger.info(
                "Allocated | "
                f"Planner: {planner_mask_size / self.num_total_params:.2%} | "
                f"Forecaster: {forecaster_mask_size / self.num_total_params:.2%}"
            )
            self.logger.info(
                "Activated | "
                f"Planner: {planner_activation:.2f}% | "
                f"Forecaster: {forecaster_activation:.2f}%"
            )
            if overlap_count:
                raise RuntimeError(
                    f"DPT ownership masks overlap at {overlap_count} parameters"
                )

            if self.config.get('dry_run', False):
                self.logger.info(
                    "Dry run completed after one planner and one forecaster batch."
                )
                return

            original_task = self.config['MODEL'].get('task')
            for turn, model, optimizer, checkpoint_name in (
                (
                    'planner', self.model_planner, self.opt_planner,
                    'dpt_plan_model.pth.tar',
                ),
                (
                    'forecaster', self.model_forecaster, self.opt_forecaster,
                    'dpt_pred_model.pth.tar',
                ),
            ):
                self.config['MODEL']['task'] = (
                    'ego' if turn == 'planner' else 'neighbor'
                )
                validation_loss, _ = evaluate_task_loss(
                    model, dataloader_val, self.config
                )
                writer_valid.add_scalar(
                    f"loss/val_{turn}", validation_loss, epoch
                )
                if validation_loss < min_val_loss[turn]:
                    min_val_loss[turn] = validation_loss
                    self.logger.info(
                        f"New best {turn} model at epoch {epoch}: "
                        f"{validation_loss:.4f}"
                    )
                    save_checkpoint(
                        model, optimizer, None, None, epoch, self.config,
                        checkpoint_name, self.logger,
                    )
            self.config['MODEL']['task'] = original_task

            elapsed_seconds = time.time() - epoch_start_time
            self.logger.info(
                f"===== Epoch {epoch} complete in {elapsed_seconds:.2f}s ====="
            )
            writer_train.add_scalar(
                "diagnostics/epoch_time_seconds", elapsed_seconds, epoch
            )

            checkpoint_frequency = int(
                self.config['TRAIN'].get('checkpoint_frequency', 5)
            )
            if checkpoint_frequency > 0 and (
                epoch + 1
            ) % checkpoint_frequency == 0:
                save_checkpoint(
                    self.model_planner, self.opt_planner, None, None,
                    epoch, self.config,
                    f'dpt_plan_checkpoint_epoch_{epoch + 1}.pth.tar',
                    self.logger,
                )
                save_checkpoint(
                    self.model_forecaster, self.opt_forecaster, None, None,
                    epoch, self.config,
                    f'dpt_pred_checkpoint_epoch_{epoch + 1}.pth.tar',
                    self.logger,
                )

        self.logger.info(f"Saving final models at epoch {epoch}...")
        save_checkpoint(
            self.model_planner, self.opt_planner, None, None, epoch, self.config,
            'final_dpt_plan_model.pth.tar', self.logger,
        )
        save_checkpoint(
            self.model_forecaster, self.opt_forecaster, None, None,
            epoch, self.config, 'final_dpt_pred_model.pth.tar', self.logger,
        )


def run_training(config, logger, experiment_name):
    from torch.utils.tensorboard import SummaryWriter

    input_frames = config['TRAIN']['input_track_size']
    output_frames = config['TRAIN']['output_track_size']
    dataset_train = ConcatDataset(
        get_datasets(config['DATA']['train_datasets'], config, logger)
    )
    dataloader_train = dataloader_for(dataset_train, config, shuffle=True)
    logger.info(f"Training on {len(dataset_train)} annotations.")

    dataset_val = create_dataset(
        config['DATA']['train_datasets'][0],
        logger,
        split="val",
        track_size=input_frames + output_frames,
    )
    dataloader_val = dataloader_for(dataset_val, config, shuffle=False)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer_name = f"{experiment_name}_{timestamp}"
    writer_train = SummaryWriter(
        os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN")
    )
    writer_valid = SummaryWriter(
        os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID")
    )
    try:
        trainer = DisjointFinetuner(config, logger)
        trainer.train(
            dataloader_train, dataloader_val, writer_train, writer_valid
        )
    finally:
        writer_train.close()
        writer_valid.close()
    logger.info("All done.")
