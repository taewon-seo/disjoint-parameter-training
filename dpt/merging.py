from collections import OrderedDict
import copy

import numpy as np
import torch

from dpt.backbone import create_model


def load_checkpoint_model(checkpoint_path, device, logger):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = copy.deepcopy(checkpoint["config"])
    config["DEVICE"] = device
    model = create_model(config, logger)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def calculate_task_vector(base_model, tuned_model):
    deltas_by_layer = OrderedDict()
    absolute_deltas = []
    tuned_parameters = dict(tuned_model.named_parameters())

    for name, base_parameter in base_model.named_parameters():
        if name not in tuned_parameters or not base_parameter.requires_grad:
            continue
        delta = tuned_parameters[name].detach() - base_parameter.detach()
        delta = delta.cpu()
        deltas_by_layer[name] = delta
        absolute_deltas.append(delta.abs().numpy().reshape(-1))

    all_absolute_deltas = np.concatenate(absolute_deltas)
    return deltas_by_layer, all_absolute_deltas, all_absolute_deltas.size


def get_activation_mask(
    deltas_by_layer, all_absolute_deltas, top_k_percentage, device
):
    percentage = float(top_k_percentage)
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("top-k percentage must be between 0 and 100")

    global_mask = np.zeros_like(all_absolute_deltas, dtype=bool)
    if percentage >= 100.0:
        global_mask = all_absolute_deltas > 0
    elif percentage > 0.0:
        nonzero_indices = np.flatnonzero(all_absolute_deltas > 0)
        if nonzero_indices.size:
            target_count = max(
                1, int(all_absolute_deltas.size * percentage / 100.0)
            )
            target_count = min(target_count, nonzero_indices.size)
            nonzero_values = all_absolute_deltas[nonzero_indices]
            selected = np.argpartition(
                nonzero_values, -target_count
            )[-target_count:]
            global_mask[nonzero_indices[selected]] = True

    masks_by_layer = OrderedDict()
    offset = 0
    for name, delta in deltas_by_layer.items():
        numel = delta.numel()
        layer_mask = global_mask[offset:offset + numel].reshape(delta.shape)
        masks_by_layer[name] = torch.from_numpy(layer_mask).float().to(device)
        offset += numel
    return masks_by_layer, global_mask


def merge_task_vectors(
    base_model,
    planner_model,
    forecaster_model,
    device,
    planner_top_k,
    forecaster_top_k,
):
    planner_deltas, planner_abs, total_params = calculate_task_vector(
        base_model, planner_model
    )
    forecaster_deltas, forecaster_abs, _ = calculate_task_vector(
        base_model, forecaster_model
    )
    planner_masks, planner_global_mask = get_activation_mask(
        planner_deltas, planner_abs, planner_top_k, device
    )
    forecaster_masks, forecaster_global_mask = get_activation_mask(
        forecaster_deltas, forecaster_abs, forecaster_top_k, device
    )

    overlap_count = np.logical_and(
        planner_global_mask, forecaster_global_mask
    ).sum()
    union_count = np.logical_or(
        planner_global_mask, forecaster_global_mask
    ).sum()
    overlap_ratio = overlap_count / union_count * 100 if union_count else 0.0

    merged_state = copy.deepcopy(base_model.state_dict())
    for name, base_parameter in base_model.named_parameters():
        if name not in planner_deltas:
            continue
        planner_delta = planner_deltas[name].to(device)
        forecaster_delta = forecaster_deltas[name].to(device)
        planner_mask = planner_masks[name]
        forecaster_mask = forecaster_masks[name]

        overlap_mask = planner_mask * forecaster_mask
        planner_only = planner_mask * (1.0 - forecaster_mask)
        forecaster_only = forecaster_mask * (1.0 - planner_mask)
        final_delta = (
            planner_delta * planner_only
            + forecaster_delta * forecaster_only
            + (planner_delta + forecaster_delta) / 2.0 * overlap_mask
        )
        merged_state[name].copy_(base_parameter.detach() + final_delta)

    return merged_state, {
        "total_params": total_params,
        "overlap_ratio": overlap_ratio,
        "planner_top_k": float(planner_top_k),
        "forecaster_top_k": float(forecaster_top_k),
    }
