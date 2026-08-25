import torch

from dpt.jrdb import batch_process_coords
from dpt.runtime import RunningAverage, create_progress_bar


def trajectory_loss(output, target, mask=None, ego=False):
    if ego:
        pred_xy = output[:, :, 0, :2]
        gt_xy = target[:, :, 0, :2]
        return torch.norm(pred_xy - gt_xy, p=2, dim=-1).mean()

    pred_xy = output[:, :, :, :2]
    gt_xy = target[:, :, :pred_xy.shape[2], :2]
    if mask is not None:
        mask = mask[:, :, :pred_xy.shape[2]]

    pred_xy = torch.nan_to_num(pred_xy, nan=0.0)
    gt_xy = torch.nan_to_num(gt_xy, nan=0.0)
    distance = torch.norm(pred_xy - gt_xy, p=2, dim=-1)
    if mask is None:
        return distance.mean()

    distance = distance * mask
    denominator = mask.sum(dim=[0, 1])
    valid_agents = denominator > 0
    if valid_agents.sum() == 0:
        return torch.tensor(0.0, device=output.device)

    mean_per_agent = distance.sum(dim=[0, 1]) / (denominator + 1e-6)
    return mean_per_agent[valid_agents].mean()


def collision_loss(
    plan, forecasts, mask_neigh=None, threshold=0.6, eps=0.2
):
    dist = torch.linalg.norm(plan.unsqueeze(2) - forecasts, dim=-1)
    l1_mask = dist < threshold
    l2_mask = (dist > threshold) & (dist < threshold + eps)
    offset = dist - threshold
    l1 = (-offset + eps / 2.0) * l1_mask
    l2 = ((offset - eps) ** 2) / (2.0 * eps) * l2_mask
    penalty = l1 + l2
    if mask_neigh is not None:
        penalty = penalty * mask_neigh
    per_human = penalty.sum(dim=1)
    if per_human.shape[-1] == 0:
        return torch.tensor(0.0, device=plan.device)
    return per_human.max(dim=-1)[0].mean()


def split_targets(out_joints, out_masks, model):
    batch_size, output_frames, packed_agents, feature_size = out_joints.shape
    token_count = model.token_num
    num_people = packed_agents // token_count

    targets = out_joints.view(
        batch_size,
        output_frames,
        num_people,
        token_count,
        feature_size,
    )
    masks = out_masks.view(
        batch_size, output_frames, num_people, token_count
    )
    return (
        targets[:, :, 0:1, 0, :2],
        targets[:, :, 1:, 0, :2],
        masks[:, :, 0:1, 0],
        masks[:, :, 1:, 0],
    )


def task_loss(model, config, in_joints, out_joints, out_masks, padding_mask):
    input_frames = in_joints.shape[1]
    predictions = model(in_joints, padding_mask)
    gt_ego, gt_neighbors, ego_mask, neighbor_mask = split_targets(
        out_joints, out_masks, model
    )

    pred_ego = predictions[:, input_frames:, 0:1, :2]
    pred_neighbors = predictions[:, input_frames:, 1:, :2]
    task = config["MODEL"].get("task")

    if task == "neighbor":
        loss_neighbors = trajectory_loss(
            pred_neighbors,
            gt_neighbors,
            mask=neighbor_mask,
            ego=False,
        )
        return loss_neighbors, {
            "loss_total": loss_neighbors.detach(),
            "loss_neigh": loss_neighbors.detach(),
        }

    if task != "ego":
        raise ValueError("Unknown task type in config['MODEL']['task']")

    loss_ego = trajectory_loss(
        pred_ego, gt_ego, mask=ego_mask, ego=True
    )
    forecasts = pred_neighbors.detach()
    threshold = config["EVAL"].get("col_threshold", 0.6)
    plan_cost = collision_loss(
        pred_ego.squeeze(-2),
        forecasts,
        mask_neigh=neighbor_mask,
        threshold=threshold,
    )
    gt_cost = collision_loss(
        gt_ego.squeeze(-2),
        forecasts,
        mask_neigh=neighbor_mask,
        threshold=threshold,
    )
    collision_difference = plan_cost - gt_cost
    total_loss = (
        loss_ego
        + float(config["TRAIN"].get("col_weight", 1.0))
        * collision_difference
    )
    return total_loss, {
        "loss_total": total_loss.detach(),
        "loss_ego": loss_ego.detach(),
        "plan_cost": plan_cost.detach(),
        "col_plan_diff": collision_difference.detach(),
    }


def evaluate_task_loss(model, dataloader, config):
    bar = create_progress_bar("EVAL", fill="#", max=len(dataloader))
    total_average = RunningAverage()
    ego_average = RunningAverage()
    neighbor_average = RunningAverage()
    collision_average = RunningAverage()

    ego_weight = config["EVAL"].get("val_loss_weight_ego", 1.0)
    neighbor_weight = config["EVAL"].get("val_loss_weight_neigh", 1.0)
    collision_weight = config["EVAL"].get("val_loss_weight_col", 0.01)
    threshold = config["EVAL"].get("col_threshold", 0.6)

    model.eval()
    with torch.no_grad():
        for joints, masks, padding_mask in dataloader:
            in_joints, _, out_joints, out_masks, padding_mask = (
                batch_process_coords(
                    joints, masks, padding_mask, config
                )
            )
            device = config["DEVICE"]
            in_joints = in_joints.to(device)
            out_joints = out_joints.to(device)
            out_masks = out_masks.to(device)
            padding_mask = padding_mask.to(device)

            input_frames = in_joints.shape[1]
            predictions = model(in_joints, padding_mask)
            pred_ego = predictions[:, input_frames:, 0:1, :2]
            pred_neighbors = predictions[:, input_frames:, 1:, :2]
            gt_ego, gt_neighbors, ego_mask, neighbor_mask = split_targets(
                out_joints, out_masks, model
            )

            loss_ego = trajectory_loss(
                pred_ego, gt_ego, mask=ego_mask, ego=True
            )
            loss_neighbors = trajectory_loss(
                pred_neighbors,
                gt_neighbors,
                mask=neighbor_mask,
                ego=False,
            )
            plan_cost = collision_loss(
                pred_ego.squeeze(-2),
                pred_neighbors,
                mask_neigh=neighbor_mask,
                threshold=threshold,
            )

            task = config["MODEL"].get("task")
            if task == "pretrain":
                total_loss = (
                    ego_weight * loss_ego
                    + neighbor_weight * loss_neighbors
                    + collision_weight * plan_cost
                )
            elif task == "ego":
                total_loss = (
                    ego_weight * loss_ego
                    + collision_weight * plan_cost
                )
            elif task == "neighbor":
                total_loss = neighbor_weight * loss_neighbors
            else:
                raise ValueError(f"Unknown task type: {task}")

            batch_size = len(in_joints)
            total_average.add(total_loss.item(), batch_size)
            ego_average.add(loss_ego.item(), batch_size)
            neighbor_average.add(loss_neighbors.item(), batch_size)
            collision_average.add(plan_cost.item(), batch_size)
            bar.next()

    bar.finish()
    metrics = {
        "loss_total": total_average.mean,
        "loss_ego": ego_average.mean,
        "loss_neigh": neighbor_average.mean,
        "collision_cost": collision_average.mean,
    }
    return total_average.mean, metrics
