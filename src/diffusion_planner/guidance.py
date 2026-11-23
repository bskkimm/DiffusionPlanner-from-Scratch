"""Differentiable, training-free trajectory energies."""

import torch


def comfort_energy(trajectory: torch.Tensor, acceleration_limit: float = 3.0) -> torch.Tensor:
    """Penalize squared second position differences; no timestep scaling is applied."""
    positions = trajectory[:, 0, :, :2]
    if positions.shape[1] < 3:
        return positions.sum(dim=(1, 2)) * 0
    acceleration = positions[:, 2:] - 2 * positions[:, 1:-1] + positions[:, :-2]
    return (acceleration.square().sum(dim=-1) - acceleration_limit**2).relu().mean(dim=-1)


def collision_energy(trajectory: torch.Tensor, radius: float = 2.0) -> torch.Tensor:
    """Penalize close neighbors; caller must exclude padded participants."""
    if trajectory.shape[1] < 2 or trajectory.shape[2] == 0:
        return trajectory.sum(dim=(1, 2, 3)) * 0
    ego = trajectory[:, :1, :, :2]
    neighbors = trajectory[:, 1:, :, :2]
    distances = (ego - neighbors).square().sum(dim=-1).clamp_min(1e-8).sqrt()
    return (radius - distances).relu().square().mean(dim=(1, 2))


def guided_x_start(
    model,
    noisy: torch.Tensor,
    time: torch.Tensor,
    context: dict[str, torch.Tensor],
    mask: dict[str, torch.Tensor],
    energy_fn,
    scale: float = 0.1,
) -> torch.Tensor:
    """Apply a heuristic energy-gradient update using the planner denoiser interface.

    This is not the released classifier-guidance wrapper. The caller controls
    energy units, padding and current-state preservation.
    """
    with torch.enable_grad():
        noisy = noisy.detach().requires_grad_(True)
        prediction = model.denoise(noisy, time, context, mask)
        energy = energy_fn(
            prediction.reshape(prediction.shape[0], prediction.shape[1], -1, 4)
        ).sum()
        gradient = torch.autograd.grad(energy, noisy)[0]
    return (prediction - scale * gradient).detach()


def oriented_box_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Signed separating-axis distance between oriented rectangles in meters.

    Inputs broadcast as (..., 6): center x/y, heading cosine/sine, length, width.
    Negative values indicate overlap; nonnegative values are the smallest
    separating-axis gap, which is not Euclidean distance for diagonal separation.
    """
    if first.shape[-1] != 6 or second.shape[-1] != 6:
        raise ValueError("boxes must have six features")
    first, second = torch.broadcast_tensors(first, second)
    direction_a, direction_b = first[..., 2:4], second[..., 2:4]
    norm_a = torch.linalg.vector_norm(direction_a, dim=-1, keepdim=True)
    norm_b = torch.linalg.vector_norm(direction_b, dim=-1, keepdim=True)
    if (norm_a <= 1e-8).any() or (norm_b <= 1e-8).any():
        raise ValueError("box headings must be nonzero")
    if (first[..., 4:] <= 0).any() or (second[..., 4:] <= 0).any():
        raise ValueError("box dimensions must be positive")
    forward_a, forward_b = direction_a / norm_a, direction_b / norm_b
    side_a = torch.stack((-forward_a[..., 1], forward_a[..., 0]), dim=-1)
    side_b = torch.stack((-forward_b[..., 1], forward_b[..., 0]), dim=-1)
    axes = torch.stack((forward_a, side_a, forward_b, side_b), dim=-2)
    delta = second[..., :2] - first[..., :2]
    center_gap = (axes * delta.unsqueeze(-2)).sum(-1).abs()
    radius_a = (axes * forward_a.unsqueeze(-2)).sum(-1).abs() * first[..., 4:5] / 2 + (
        axes * side_a.unsqueeze(-2)
    ).sum(-1).abs() * first[..., 5:6] / 2
    radius_b = (axes * forward_b.unsqueeze(-2)).sum(-1).abs() * second[..., 4:5] / 2 + (
        axes * side_b.unsqueeze(-2)
    ).sum(-1).abs() * second[..., 5:6] / 2
    separation = center_gap - radius_a - radius_b
    overlaps = (separation < 0).all(dim=-1)
    smallest_gap = separation.masked_fill(separation < 0, float("inf")).min(dim=-1).values
    return torch.where(overlaps, separation.max(dim=-1).values, smallest_gap)
