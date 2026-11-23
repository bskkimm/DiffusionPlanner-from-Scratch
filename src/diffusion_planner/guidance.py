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
