"""Differentiable, training-free trajectory energies."""

import torch


def comfort_energy(trajectory: torch.Tensor, acceleration_limit: float = 3.0) -> torch.Tensor:
    """Penalize squared acceleration of the ego trajectory."""
    positions = trajectory[:, 0, :, :2]
    acceleration = positions[:, 2:] - 2 * positions[:, 1:-1] + positions[:, :-2]
    return (acceleration.square().sum(dim=-1) - acceleration_limit**2).relu().mean(dim=-1)


def collision_energy(trajectory: torch.Tensor, radius: float = 2.0) -> torch.Tensor:
    """Softly penalize ego-neighbor distances below a collision radius."""
    ego = trajectory[:, :1, :, :2]
    neighbors = trajectory[:, 1:, :, :2]
    distances = (ego - neighbors).square().sum(dim=-1).sqrt().clamp_min(1e-4)
    return (radius - distances).relu().square().mean(dim=(1, 2))


def guided_x_start(
    model,
    noisy: torch.Tensor,
    time: torch.Tensor,
    context: torch.Tensor,
    mask: torch.Tensor,
    energy_fn,
    scale: float = 0.1,
) -> torch.Tensor:
    """Move the denoiser prediction down an energy gradient, without training a classifier."""
    noisy = noisy.detach().requires_grad_(True)
    prediction = model(noisy, time, context, mask)
    energy = energy_fn(prediction.reshape(prediction.shape[0], prediction.shape[1], -1, 4)).sum()
    gradient = torch.autograd.grad(energy, noisy)[0]
    return prediction - scale * gradient
