"""Continuous linear VP-SDE used by the official planner."""

import torch


class VPSchedule:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max

    def marginal_prob(self, x: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (-1,) + (1,) * (x.ndim - 1)
        time = time.reshape(shape)
        mean_log_coefficient = -0.25 * time.square() * (self.beta_max - self.beta_min) - 0.5 * self.beta_min * time
        return mean_log_coefficient.exp() * x, (1 - (2 * mean_log_coefficient).exp()).sqrt()

    def std(self, time: torch.Tensor) -> torch.Tensor:
        return (1 - torch.exp(-0.5 * time.square() * (self.beta_max - self.beta_min) - self.beta_min * time)).sqrt()

    def add_noise(self, clean: torch.Tensor, time: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        mean, std = self.marginal_prob(clean, time)
        return mean + std * (torch.randn_like(clean) if noise is None else noise)
