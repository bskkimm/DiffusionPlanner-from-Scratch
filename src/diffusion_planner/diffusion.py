"""Variance-preserving diffusion utilities used by the planner."""

import torch


class VPSchedule:
    """Continuous linear-beta VP schedule from the paper's implementation family."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max

    def mean_scale(self, time: torch.Tensor) -> torch.Tensor:
        log_alpha = -0.25 * time.square() * (self.beta_max - self.beta_min) - 0.5 * self.beta_min * time
        return log_alpha.exp()

    def std(self, time: torch.Tensor) -> torch.Tensor:
        return (1 - self.mean_scale(time).square()).clamp_min(1e-6).sqrt()

    def add_noise(self, clean: torch.Tensor, time: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        return self.mean_scale(time).view(-1, 1, 1) * clean + self.std(time).view(-1, 1, 1) * noise
