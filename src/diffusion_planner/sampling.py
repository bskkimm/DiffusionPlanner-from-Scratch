"""A deterministic VP schedule sampler for x-start models.

This practical sampler is not the upstream DPM-Solver++ implementation.
"""

import torch

from diffusion_planner.diffusion import VPSchedule


@torch.no_grad()
def sample_ddim(model, encoded, inputs, current, noise, *, steps: int = 20, eps: float = 1e-3):
    """Sample normalized trajectories with a fixed current state.

    ``noise`` is (B, participants, 1 + horizon, 4), and ``current`` is
    (B, participants, 4). Caller supplies encoded context and seeded noise.
    The output is in normalized trajectory coordinates; inverse-normalize future
    states before using them as physical positions.
    """
    if getattr(getattr(model, "config", None), "diffusion_model_type", "x_start") != "x_start":
        raise ValueError("sampler requires an x_start model")
    if steps < 1 or not 0 < eps < 1:
        raise ValueError("steps must be positive and eps must be between zero and one")
    if (
        noise.ndim != 4
        or noise.shape[-1] != 4
        or noise.shape[:2] != current.shape[:2]
        or current.shape[-1] != 4
    ):
        raise ValueError("current and noise have incompatible shapes")
    schedule = VPSchedule()
    x = noise.clone()
    x[:, :, 0] = current
    times = torch.linspace(1.0, eps, steps, device=x.device, dtype=x.dtype)
    for index, scalar in enumerate(times):
        t = scalar.expand(x.shape[0])
        estimate = model.denoise(x.flatten(2), t, encoded, inputs).reshape_as(x)
        estimate[:, :, 0] = current
        if index == steps - 1:
            x = estimate
            break
        alpha, sigma = schedule.marginal_prob(torch.ones_like(x[:, :, 1:]), t)
        predicted_noise = (x[:, :, 1:] - alpha * estimate[:, :, 1:]) / sigma
        next_t = times[index + 1].expand(x.shape[0])
        next_alpha, next_sigma = schedule.marginal_prob(torch.ones_like(x[:, :, 1:]), next_t)
        x = torch.cat(
            (current.unsqueeze(2), next_alpha * estimate[:, :, 1:] + next_sigma * predicted_noise),
            dim=2,
        )
    return x
