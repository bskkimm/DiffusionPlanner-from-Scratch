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


@torch.no_grad()
def sample_dpm_solver(
    model,
    encoded,
    inputs,
    current,
    noise,
    *,
    steps: int = 10,
    eps: float = 1e-3,
    energy_fn=None,
    guidance_scale: float = 0.0,
    score_fn=None,
    score_scale: float = 0.0,
):
    """Second-order multistep DPM-Solver++ with log-SNR steps and final denoising.

    This implements the released solver's unguided x-start settings and retains
    the supplied current state. Numerical parity is tested on three fixed CPU fixtures;
    model/checkpoint parity and guided sampling remain separate requirements.
    An optional differentiable energy receives normalized predicted trajectories;
    its gradient provides experimental guidance. An optional score function receives
    a stop-gradient denoiser proxy and time, matching the released wrapper semantics.
    """
    if getattr(getattr(model, "config", None), "diffusion_model_type", "x_start") != "x_start":
        raise ValueError("sampler requires an x_start model")
    if steps < 2 or not 0 < eps < 1:
        raise ValueError("at least two steps and 0 < eps < 1 are required")
    if noise.ndim != 4 or current.shape != noise.shape[:2] + (4,) or noise.shape[2] < 2:
        raise ValueError("current and noise have incompatible shapes")
    if guidance_scale < 0 or (guidance_scale and energy_fn is None):
        raise ValueError("nonnegative guidance requires an energy function")
    if score_scale < 0 or (score_scale and score_fn is None):
        raise ValueError("nonnegative score guidance requires a score function")
    schedule = VPSchedule()

    def predict(trajectory, t):
        if (energy_fn is None or guidance_scale == 0) and (score_fn is None or score_scale == 0):
            prediction = model.denoise(trajectory.flatten(2), t, encoded, inputs).reshape_as(
                trajectory
            )
            prediction[:, :, 0] = current
            return prediction
        with torch.enable_grad():
            variable = trajectory.detach().requires_grad_(True)
            raw = model.denoise(variable.flatten(2), t, encoded, inputs).reshape_as(variable)
            prediction = torch.cat((current.unsqueeze(2), raw[:, :, 1:]), dim=2)
            correction = torch.zeros_like(variable)
            if energy_fn is not None and guidance_scale:
                energy_gradient = torch.autograd.grad(energy_fn(prediction).sum(), variable)[0]
                correction = correction - guidance_scale * energy_gradient
            if score_fn is not None and score_scale:
                proxy = variable + (prediction.detach() - variable.detach())
                score_gradient = torch.autograd.grad(score_fn(proxy, t).sum(), variable)[0]
                correction = correction + score_scale * score_gradient
        if not torch.isfinite(correction).all():
            raise FloatingPointError("non-finite guidance gradient")
        alpha, sigma = schedule.marginal_prob(torch.ones_like(variable), t)
        guided = prediction.detach() + sigma.square() / alpha * correction
        guided[:, :, 0] = current
        return guided

    def log_snr(t):
        log_alpha = -0.25 * t.square() * 19.9 - 0.05 * t
        return log_alpha - 0.5 * torch.log1p(-torch.exp(2 * log_alpha))

    endpoints = torch.tensor([1.0, eps], device=noise.device, dtype=noise.dtype)
    lambdas = torch.linspace(
        log_snr(endpoints)[0],
        log_snr(endpoints)[1],
        steps + 1,
        device=noise.device,
        dtype=noise.dtype,
    )
    tmp = 39.8 * torch.logaddexp(-2 * lambdas, torch.zeros_like(lambdas))
    times = tmp / (torch.sqrt(0.01 + tmp) + 0.1) / 19.9
    x = noise.clone()
    x[:, :, 0] = current
    previous = None
    previous_lambda = None
    for index in range(steps):
        t = times[index].expand(x.shape[0])
        prediction = predict(x, t)
        h = lambdas[index + 1] - lambdas[index]
        correction = prediction
        if previous is not None and not (steps < 10 and index == steps - 1):
            ratio = (lambdas[index] - previous_lambda) / h
            correction = prediction + (prediction - previous) / (2 * ratio)
        alpha_next, sigma_next = schedule.marginal_prob(
            torch.ones_like(x), times[index + 1].expand(x.shape[0])
        )
        sigma = schedule.std(t).reshape(-1, 1, 1, 1)
        x = sigma_next / sigma * x - alpha_next * torch.expm1(-h) * correction
        x[:, :, 0] = current
        previous, previous_lambda = prediction, lambdas[index]
    return predict(x, times[-1].expand(x.shape[0]))
