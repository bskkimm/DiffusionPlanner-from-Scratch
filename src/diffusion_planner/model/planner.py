"""End-to-end structured diffusion planner."""

from collections.abc import Callable

import torch
from torch import nn

from diffusion_planner.config import PlannerConfig
from diffusion_planner.diffusion import VPSchedule
from diffusion_planner.model.dit import TrajectoryDiT
from diffusion_planner.model.encoder import ContextEncoder


class DiffusionPlanner(nn.Module):
    """Jointly denoise ego and neighbor future trajectories."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PlannerConfig()
        self.encoder = ContextEncoder(self.config)
        self.denoiser = TrajectoryDiT(self.config.trajectory_dim, self.config.hidden_dim, self.config.heads, self.config.decoder_depth)
        self.schedule = VPSchedule(self.config.beta_min, self.config.beta_max)

    def encode(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(inputs)

    def loss(self, inputs: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        context, mask = self.encode(inputs)
        time = torch.rand(target.shape[0], device=target.device).clamp(1e-4, 1 - 1e-4)
        noisy = self.schedule.add_noise(target.reshape(target.shape[0], target.shape[1], -1), time)
        prediction = self.denoiser(noisy, time, context, mask)
        return (prediction - target.reshape(target.shape[0], target.shape[1], -1)).square().mean()

    @torch.no_grad()
    def sample(self, inputs: dict[str, torch.Tensor], steps: int = 20, energy_fn: Callable | None = None, guidance_scale: float = 0.1) -> torch.Tensor:
        context, mask = self.encode(inputs)
        batch = context.shape[0]
        participants = self.config.participant_count
        state = torch.randn(batch, participants, self.config.trajectory_dim, device=context.device)
        current = torch.cat([inputs["ego_current_state"][:, None], inputs["neighbor_agents_past"][:, :, -1, :4]], dim=1)
        times = torch.linspace(1.0, 1e-3, steps + 1, device=state.device)
        for index in range(steps):
            time = times[index].expand(batch)
            prediction = self.denoiser(state, time, context, mask)
            if energy_fn is not None:
                with torch.enable_grad():
                    from diffusion_planner.guidance import guided_x_start

                    prediction = guided_x_start(self.denoiser, state, time, context, mask, energy_fn, guidance_scale).detach()
            state = prediction + (times[index + 1] / times[index]).clamp(0, 1) * (state - prediction)
            state = state.reshape(batch, participants, -1, 4)
            state[:, :, 0] = current
            state = state.reshape(batch, participants, -1)
        return state.reshape(batch, participants, -1, 4)[:, :, 1:]
