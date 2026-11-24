"""Faithful encoder-decoder composition and official diffusion loss contract."""

import torch
from torch import nn

from diffusion_planner.config import PlannerConfig
from diffusion_planner.diffusion import VPSchedule
from diffusion_planner.model.dit import TrajectoryDiT
from diffusion_planner.model.encoder import ContextEncoder
from diffusion_planner.normalizer import StateNormalizer


class DiffusionPlanner(nn.Module):
    """Official architecture with train and inference forward paths.

    Training inputs add ``sampled_trajectories`` `(B, 1+Pn, 1+T, 4)` and
    ``diffusion_time`` `(B,)`. Inference starts from a provided ``x_T`` through
    ``denoise`` so a DPM-Solver implementation can drive the same denoiser.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PlannerConfig()
        self.encoder = ContextEncoder(self.config)
        self.decoder = TrajectoryDiT(self.config)
        self.schedule = VPSchedule()
        self.normalizer = StateNormalizer(self.config.participant_num)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(initialize)
        nn.init.normal_(self.encoder.position.weight, std=0.02)
        nn.init.normal_(self.encoder.agents.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lanes.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lanes.traffic_emb.weight, std=0.02)
        nn.init.normal_(self.decoder.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.t_embedder.mlp[2].weight, std=0.02)
        for block in self.decoder.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.final_layer.proj[-1].bias, 0)

    def _current_states(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ego = inputs["ego_current_state"][:, None, :4]
        neighbors = inputs["neighbor_agents_past"][:, : self.config.predicted_neighbor_num, -1, :4]
        neighbor_mask = neighbors.ne(0).any(dim=-1).logical_not()
        return torch.cat([ego, neighbors], dim=1), neighbor_mask

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        encoded = self.encoder(inputs)
        current, neighbor_mask = self._current_states(inputs)
        if self.training or ("sampled_trajectories" in inputs and "diffusion_time" in inputs):
            noisy = inputs["sampled_trajectories"].reshape(
                current.shape[0], self.config.participant_num, -1
            )
            score = self.decoder(
                noisy,
                inputs["diffusion_time"],
                encoded["encoding"],
                inputs["route_lanes"],
                neighbor_mask,
            )
            return encoded, {
                "score": score.reshape(current.shape[0], self.config.participant_num, -1, 4)
            }
        return encoded, {"current_states": current, "neighbor_current_mask": neighbor_mask}

    def denoise(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        encoded: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Denoiser callback for DPM-Solver or any inference integrator."""
        _, mask = self._current_states(inputs)
        return self.decoder(x, time, encoded["encoding"], inputs["route_lanes"], mask)

    def diffusion_loss(
        self,
        inputs: dict[str, torch.Tensor],
        ego_future: torch.Tensor,
        neighbor_future: torch.Tensor,
        neighbor_future_mask: torch.Tensor,
        eps: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        """Exact released loss construction, including current-state preservation."""
        batch, predicted_neighbors, horizon, _ = neighbor_future.shape
        if (
            predicted_neighbors != self.config.predicted_neighbor_num
            or horizon != self.config.future_len
        ):
            raise ValueError(
                "Future tensors do not match the configured official model dimensions."
            )
        current, current_mask = self._current_states(inputs)
        all_mask = torch.cat([current_mask.unsqueeze(-1), neighbor_future_mask], dim=-1)
        future = torch.cat([ego_future[:, None], neighbor_future], dim=1)
        clean = torch.cat([current[:, :, None], self.normalizer(future)], dim=2)
        clean[:, 1:][all_mask] = 0
        time = torch.rand(batch, device=clean.device) * (1 - eps) + eps
        mean, std = self.schedule.marginal_prob(clean[:, :, 1:], time)
        noisy = torch.cat([clean[:, :, :1], mean + std * torch.randn_like(mean)], dim=2)
        _, output = self({**inputs, "sampled_trajectories": noisy, "diffusion_time": time})
        score = output["score"][:, :, 1:]
        if self.config.diffusion_model_type == "score":
            loss = (score * std + (noisy[:, :, 1:] - mean) / std).square().sum(dim=-1)
        else:
            loss = (score - clean[:, :, 1:]).square().sum(dim=-1)
        valid_neighbors = ~neighbor_future_mask
        neighbor_loss = (
            loss[:, 1:][valid_neighbors].mean() if valid_neighbors.any() else loss.new_zeros(())
        )
        return {"ego_planning_loss": loss[:, 0].mean(), "neighbor_prediction_loss": neighbor_loss}
