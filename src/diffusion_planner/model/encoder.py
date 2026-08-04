"""Structured agent, static-object, lane, and route encoders."""

import torch
from torch import nn

from diffusion_planner.config import PlannerConfig
from diffusion_planner.model.mixer import MixerBlock


class VectorEncoder(nn.Module):
    """Encode a fixed number of vectors with masking and MLP-Mixer layers."""

    def __init__(self, input_dim: int, vector_count: int, token_count: int, config: PlannerConfig) -> None:
        super().__init__()
        self.vector_count = vector_count
        self.project = nn.Linear(input_dim, config.mixer_channels)
        self.mixers = nn.Sequential(
            *(MixerBlock(token_count, config.mixer_channels) for _ in range(config.encoder_depth))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.mixer_channels),
            nn.Linear(config.mixer_channels, config.hidden_dim),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, vectors, features = x.shape
        if count != self.vector_count or features != self.input_dim:
            raise ValueError(f"Expected (..., {self.vector_count}, {self.input_dim}), got {tuple(x.shape)}")
        valid = x.abs().sum(dim=(-1, -2)) > 0
        if vectors != self.mixers[0].token_mlp[0].in_features:
            raise ValueError("The vector length does not match the configured vector length.")
        encoded = self.mixers(self.project(x.reshape(batch * count, vectors, features)))
        encoded = self.output(encoded.mean(dim=1)).reshape(batch, count, -1)
        return encoded, ~valid


class ContextEncoder(nn.Module):
    """Aggregate structured context into tokens consumed by the DiT decoder."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.agents = VectorEncoder(11, config.neighbor_count, config.history_steps, config)
        self.static = nn.Sequential(nn.Linear(10, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.lanes = VectorEncoder(12, config.lane_count, config.lane_points, config)
        self.position = nn.Linear(7, config.hidden_dim)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.hidden_dim, config.heads, batch_first=True, norm_first=True),
            num_layers=config.encoder_depth,
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        agents, agent_mask = self.agents(inputs["neighbor_agents_past"])
        static = inputs["static_objects"]
        static_mask = static.abs().sum(dim=-1) == 0
        static_tokens = self.static(static)
        lanes, lane_mask = self.lanes(inputs["lanes"])
        tokens = torch.cat([agents, static_tokens, lanes], dim=1)
        mask = torch.cat([agent_mask, static_mask, lane_mask], dim=1)
        self_token = inputs["ego_current_state"][:, None, :]
        self_token = torch.cat([self_token, torch.zeros_like(self_token[:, :, :3])], dim=-1)
        tokens = tokens + self.position(torch.cat([tokens.new_zeros(tokens.shape[0], tokens.shape[1], 3), tokens.new_zeros(tokens.shape[0], tokens.shape[1], 4)], dim=-1))
        tokens[:, 0] = tokens[:, 0] + self.position(self_token).squeeze(1)
        return self.fusion(tokens, src_key_padding_mask=mask), mask
