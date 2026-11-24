"""Faithful DiT decoder modules from the official Diffusion Planner."""

import math

import torch
from torch import nn

from diffusion_planner.config import PlannerConfig
from diffusion_planner.diffusion import VPSchedule
from diffusion_planner.model.mixer import MixerBlock, Mlp


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        embedding = torch.cat(
            [
                (t[:, None].float() * frequencies[None]).cos(),
                (t[:, None].float() * frequencies[None]).sin(),
            ],
            dim=-1,
        )
        return (
            torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            if dim % 2
            else embedding
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DiTBlock(nn.Module):
    """Official adaLN-Zero self-attention/MLP followed by cross-attention/MLP."""

    def __init__(self, dim: int, heads: int, dropout: float, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(dim, hidden, approximate="tanh")
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(dim, hidden, approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,
        cross_c: torch.Tensor,
        condition: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            condition
        ).chunk(6, dim=1)
        modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(modulated, modulated, modulated, key_padding_mask=attn_mask)[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # These two assignments intentionally match the official released implementation.
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        return self.mlp2(self.norm4(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.proj(modulate(self.norm_final(x), shift, scale))


class RouteEncoder(nn.Module):
    """Encode all route-lane points into the navigation condition token."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.channel_pre_project = Mlp(4, 64, 64)
        self.token_pre_project = Mlp(config.route_num * config.lane_len, 32, 32)
        self.mixer = MixerBlock(32, 64, config.encoder_drop_path_rate)
        self.norm = nn.LayerNorm(64)
        self.emb_project = Mlp(
            64, config.hidden_dim, config.hidden_dim, drop=config.encoder_drop_path_rate
        )

    def forward(self, route_lanes: torch.Tensor) -> torch.Tensor:
        x = route_lanes[..., :4]
        batch, routes, points, _ = x.shape
        point_mask = x.abs().sum(dim=-1) == 0
        lane_mask = (~point_mask).sum(dim=-1) == 0
        valid = ~(lane_mask.sum(dim=-1) == routes)
        flattened = x.reshape(batch, routes * points, 4)
        result = x.new_zeros(batch, self.emb_project.fc2.out_features)
        if valid.any():
            encoded = self.channel_pre_project(flattened[valid]).transpose(1, 2)
            encoded = self.token_pre_project(encoded).transpose(1, 2)
            encoded = self.mixer(encoded).mean(dim=1)
            result[valid] = self.emb_project(self.norm(encoded))
        return result


class TrajectoryDiT(nn.Module):
    """Official route-conditioned DiT trajectory model."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.model_type = config.diffusion_model_type
        if self.model_type not in {"score", "x_start"}:
            raise ValueError("diffusion_model_type must be 'score' or 'x_start'.")
        self.schedule = VPSchedule()
        self.route_encoder = RouteEncoder(config)
        self.agent_embedding = nn.Embedding(2, config.hidden_dim)
        self.preproj = Mlp(config.trajectory_dim, 512, config.hidden_dim)
        self.t_embedder = TimestepEmbedder(config.hidden_dim)
        self.blocks = nn.ModuleList(
            DiTBlock(config.hidden_dim, config.num_heads, config.decoder_drop_path_rate)
            for _ in range(config.decoder_depth)
        )
        self.final_layer = FinalLayer(config.hidden_dim, config.trajectory_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_c: torch.Tensor,
        route_lanes: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, participants, _ = x.shape
        x = self.preproj(x)
        identities = torch.zeros(participants, dtype=torch.long, device=x.device)
        identities[1:] = 1
        x = x + self.agent_embedding(identities)[None]
        condition = self.route_encoder(route_lanes) + self.t_embedder(t)
        mask = torch.zeros(batch, participants, dtype=torch.bool, device=x.device)
        mask[:, 1:] = neighbor_current_mask
        for block in self.blocks:
            x = block(x, cross_c, condition, mask)
        x = self.final_layer(x, condition)
        if self.model_type == "score":
            return x / (self.schedule.std(t)[:, None, None] + 1e-6)
        return x
