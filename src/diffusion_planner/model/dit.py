"""A small DiT-style trajectory denoiser."""

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    values = timesteps[:, None] * frequencies[None]
    embedding = torch.cat([values.cos(), values.sin()], dim=-1)
    return torch.cat([embedding, embedding[:, :1]], dim=-1)[:, :dimension]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Self-attention, context cross-attention, and gated feed-forward block."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

    def forward(self, x: torch.Tensor, context: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(condition).chunk(6, dim=-1)
        update, _ = self.self_attn(modulate(self.norm1(x), shift1, scale1), modulate(self.norm1(x), shift1, scale1), modulate(self.norm1(x), shift1, scale1), key_padding_mask=mask)
        x = x + gate1[:, None] * update
        query = modulate(self.norm2(x), shift2, scale2)
        update, _ = self.cross_attn(query, context, context)
        x = x + update
        return x + gate2[:, None] * self.ffn(self.norm3(x))


class TrajectoryDiT(nn.Module):
    """Predict clean trajectory states from noisy states and structured context."""

    def __init__(self, trajectory_dim: int, hidden_dim: int, heads: int, depth: int) -> None:
        super().__init__()
        self.input = nn.Linear(trajectory_dim, hidden_dim)
        self.participant_embedding = nn.Embedding(2, hidden_dim)
        self.time = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.blocks = nn.ModuleList(DiTBlock(hidden_dim, heads) for _ in range(depth))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, trajectory_dim))

    def forward(self, noisy: torch.Tensor, time: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, participants, _ = noisy.shape
        x = self.input(noisy)
        ids = torch.zeros(participants, dtype=torch.long, device=noisy.device)
        ids[1:] = 1
        x = x + self.participant_embedding(ids)[None]
        condition = self.time(timestep_embedding(time, x.shape[-1]))
        participant_mask = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=noisy.device), mask[:, : participants - 1]], dim=1)
        for block in self.blocks:
            x = block(x, context, condition, participant_mask)
        return self.output(x)
