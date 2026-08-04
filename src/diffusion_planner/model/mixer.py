"""MLP-Mixer blocks used to turn sparse vectors into dense tokens."""

import torch
from torch import nn


class MixerBlock(nn.Module):
    """Mix information across sequence positions and feature channels."""

    def __init__(self, tokens: int, channels: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(tokens, tokens),
            nn.GELU(),
            nn.Linear(tokens, tokens),
        )
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_update = self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + token_update
        return x + self.channel_mlp(self.channel_norm(x))
