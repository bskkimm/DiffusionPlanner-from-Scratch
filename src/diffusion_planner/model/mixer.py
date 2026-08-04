"""The MLP and Mixer primitives used by the official architecture."""

import torch
from torch import nn


class Mlp(nn.Module):
    """Equivalent to timm's two-layer MLP used in the official source."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    """Per-sample stochastic depth, matching timm DropPath semantics."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
        return x * mask / keep


class MixerBlock(nn.Module):
    """Official token-mixing then channel-mixing residual block."""

    def __init__(self, tokens_mlp_dim: int, channels_mlp_dim: int, drop_path_rate: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(channels_mlp_dim, channels_mlp_dim)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(tokens_mlp_dim, tokens_mlp_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tokens_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.drop_path(y)
        return x + self.drop_path(self.channels_mlp(self.norm2(x)))
