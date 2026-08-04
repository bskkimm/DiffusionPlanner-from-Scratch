"""Faithful structured-context encoder from the official Diffusion Planner."""

import torch
from torch import nn

from diffusion_planner.config import PlannerConfig
from diffusion_planner.model.mixer import DropPath, Mlp, MixerBlock


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.drop_path = DropPath(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        return x + self.drop_path(self.mlp(self.norm2(x)))


class AgentFusionEncoder(nn.Module):
    """Encode `(x,y,cos,sin,vx,vy,w,l,type[3])` histories per neighbor."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.type_emb = nn.Linear(3, 128)
        self.channel_pre_project = Mlp(9, 128, 128)
        self.token_pre_project = Mlp(config.time_len, 64, 64)
        self.blocks = nn.ModuleList(MixerBlock(64, 128, config.encoder_drop_path_rate) for _ in range(config.encoder_depth))
        self.norm = nn.LayerNorm(128)
        self.emb_project = Mlp(128, config.hidden_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, agents, frames, _ = x.shape
        types = x[:, :, -1, 8:]
        states = x[..., :8]
        pos = states[:, :, -1, :7].clone()
        pos[..., -3:] = 0
        pos[..., -3] = 1
        vector_mask = states.abs().sum(dim=-1) == 0
        token_mask = (~vector_mask).sum(dim=-1) == 0
        states = torch.cat([states, (~vector_mask).float().unsqueeze(-1)], dim=-1).reshape(batch * agents, frames, 9)
        valid = ~token_mask.reshape(-1)
        result = states.new_zeros(batch * agents, self.emb_project.fc2.out_features)
        if valid.any():
            encoded = self.channel_pre_project(states[valid]).transpose(1, 2)
            encoded = self.token_pre_project(encoded).transpose(1, 2)
            for block in self.blocks:
                encoded = block(encoded)
            encoded = encoded.mean(dim=1) + self.type_emb(types.reshape(batch * agents, 3)[valid])
            result[valid] = self.emb_project(self.norm(encoded))
        return result.reshape(batch, agents, -1), token_mask, pos


class StaticFusionEncoder(nn.Module):
    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.projection = Mlp(config.static_objects_state_dim, config.hidden_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, objects, _ = x.shape
        pos = x[:, :, :7].clone()
        pos[..., -3:] = 0
        pos[..., -2] = 1
        mask = x[..., :10].abs().sum(dim=-1) == 0
        valid = ~mask.reshape(-1)
        result = x.new_zeros(batch * objects, self.hidden_dim)
        if valid.any():
            result[valid] = self.projection(x.reshape(batch * objects, -1)[valid])
        return result.reshape(batch, objects, -1), mask, pos


class LaneFusionEncoder(nn.Module):
    """Encode lane geometry, speed-limit availability, and traffic-light state."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.lane_len = config.lane_len
        self.speed_limit_emb = nn.Linear(1, 128)
        self.unknown_speed_emb = nn.Embedding(1, 128)
        self.traffic_emb = nn.Linear(4, 128)
        self.channel_pre_project = Mlp(8, 128, 128)
        self.token_pre_project = Mlp(config.lane_len, 64, 64)
        self.blocks = nn.ModuleList(MixerBlock(64, 128, config.encoder_drop_path_rate) for _ in range(config.encoder_depth))
        self.norm = nn.LayerNorm(128)
        self.emb_project = Mlp(128, config.hidden_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor, speed_limit: torch.Tensor, has_speed_limit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, lanes, points, _ = x.shape
        traffic = x[:, :, 0, 8:]
        states = x[..., :8]
        pos = states[:, :, self.lane_len // 2, :7].clone()
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2], pos[..., 3] = heading.cos(), heading.sin()
        pos[..., -3:] = 0
        pos[..., -1] = 1
        vector_mask = states.abs().sum(dim=-1) == 0
        token_mask = (~vector_mask).sum(dim=-1) == 0
        valid = ~token_mask.reshape(-1)
        result = states.new_zeros(batch * lanes, self.emb_project.fc2.out_features)
        if valid.any():
            encoded = self.channel_pre_project(states.reshape(batch * lanes, points, 8)[valid]).transpose(1, 2)
            encoded = self.token_pre_project(encoded).transpose(1, 2)
            for block in self.blocks:
                encoded = block(encoded)
            encoded = encoded.mean(dim=1)
            speed = speed_limit.reshape(batch * lanes, 1)[valid]
            known = has_speed_limit.reshape(batch * lanes)[valid].bool()
            speed_embedding = encoded.new_zeros(encoded.shape)
            if known.any():
                speed_embedding[known] = self.speed_limit_emb(speed[known])
            if (~known).any():
                speed_embedding[~known] = self.unknown_speed_emb.weight.expand((~known).sum(), -1)
            encoded = encoded + speed_embedding + self.traffic_emb(traffic.reshape(batch * lanes, 4)[valid])
            result[valid] = self.emb_project(self.norm(encoded))
        return result.reshape(batch, lanes, -1), token_mask, pos


class FusionEncoder(nn.Module):
    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(SelfAttentionBlock(config.hidden_dim, config.num_heads, config.encoder_drop_path_rate) for _ in range(config.encoder_depth))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.clone()
        mask[:, 0] = False
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class ContextEncoder(nn.Module):
    """Official concatenation of agent, static-object, and lane context tokens."""

    def __init__(self, config: PlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.agents = AgentFusionEncoder(config)
        self.static = StaticFusionEncoder(config)
        self.lanes = LaneFusionEncoder(config)
        self.position = nn.Linear(7, config.hidden_dim)
        self.fusion = FusionEncoder(config)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        agents, agent_mask, agent_pos = self.agents(inputs["neighbor_agents_past"])
        static, static_mask, static_pos = self.static(inputs["static_objects"])
        lanes, lane_mask, lane_pos = self.lanes(inputs["lanes"], inputs["lanes_speed_limit"], inputs["lanes_has_speed_limit"])
        tokens = torch.cat([agents, static, lanes], dim=1)
        mask = torch.cat([agent_mask, static_mask, lane_mask], dim=1)
        positions = torch.cat([agent_pos, static_pos, lane_pos], dim=1)
        tokens = tokens + self.position(positions) * (~mask).unsqueeze(-1)
        return {"encoding": self.fusion(tokens, mask), "mask": mask}
