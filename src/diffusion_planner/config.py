"""Small, explicit configuration for the study implementation."""

from dataclasses import dataclass


@dataclass
class PlannerConfig:
    """Tensor dimensions used by the structured toy/nuPlan-like interface."""

    history_steps: int = 8
    future_steps: int = 12
    neighbor_count: int = 4
    static_count: int = 4
    lane_count: int = 8
    lane_points: int = 12
    route_count: int = 2
    hidden_dim: int = 96
    heads: int = 4
    encoder_depth: int = 2
    decoder_depth: int = 2
    mixer_tokens: int = 16
    mixer_channels: int = 64
    beta_min: float = 0.1
    beta_max: float = 20.0

    @property
    def participant_count(self) -> int:
        return 1 + self.neighbor_count

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def trajectory_dim(self) -> int:
        return (self.future_steps + 1) * self.state_dim
