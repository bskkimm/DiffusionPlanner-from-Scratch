"""Official Diffusion Planner architecture configuration."""

from dataclasses import dataclass


@dataclass
class PlannerConfig:
    """Defaults from the official released checkpoint ``args.json``."""

    future_len: int = 80
    time_len: int = 21
    agent_state_dim: int = 11
    agent_num: int = 32
    static_objects_state_dim: int = 10
    static_objects_num: int = 5
    lane_len: int = 20
    lane_state_dim: int = 12
    lane_num: int = 70
    route_len: int = 20
    route_state_dim: int = 12
    route_num: int = 25
    predicted_neighbor_num: int = 10
    hidden_dim: int = 192
    num_heads: int = 6
    encoder_depth: int = 3
    decoder_depth: int = 3
    encoder_drop_path_rate: float = 0.1
    decoder_drop_path_rate: float = 0.1
    diffusion_model_type: str = "x_start"

    @property
    def participant_num(self) -> int:
        return 1 + self.predicted_neighbor_num

    @property
    def trajectory_dim(self) -> int:
        return (self.future_len + 1) * 4
