import torch

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.model.dit import RouteEncoder
from diffusion_planner.model.encoder import ContextEncoder


def config() -> PlannerConfig:
    return PlannerConfig(
        future_len=3,
        time_len=4,
        agent_num=3,
        static_objects_num=2,
        lane_len=4,
        lane_num=3,
        route_num=2,
        predicted_neighbor_num=2,
        hidden_dim=32,
        num_heads=4,
        encoder_depth=1,
        decoder_depth=1,
    )


def inputs(model_config: PlannerConfig, batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "ego_current_state": torch.randn(batch, 10),
        "neighbor_agents_past": torch.randn(batch, model_config.agent_num, model_config.time_len, 11),
        "static_objects": torch.randn(batch, model_config.static_objects_num, 10),
        "lanes": torch.randn(batch, model_config.lane_num, model_config.lane_len, 12),
        "lanes_speed_limit": torch.rand(batch, model_config.lane_num, 1),
        "lanes_has_speed_limit": torch.ones(batch, model_config.lane_num, 1, dtype=torch.bool),
        "route_lanes": torch.randn(batch, model_config.route_num, model_config.lane_len, 12),
    }


def test_context_and_route_encoders_match_official_token_shapes() -> None:
    model_config = config()
    batch = inputs(model_config)
    context = ContextEncoder(model_config)(batch)
    assert context["encoding"].shape == (2, model_config.agent_num + model_config.static_objects_num + model_config.lane_num, model_config.hidden_dim)
    assert context["mask"].shape == (2, model_config.agent_num + model_config.static_objects_num + model_config.lane_num)
    assert RouteEncoder(model_config)(batch["route_lanes"]).shape == (2, model_config.hidden_dim)


def test_official_training_contract_and_loss() -> None:
    model_config = config()
    model = DiffusionPlanner(model_config).train()
    batch = inputs(model_config)
    ego_future = torch.randn(2, model_config.future_len, 4)
    neighbor_future = torch.randn(2, model_config.predicted_neighbor_num, model_config.future_len, 4)
    neighbor_mask = torch.zeros(2, model_config.predicted_neighbor_num, model_config.future_len, dtype=torch.bool)
    loss = model.diffusion_loss(batch, ego_future, neighbor_future, neighbor_mask)
    total = loss["ego_planning_loss"] + loss["neighbor_prediction_loss"]
    total.backward()
    assert total.ndim == 0
    assert torch.isfinite(total)


def test_denoiser_preserves_official_trajectory_shape() -> None:
    model_config = config()
    model = DiffusionPlanner(model_config).eval()
    batch = inputs(model_config)
    encoded, output = model(batch)
    noisy = torch.randn(2, model_config.participant_num, model_config.trajectory_dim)
    denoised = model.denoise(noisy, torch.full((2,), 0.5), encoded, batch)
    assert output["current_states"].shape == (2, model_config.participant_num, 4)
    assert denoised.shape == (2, model_config.participant_num, model_config.trajectory_dim)
