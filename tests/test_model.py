import torch

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.guidance import collision_energy


def inputs(config: PlannerConfig, batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "ego_current_state": torch.randn(batch, 4),
        "neighbor_agents_past": torch.randn(batch, config.neighbor_count, config.history_steps, 11),
        "static_objects": torch.randn(batch, config.static_count, 10),
        "lanes": torch.randn(batch, config.lane_count, config.lane_points, 12),
    }


def test_forward_loss_and_sampling_shapes() -> None:
    config = PlannerConfig(hidden_dim=32, heads=4, encoder_depth=1, decoder_depth=1)
    model = DiffusionPlanner(config)
    batch = inputs(config)
    target = torch.randn(2, config.participant_count, config.future_steps + 1, 4)
    loss = model.loss(batch, target)
    assert loss.ndim == 0
    prediction = model.sample(batch, steps=2)
    assert prediction.shape == (2, config.participant_count, config.future_steps, 4)


def test_collision_energy_is_differentiable() -> None:
    trajectory = torch.randn(2, 3, 5, 4, requires_grad=True)
    energy = collision_energy(trajectory)
    energy.sum().backward()
    assert trajectory.grad is not None
