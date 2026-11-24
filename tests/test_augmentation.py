import torch

from diffusion_planner.augmentation import quintic_positions


def test_quintic_interpolation_respects_constant_velocity():
    initial = torch.tensor([[0.0, 2.0]])
    velocity = torch.tensor([[1.0, -1.0]])
    acceleration = torch.zeros_like(initial)
    horizon = 2.0
    times = torch.tensor([0.0, 0.5, 1.0, 2.0])
    actual = quintic_positions(
        initial,
        velocity,
        acceleration,
        initial + horizon * velocity,
        velocity,
        acceleration,
        horizon,
        times,
    )
    expected = initial[:, None, :] + times[None, :, None] * velocity[:, None, :]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_ego_frame_rotation_preserves_padding_and_transforms_future():
    from diffusion_planner.augmentation import to_ego_frame

    inputs = {
        "ego_current_state": torch.tensor([[10.0, 5.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        "neighbor_agents_past": torch.zeros(1, 1, 1, 11),
        "static_objects": torch.zeros(1, 1, 10),
        "lanes": torch.zeros(1, 1, 1, 12),
        "route_lanes": torch.zeros(1, 1, 1, 12),
    }
    ego = torch.tensor([[[10.0, 7.0, torch.pi / 2]]])
    neighbors = torch.zeros(1, 1, 1, 3)
    transformed, target, others = to_ego_frame(inputs, ego, neighbors)
    torch.testing.assert_close(target[0, 0], torch.tensor([2.0, 0.0, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        transformed["ego_current_state"][0, :4], torch.tensor([0.0, 0.0, 1.0, 0.0])
    )
    assert transformed["lanes"].count_nonzero() == 0
    assert others.count_nonzero() == 0
    assert inputs["ego_current_state"][0, 0] == 10


def test_zero_probability_augmentation_only_changes_coordinate_frame():
    from diffusion_planner.augmentation import perturb_and_reframe

    inputs = {
        "ego_current_state": torch.tensor([[0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "neighbor_agents_past": torch.zeros(1, 1, 2, 11),
        "static_objects": torch.zeros(1, 1, 10),
        "lanes": torch.zeros(1, 1, 2, 12),
        "route_lanes": torch.zeros(1, 1, 2, 12),
    }
    ego = torch.zeros(1, 21, 3)
    ego[0, :, 0] = torch.arange(1, 22) * 0.5
    neighbors = torch.zeros(1, 1, 21, 3)
    actual_inputs, actual_ego, actual_neighbors = perturb_and_reframe(
        inputs, ego, neighbors, probability=0.0
    )
    torch.testing.assert_close(actual_ego, ego)
    torch.testing.assert_close(actual_neighbors, neighbors)
    torch.testing.assert_close(actual_inputs["ego_current_state"], inputs["ego_current_state"])


def test_perturbation_matches_upstream_cpu_fixture():
    """Fixture from upstream a3a621f with Pacifica wheelbase 3.089 m."""
    from diffusion_planner.augmentation import perturb_and_reframe

    inputs = {
        "ego_current_state": torch.tensor([[0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "neighbor_agents_past": torch.zeros(1, 1, 2, 11),
        "static_objects": torch.zeros(1, 1, 10),
        "lanes": torch.zeros(1, 1, 2, 12),
        "route_lanes": torch.zeros(1, 1, 2, 12),
    }
    ego = torch.zeros(1, 21, 3)
    ego[0, :, 0] = torch.arange(1, 22) * 0.5
    neighbors = torch.zeros(1, 1, 21, 3)
    torch.manual_seed(42)
    _, actual, _ = perturb_and_reframe(inputs, ego, neighbors)
    expected = torch.tensor(
        [
            [0.45901775, -0.13679464, -0.28963563],
            [2.79610634, -0.83889902, -0.29420137],
            [9.54293060, -2.99332476, -0.32086667],
        ]
    )
    torch.testing.assert_close(actual[0, [0, 5, 19]], expected, rtol=1e-4, atol=1e-4)
