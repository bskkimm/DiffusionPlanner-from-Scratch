import torch

from diffusion_planner.guidance import collision_energy, comfort_energy


def test_collision_energy_is_finite_with_identical_positions():
    trajectory = torch.zeros(1, 2, 3, 4, requires_grad=True)
    energy = collision_energy(trajectory)
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(trajectory.grad).all()


def test_collision_energy_with_no_neighbors_is_zero():
    trajectory = torch.zeros(2, 1, 3, 4)
    torch.testing.assert_close(collision_energy(trajectory), torch.zeros(2))


def test_comfort_energy_handles_short_trajectory():
    torch.testing.assert_close(comfort_energy(torch.zeros(2, 1, 2, 4)), torch.zeros(2))


def test_guided_x_start_accepts_planner_denoise_interface():
    from diffusion_planner.guidance import guided_x_start

    class Planner:
        def denoise(self, noisy, time, encoded, inputs):
            return noisy * 0.5

    noisy = torch.ones(1, 1, 8)
    prediction = guided_x_start(
        Planner(),
        noisy,
        torch.ones(1),
        {},
        {},
        lambda trajectory: trajectory.square().sum(dim=(1, 2, 3)),
        scale=0.1,
    )
    assert prediction.shape == noisy.shape
    assert torch.all(prediction < 0.5)


def test_guidance_enables_grad_during_inference():
    from diffusion_planner.guidance import guided_x_start

    class Planner:
        def denoise(self, noisy, time, encoded, inputs):
            return noisy * 0.5

    with torch.no_grad():
        prediction = guided_x_start(
            Planner(),
            torch.ones(1, 1, 8),
            torch.ones(1),
            {},
            {},
            lambda value: value.square().sum(dim=(1, 2, 3)),
        )
    assert torch.isfinite(prediction).all()


def test_oriented_box_distance_signed_overlap_and_separation():
    from diffusion_planner.guidance import oriented_box_distance

    # Boxes are (x, y, cos(yaw), sin(yaw), length, width), all in meters.
    a = torch.tensor([[0.0, 0.0, 1.0, 0.0, 2.0, 2.0]]).expand(3, -1)
    b = torch.tensor(
        [
            [3.0, 0.0, 1.0, 0.0, 2.0, 2.0],
            [2.0, 0.0, 1.0, 0.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 0.0, 2.0, 2.0],
        ]
    )
    torch.testing.assert_close(oriented_box_distance(a, b), torch.tensor([1.0, 0.0, -1.0]))


def test_oriented_box_distance_is_rotation_invariant():
    from diffusion_planner.guidance import oriented_box_distance

    a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 4.0, 2.0]])
    b = torch.tensor([[0.0, 5.0, 0.0, 1.0, 4.0, 2.0]])
    torch.testing.assert_close(oriented_box_distance(a, b), torch.ones(1))


def test_oriented_box_overlap_gradient_is_finite():
    from diffusion_planner.guidance import oriented_box_distance

    a = torch.tensor([[0.0, 0.0, 1.0, 0.0, 2.0, 2.0]], requires_grad=True)
    b = torch.tensor([[1.0, 0.2, 1.0, 0.0, 2.0, 2.0]])
    oriented_box_distance(a, b).sum().backward()
    assert torch.isfinite(a.grad).all()


def test_released_collision_guidance_matches_upstream_cpu_fixture():
    from diffusion_planner.guidance import collision_guidance_reward

    trajectory = torch.zeros(1, 2, 5, 4)
    trajectory[:, :, :, 2] = 1
    trajectory[:, 0, :, 0] = torch.arange(5) * 0.5
    trajectory[:, 1, :, 0] = 3
    trajectory[:, 1, :, 1] = 1
    trajectory.requires_grad_(True)
    reward = collision_guidance_reward(
        trajectory,
        torch.tensor([[[4.5, 2.0]]]),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.tensor([0.05]),
    )
    gradient = torch.autograd.grad(reward.sum(), trajectory)[0]
    expected = torch.tensor([-124.881706, -142.357346, -159.659103, -176.449493, -192.421021])
    torch.testing.assert_close(gradient[0, 0, :, 1], expected, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(gradient[0, 1], torch.zeros_like(gradient[0, 1]))


def test_collision_guidance_masks_time_and_missing_neighbors():
    from diffusion_planner.guidance import collision_guidance_reward

    x = torch.zeros(2, 2, 5, 4)
    x[:, :, :, 2] = 1
    x[:, 1, :, 0] = 3
    x[:, 1, :, 1] = 1
    x.requires_grad_(True)
    reward = collision_guidance_reward(
        x,
        torch.tensor([[[4.5, 2.0]], [[4.5, 2.0]]]),
        torch.tensor([[False], [True]]),
        torch.tensor([0.2, 0.05]),
    )
    gradient = torch.autograd.grad(reward.sum(), x)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
