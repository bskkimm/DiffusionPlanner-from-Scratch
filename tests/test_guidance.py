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
