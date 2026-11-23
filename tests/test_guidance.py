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
