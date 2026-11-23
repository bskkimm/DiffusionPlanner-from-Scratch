import torch

from diffusion_planner.sampling import sample_ddim


def test_sample_preserves_current_state_and_is_seeded():
    class Oracle:
        def denoise(self, x, time, encoded, inputs):
            return torch.zeros_like(x)

    current = torch.tensor([[[1.0, 2.0, 1.0, 0.0]]])
    noise = torch.randn(1, 1, 3, 4)
    result = sample_ddim(Oracle(), {}, {}, current, noise, steps=4)
    torch.testing.assert_close(result[:, :, 0], current)
    torch.testing.assert_close(
        result[:, :, 1:], torch.zeros_like(result[:, :, 1:]), atol=1e-5, rtol=0
    )


def test_sampler_rejects_score_models():
    from types import SimpleNamespace

    import pytest

    model = SimpleNamespace(config=SimpleNamespace(diffusion_model_type="score"))
    with pytest.raises(ValueError, match="x_start"):
        sample_ddim(model, {}, {}, torch.zeros(1, 1, 4), torch.zeros(1, 1, 2, 4))


def test_dpm_solver_matches_official_fixed_operator_fixture():
    from diffusion_planner.sampling import sample_dpm_solver

    class Model:
        def denoise(self, x, time, encoded, inputs):
            return 0.25 * x + time[:, None, None]

    current = torch.zeros(1, 1, 4)
    noise = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, -0.4]]]])
    result = sample_dpm_solver(Model(), {}, {}, current, noise, steps=4)
    expected = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [0.0324631445, 0.0242978651, 0.0379066579, 0.0188543536]]]]
    )
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


def test_dpm_solver_official_fixtures_at_ten_and_twenty_steps():
    from diffusion_planner.sampling import sample_dpm_solver

    class Model:
        def denoise(self, x, time, encoded, inputs):
            return 0.25 * x + time[:, None, None]

    noise = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, -0.4]]]])
    fixtures = {
        10: [0.0085172402, 0.0060069547, 0.0101907598, 0.0043334304],
        20: [0.0108695729, 0.0080354623, 0.0127589786, 0.0061460570],
    }
    for steps, expected in fixtures.items():
        actual = sample_dpm_solver(Model(), {}, {}, torch.zeros(1, 1, 4), noise, steps=steps)
        torch.testing.assert_close(actual[0, 0, 1], torch.tensor(expected), rtol=1e-4, atol=1e-6)
