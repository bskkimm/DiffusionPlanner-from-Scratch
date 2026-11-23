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
