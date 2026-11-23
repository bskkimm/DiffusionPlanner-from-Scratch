import pytest
import torch

from diffusion_planner.metrics import displacement_errors


def test_displacement_errors_use_only_valid_steps_and_last_valid_step():
    prediction = torch.tensor(
        [[[0.0, 0.0], [3.0, 4.0], [9.0, 9.0]], [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    ade, fde = displacement_errors(prediction, target, valid)
    assert ade.item() == pytest.approx(7 / 3)
    assert fde.item() == pytest.approx(3.5)


def test_displacement_errors_reject_empty_targets():
    with pytest.raises(ValueError, match="valid"):
        displacement_errors(
            torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)
        )
