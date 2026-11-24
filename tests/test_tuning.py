import torch

from diffusion_planner.tuning import equivalent


def test_tuning_rejects_changed_outputs_gradients_or_learning():
    baseline = {
        "output": torch.tensor([1.0]),
        "losses": torch.tensor([2.0, 1.0]),
        "gradients": {"weight": torch.tensor([0.2])},
        "parameters": {"weight": torch.tensor([0.8])},
    }
    same = {
        key: value.clone()
        if isinstance(value, torch.Tensor)
        else {name: value.clone() for name, value in value.items()}
        for key, value in baseline.items()
    }
    assert equivalent(baseline, same)
    same["gradients"]["weight"] += 1
    assert not equivalent(baseline, same)
