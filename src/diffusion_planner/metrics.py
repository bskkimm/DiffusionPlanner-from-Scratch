"""Open-loop displacement diagnostics; these are not nuPlan closed-loop metrics."""

import torch


def displacement_errors(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    """Return valid-point ADE and mean last-valid-point FDE, in input position units.

    Inputs are (..., time, features), with x/y in the first two features, and
    valid is (..., time). Sequences with no valid points are excluded from FDE.
    """
    if prediction.shape != target.shape or prediction.ndim < 2 or prediction.shape[-1] < 2:
        raise ValueError("prediction and target must have matching (..., time, features) shapes")
    if valid.shape != prediction.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask matching the sequence dimensions")
    if not valid.any():
        raise ValueError("at least one valid point is required")
    distance = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
    ade = distance[valid].mean()
    time_index = torch.arange(valid.shape[-1], device=valid.device)
    last = torch.where(valid, time_index, -1).max(dim=-1).values
    eligible = last >= 0
    fde = distance.gather(-1, last.clamp_min(0).unsqueeze(-1)).squeeze(-1)[eligible].mean()
    return ade, fde
