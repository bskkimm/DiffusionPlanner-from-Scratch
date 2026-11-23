"""Utilities for the official dataset's future-state representation."""

import torch


def encode_future(future: torch.Tensor, *, mask_padding: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode (..., x, y, yaw radians) as (..., x, y, cos(yaw), sin(yaw)).

    For neighbor labels, upstream treats a zero (x, y, yaw) vector as padding.
    Determine padding before encoding: cos(0) would otherwise turn it nonzero.
    Ego labels are never interpreted as padding. Extra features are ignored.
    """
    if future.ndim < 1 or future.shape[-1] < 3:
        raise ValueError("future must contain at least x, y, and yaw")
    state = future[..., :3]
    missing = (
        ~state.ne(0).any(dim=-1)
        if mask_padding
        else torch.zeros_like(state[..., 0], dtype=torch.bool)
    )
    encoded = torch.cat((state[..., :2], state[..., 2:3].cos(), state[..., 2:3].sin()), dim=-1)
    return encoded.masked_fill(missing.unsqueeze(-1), 0), missing


class NuPlanDataset(torch.utils.data.Dataset):
    """Read upstream-generated NPZ samples listed by a JSON split manifest.

    No raw nuPlan preprocessing is performed. Paths must stay under data_dir.
    Input observations are raw; normalize them before calling the model.
    """

    def __init__(self, data_dir, manifest, *, past_neighbors: int, predicted_neighbors: int):
        import json
        from pathlib import Path

        self.data_dir = Path(data_dir).resolve()
        self.files = json.loads(Path(manifest).read_text())
        if not isinstance(self.files, list) or not self.files:
            raise ValueError("manifest must be a nonempty JSON list")
        self.past_neighbors = past_neighbors
        self.predicted_neighbors = predicted_neighbors
        for filename in self.files:
            if not isinstance(filename, str) or not (
                self.data_dir / filename
            ).resolve().is_relative_to(self.data_dir):
                raise ValueError("manifest paths must be relative to the data directory")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import numpy as np

        with np.load(self.data_dir / self.files[index], allow_pickle=False) as sample:
            names = (
                "ego_current_state",
                "neighbor_agents_past",
                "static_objects",
                "lanes",
                "lanes_speed_limit",
                "lanes_has_speed_limit",
                "route_lanes",
            )
            inputs = {name: torch.from_numpy(sample[name].copy()) for name in names}
            inputs["neighbor_agents_past"] = inputs["neighbor_agents_past"][: self.past_neighbors]
            ego, _ = encode_future(
                torch.from_numpy(sample["ego_agent_future"].copy()), mask_padding=False
            )
            neighbors, mask = encode_future(
                torch.from_numpy(
                    sample["neighbor_agents_future"][: self.predicted_neighbors].copy()
                ),
                mask_padding=True,
            )
        return inputs, ego, neighbors, mask


def normalize_observations(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize observations using the cached upstream normalization.json values."""
    from diffusion_planner.normalizer import ObservationNormalizer

    stats = {
        "ego_current_state": (
            [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [20, 20, 1, 1, 20, 20, 20, 20, 1, 1],
        ),
        "neighbor_agents_past": (
            [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [20, 20, 1, 1, 20, 20, 20, 20, 1, 1, 1],
        ),
        "static_objects": ([10, 0, 0, 0, 0, 0, 0, 0, 0, 0], [20, 20, 1, 1, 20, 20, 1, 1, 1, 1]),
        "lanes": (
            [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1],
        ),
        "route_lanes": (
            [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1],
        ),
        "lanes_speed_limit": ([0], [20]),
        "route_lanes_speed_limit": ([0], [20]),
    }
    return ObservationNormalizer(stats)(inputs)
