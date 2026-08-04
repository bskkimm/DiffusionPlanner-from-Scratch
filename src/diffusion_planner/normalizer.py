"""State and observation normalization used by the released checkpoint."""

from copy import copy

import torch


class StateNormalizer:
    def __init__(self, participant_num: int) -> None:
        mean = torch.tensor([10.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 4)
        std = torch.tensor([20.0, 20.0, 1.0, 1.0]).view(1, 1, 1, 4)
        self.mean = mean.expand(1, participant_num, 1, 4).clone()
        self.std = std.expand(1, participant_num, 1, 4).clone()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std.to(data.device) + self.mean.to(data.device)


class ObservationNormalizer:
    """Apply official means/stds while preserving all-zero padded vectors."""

    def __init__(self, statistics: dict[str, tuple[list[float], list[float]]]) -> None:
        self.statistics = statistics

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = copy(data)
        for name, (mean, std) in self.statistics.items():
            if name not in data:
                continue
            value = data[name]
            mask = value.ne(0).any(dim=-1, keepdim=True)
            output[name] = (value - torch.tensor(mean, device=value.device)) / torch.tensor(std, device=value.device)
            output[name] = output[name] * mask
        return output
