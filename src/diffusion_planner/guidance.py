"""Differentiable, training-free trajectory energies."""

import torch


def comfort_energy(trajectory: torch.Tensor, acceleration_limit: float = 3.0) -> torch.Tensor:
    """Penalize squared second position differences; no timestep scaling is applied."""
    positions = trajectory[:, 0, :, :2]
    if positions.shape[1] < 3:
        return positions.sum(dim=(1, 2)) * 0
    acceleration = positions[:, 2:] - 2 * positions[:, 1:-1] + positions[:, :-2]
    return (acceleration.square().sum(dim=-1) - acceleration_limit**2).relu().mean(dim=-1)


def collision_energy(trajectory: torch.Tensor, radius: float = 2.0) -> torch.Tensor:
    """Penalize close neighbors; caller must exclude padded participants."""
    if trajectory.shape[1] < 2 or trajectory.shape[2] == 0:
        return trajectory.sum(dim=(1, 2, 3)) * 0
    ego = trajectory[:, :1, :, :2]
    neighbors = trajectory[:, 1:, :, :2]
    distances = (ego - neighbors).square().sum(dim=-1).clamp_min(1e-8).sqrt()
    return (radius - distances).relu().square().mean(dim=(1, 2))


def guided_x_start(
    model,
    noisy: torch.Tensor,
    time: torch.Tensor,
    context: dict[str, torch.Tensor],
    mask: dict[str, torch.Tensor],
    energy_fn,
    scale: float = 0.1,
) -> torch.Tensor:
    """Apply a heuristic energy-gradient update using the planner denoiser interface.

    This is not the released classifier-guidance wrapper. The caller controls
    energy units, padding and current-state preservation.
    """
    with torch.enable_grad():
        noisy = noisy.detach().requires_grad_(True)
        prediction = model.denoise(noisy, time, context, mask)
        energy = energy_fn(
            prediction.reshape(prediction.shape[0], prediction.shape[1], -1, 4)
        ).sum()
        gradient = torch.autograd.grad(energy, noisy)[0]
    return (prediction - scale * gradient).detach()


def oriented_box_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Signed separating-axis distance between oriented rectangles in meters.

    Inputs broadcast as (..., 6): center x/y, heading cosine/sine, length, width.
    Negative values indicate overlap; nonnegative values are the smallest
    separating-axis gap, which is not Euclidean distance for diagonal separation.
    """
    if first.shape[-1] != 6 or second.shape[-1] != 6:
        raise ValueError("boxes must have six features")
    first, second = torch.broadcast_tensors(first, second)
    direction_a, direction_b = first[..., 2:4], second[..., 2:4]
    norm_a = torch.linalg.vector_norm(direction_a, dim=-1, keepdim=True)
    norm_b = torch.linalg.vector_norm(direction_b, dim=-1, keepdim=True)
    if (norm_a <= 1e-8).any() or (norm_b <= 1e-8).any():
        raise ValueError("box headings must be nonzero")
    if (first[..., 4:] <= 0).any() or (second[..., 4:] <= 0).any():
        raise ValueError("box dimensions must be positive")
    forward_a, forward_b = direction_a / norm_a, direction_b / norm_b
    side_a = torch.stack((-forward_a[..., 1], forward_a[..., 0]), dim=-1)
    side_b = torch.stack((-forward_b[..., 1], forward_b[..., 0]), dim=-1)
    axes = torch.stack((forward_a, side_a, forward_b, side_b), dim=-2)
    delta = second[..., :2] - first[..., :2]
    center_gap = (axes * delta.unsqueeze(-2)).sum(-1).abs()
    radius_a = (axes * forward_a.unsqueeze(-2)).sum(-1).abs() * first[..., 4:5] / 2 + (
        axes * side_a.unsqueeze(-2)
    ).sum(-1).abs() * first[..., 5:6] / 2
    radius_b = (axes * forward_b.unsqueeze(-2)).sum(-1).abs() * second[..., 4:5] / 2 + (
        axes * side_b.unsqueeze(-2)
    ).sum(-1).abs() * second[..., 5:6] / 2
    separation = center_gap - radius_a - radius_b
    overlaps = (separation < 0).all(dim=-1)
    smallest_gap = separation.masked_fill(separation < 0, float("inf")).min(dim=-1).values
    return torch.where(overlaps, separation.max(dim=-1).values, smallest_gap)


def collision_guidance_reward(trajectory, neighbor_dimensions, neighbor_mask, time):
    """Released-style lateral collision score in physical ego-frame units.

    ``trajectory`` is (B, 1+N, 1+T, 4), with x/y in meters and heading
    cosine/sine. Neighbor dimensions are (B, N, 2), length then width;
    True in neighbor_mask means absent. The score is active for 0.005 < t < 0.1.
    The gradient is a smoothed lateral signal, not a calibrated collision rate.
    """
    import torch.nn.functional as functional

    if not trajectory.requires_grad:
        raise ValueError("collision guidance requires a differentiable trajectory")
    batch, participants, horizon, features = trajectory.shape
    if (
        features != 4
        or neighbor_dimensions.shape != (batch, participants - 1, 2)
        or neighbor_mask.shape != (batch, participants - 1)
        or time.shape != (batch,)
    ):
        raise ValueError("collision guidance inputs have incompatible shapes")
    active = ((time > 0.005) & (time < 0.1))[:, None, None, None]
    bridged = torch.where(active, trajectory, trajectory.detach())
    heading = bridged[..., 2:].detach()
    heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1e-8)
    # A zero heading has no orientation; use a fixed x-facing direction.
    fallback = torch.zeros_like(heading)
    fallback[..., 0] = 1
    heading = torch.where(heading.abs().sum(dim=-1, keepdim=True) > 0, heading, fallback)
    state = torch.cat((bridged[..., :2], heading), dim=-1)
    valid = ~neighbor_mask
    if not valid.any() or horizon < 2:
        return trajectory[:, 0].sum(dim=(1, 2)) * 0
    ego = state[:, :1, 1:].expand(-1, participants - 1, -1, -1).clone()
    ego[..., :2] = ego[..., :2] + ego[..., 2:] * 1.67
    other = state[:, 1:, 1:].detach()
    ego_size = state.new_tensor([5.176 + 1, 2.297 + 1]).expand(
        batch, participants - 1, horizon - 1, 2
    )
    other_size = (neighbor_dimensions + 1).unsqueeze(2).expand(-1, -1, horizon - 1, -1)
    ego_boxes = torch.cat((ego, ego_size), dim=-1)[valid]
    other_boxes = torch.cat((other, other_size), dim=-1)[valid]
    distances = oriented_box_distance(ego_boxes, other_boxes)
    proximity = (1 - distances).clamp_min(0)
    overlap = proximity > 1
    total = proximity[overlap].sum() / (overlap.sum() + 1e-5) + proximity[~overlap].sum() / (
        (proximity[~overlap] > 0).sum() + 1e-5
    )
    raw_reward = -total.exp()
    gradient = torch.autograd.grad(raw_reward, state, retain_graph=True)[0][:, 0, :, :2]
    c, s = state[:, 0, :, 2], state[:, 0, :, 3]
    lateral = -s * gradient[..., 0] + c * gradient[..., 1]
    kernel = torch.exp(
        -torch.linspace(-2, 2, 21, device=state.device, dtype=state.dtype).square() / 4
    )
    lateral = functional.conv1d(
        functional.pad(lateral[:, None], (10, 10), mode="replicate"), kernel.view(1, 1, -1)
    )[:, 0]
    world = torch.stack((-s * lateral, c * lateral), dim=-1)
    return 3 * (world.detach() * state[:, 0, :, :2]).sum(dim=(1, 2))


def make_collision_score(normalizer, neighbor_dimensions, neighbor_mask):
    """Build the released-style collision callback for normalized solver states.

    Dimensions must be physical lengths and widths, not normalized features.
    """

    def score(proxy, time):
        return collision_guidance_reward(
            normalizer.inverse(proxy), neighbor_dimensions, neighbor_mask, time
        )

    return score
