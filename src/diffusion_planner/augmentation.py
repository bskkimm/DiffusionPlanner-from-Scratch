"""Geometry primitives for trajectory perturbation; no SDK dependency."""

import torch


def quintic_positions(
    start,
    start_velocity,
    start_acceleration,
    end,
    end_velocity,
    end_acceleration,
    horizon: float,
    times: torch.Tensor,
):
    """Interpolate positions with specified endpoint position, velocity, acceleration.

    Endpoint tensors are (B, 2), times are (S,) in seconds, and output is (B,S,2).
    """
    if (
        horizon <= 0
        or times.ndim != 1
        or any(
            x.shape != start.shape
            for x in (start_velocity, start_acceleration, end, end_velocity, end_acceleration)
        )
    ):
        raise ValueError("invalid interpolation inputs")
    if start.ndim != 2 or start.shape[-1] != 2:
        raise ValueError("endpoint tensors must have shape (batch, 2)")
    T = horizon
    matrix = start.new_tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
            [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
        ]
    )
    bounds = torch.stack(
        (start, start_velocity, start_acceleration, end, end_velocity, end_acceleration), dim=1
    )
    coefficients = torch.linalg.solve(matrix, bounds)
    powers = times.to(start).unsqueeze(-1).pow(torch.arange(6, device=start.device))
    return torch.einsum("sk,bkd->bsd", powers, coefficients)


def to_ego_frame(inputs, ego_future, neighbor_future):
    """Transform raw positions/vectors and yaw to the current ego frame.

    Preserves upstream zero padding for neighbors, map vectors and static objects.
    Inputs and futures are copied; headings in futures are radians.
    """
    result = {name: value.clone() for name, value in inputs.items()}
    ego = ego_future.clone()
    neighbors = neighbor_future.clone()
    state = inputs["ego_current_state"]
    origin = state[:, :2]
    heading = state[:, 2:4]

    def rotate(vector, translate=False):
        extra = vector.ndim - 2
        c = heading[:, 0].reshape(-1, *([1] * extra))
        s = heading[:, 1].reshape(-1, *([1] * extra))
        if translate:
            vector = vector - origin.reshape(origin.shape[0], *([1] * extra), 2)
        x, y = vector.unbind(-1)
        return torch.stack((c * x + s * y, -s * x + c * y), dim=-1)

    def transform_features(name, vector_pairs, mask_width):
        value = result[name]
        absent = ~inputs[name][..., :mask_width].ne(0).any(dim=-1)
        value[..., :2] = rotate(value[..., :2], translate=True)
        for start in vector_pairs:
            value[..., start : start + 2] = rotate(value[..., start : start + 2])
        value[absent] = 0

    result["ego_current_state"][..., :2] = rotate(state[..., :2], translate=True)
    for start in (2, 4, 6):
        result["ego_current_state"][..., start : start + 2] = rotate(state[..., start : start + 2])
    transform_features("neighbor_agents_past", (2, 4), 6)
    transform_features("static_objects", (2,), 10)
    transform_features("lanes", (2, 4, 6), 8)
    transform_features("route_lanes", (2, 4, 6), 8)
    yaw = torch.atan2(heading[:, 1], heading[:, 0])
    ego[..., :2] = rotate(ego[..., :2], translate=True)
    ego[..., 2] = torch.atan2(
        torch.sin(ego[..., 2] - yaw[:, None]), torch.cos(ego[..., 2] - yaw[:, None])
    )
    missing = ~neighbors[..., :2].ne(0).any(dim=-1)
    neighbors[..., :2] = rotate(neighbors[..., :2], translate=True)
    relative = neighbors[..., 2] - yaw[:, None, None]
    neighbors[..., 2] = torch.atan2(relative.sin(), relative.cos())
    neighbors[missing] = 0
    return result, ego, neighbors


def perturb_and_reframe(
    inputs, ego_future, neighbor_future, *, probability: float = 0.5, wheelbase: float = 3.089
):
    """Perturb ego state, reconnect its future with a quintic, and reframe.

    Uses the released default perturbation ranges, 0.1 s steps and a 2.1 s
    interpolation endpoint. Input futures are raw (x, y, yaw), not encoded.
    """
    if not 0 <= probability <= 1 or wheelbase <= 0:
        raise ValueError("invalid augmentation probability or wheelbase")
    state = inputs["ego_current_state"].clone()
    batch = state.shape[0]
    selected = (torch.rand(batch, device=state.device) >= 1 - probability) & (
        state[:, 4].abs() >= 2
    )
    augmented = state.clone()
    future = ego_future.clone()
    limits = state.new_tensor([0.0, 0.75, 0.35, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0])
    perturbation = (2 * torch.rand(batch, 9, device=state.device, dtype=state.dtype) - 1) * limits
    if selected.any():
        if future.shape[1] < 21:
            raise ValueError("augmentation requires at least 21 future steps at 10 Hz")
        values = state.new_zeros(batch, 9)
        values[:, 3:] = state[:, 4:10]
        values += perturbation
        values[:, 3] = values[:, 3].clamp_min(0)
        values[:, 8] = values[:, 8].clamp(-0.85, 0.85)
        augmented[:, :2] = values[:, :2]
        augmented[:, 2] = values[:, 2].cos()
        augmented[:, 3] = values[:, 2].sin()
        augmented[:, 4:8] = values[:, 3:7]
        speed = augmented[:, 4]
        yaw_rate = values[:, 8]
        fast = speed.abs() >= 0.2
        augmented[:, 8] = torch.where(
            fast, torch.atan(yaw_rate * wheelbase / speed.abs().clamp_min(0.2)), 0
        ).clamp(-2 * torch.pi / 3, 2 * torch.pi / 3)
        augmented[:, 9] = torch.where(fast, yaw_rate, 0)
        p0 = augmented[:, :2]
        p1 = future[:, 20, :2]
        theta0 = torch.atan2(future[:, 10, 1] - p0[:, 1], future[:, 10, 0] - p0[:, 0])
        theta1 = future[:, 20, 2]
        v0 = torch.linalg.vector_norm(augmented[:, 4:6], dim=-1)
        v1 = torch.linalg.vector_norm(future[:, 20, :2] - future[:, 19, :2], dim=-1) / 0.1
        a0 = torch.linalg.vector_norm(augmented[:, 6:8], dim=-1)
        a1 = (
            torch.linalg.vector_norm(
                future[:, 20, :2] - 2 * future[:, 19, :2] + future[:, 18, :2], dim=-1
            )
            / 0.01
        )
        delta = future[:, 20, 2] - future[:, 19, 2]
        w0, w1 = augmented[:, 9], torch.atan2(delta.sin(), delta.cos()) / 0.1

        def motion(theta, speed, acceleration, yaw_rate):
            direction = torch.stack((theta.cos(), theta.sin()), dim=-1)
            perpendicular = torch.stack((-theta.sin(), theta.cos()), dim=-1)
            return speed[:, None] * direction, acceleration[:, None] * direction + (
                speed * yaw_rate
            )[:, None] * perpendicular

        vel0, acc0 = motion(theta0, v0, a0, w0)
        vel1, acc1 = motion(theta1, v1, a1, w1)
        times = torch.linspace(0.1, 2.0, 20, device=state.device, dtype=state.dtype)
        position = quintic_positions(p0, vel0, acc0, p1, vel1, acc1, 2.1, times)
        displacement = torch.diff(torch.cat((p0[:, None], position), dim=1), dim=1)
        angles = torch.atan2(displacement[..., 1], displacement[..., 0])
        refined = torch.cat((position, angles.unsqueeze(-1)), dim=-1)
        future[selected, :20, :3] = refined[selected]
        state[selected] = augmented[selected]
    copied = dict(inputs)
    copied["ego_current_state"] = state
    return to_ego_frame(copied, future, neighbor_future)
