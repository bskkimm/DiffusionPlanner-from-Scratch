"""Epoch-boundary checkpoint utilities for training experiments."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


class ExponentialMovingAverage:
    """Keep a detached moving average of all floating-point model state."""

    def __init__(self, model, decay: float = 0.999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA model state mismatch")
        for key, value in state.items():
            self.shadow[key].copy_(value)


def save_checkpoint(
    path, model, optimizer, *, epoch: int, identity: str, scheduler=None, ema=None
) -> None:
    """Atomically store CPU/GPU random state and optimizer state at an epoch boundary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "identity": identity,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "ema": ema.state_dict() if ema else None,
    }
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, optimizer, *, identity: str, scheduler=None, ema=None) -> int:
    """Restore a trusted checkpoint after verifying its recipe/data identity.

    PyTorch pickle checkpoints must only be loaded from trusted sources.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise ValueError("checkpoint identity does not match the current recipe/data")
    if (scheduler is None) != (state.get("scheduler") is None) or (ema is None) != (
        state.get("ema") is None
    ):
        raise ValueError("checkpoint scheduler/EMA configuration mismatch")
    if state["cuda_rng"] is not None and not torch.cuda.is_available():
        raise RuntimeError("checkpoint requires CUDA random state")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None:
        ema.load_state_dict(state["ema"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    if state["cuda_rng"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint requires CUDA random state")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["epoch"]


def train_step(
    model,
    optimizer,
    inputs,
    ego_future,
    neighbor_future,
    neighbor_mask,
    *,
    planning_weight: float = 1.0,
    max_grad_norm: float = 5.0,
):
    """Run one actual diffusion objective, backward pass and optimizer update.

    Observations must already be normalized and future targets encoded.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model.diffusion_loss(inputs, ego_future, neighbor_future, neighbor_mask)
    total = planning_weight * losses["ego_planning_loss"] + losses["neighbor_prediction_loss"]
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite training loss")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    optimizer.step()
    return {
        **{name: value.detach() for name, value in losses.items()},
        "loss": total.detach(),
        "gradient_norm": gradient_norm.detach(),
    }


def dataset_identity(data_dir, manifest, recipe: dict, *, files_out: set[str] | None = None) -> str:
    """Hash ordered split contents and the resolved recipe before training/resume."""
    import hashlib
    import json

    root = Path(data_dir).resolve()
    names = json.loads(Path(manifest).read_text())
    if not isinstance(names, list) or not names:
        raise ValueError("manifest must be a nonempty JSON list")
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())
    for name in names:
        if not isinstance(name, str) or not (root / name).resolve().is_relative_to(root):
            raise ValueError("manifest paths must be relative to data directory")
        digest.update(json.dumps(name).encode())
        sample_digest = hashlib.sha256() if files_out is not None else None
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                if sample_digest is not None:
                    sample_digest.update(chunk)
        if sample_digest is not None and files_out is not None:
            files_out.add(sample_digest.hexdigest())
    return digest.hexdigest()


def estimate_progress(*, start_epoch, completed_epoch, total_epochs, elapsed_seconds, now):
    """Estimate remaining work using current-session epoch time, including checkpoints."""
    from datetime import timedelta

    completed = completed_epoch - start_epoch
    if completed <= 0 or completed_epoch > total_epochs or elapsed_seconds < 0:
        raise ValueError("invalid progress")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    remaining = elapsed_seconds / completed * (total_epochs - completed_epoch)
    return {
        "remaining_seconds": remaining,
        "finish": now + timedelta(seconds=remaining),
        "provisional": completed < 2,
    }


def validate(model, loader, *, device: torch.device, seed: int, planning_weight: float = 1.0):
    """Compute deterministic diagnostic diffusion losses without advancing training RNG."""
    from diffusion_planner.data import normalize_observations

    was_training = model.training
    model.eval()
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    sums = {"ego_planning_loss": 0.0, "neighbor_prediction_loss": 0.0}
    count = 0
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            for inputs, ego, neighbors, mask in loader:
                inputs = normalize_observations(
                    {name: value.to(device) for name, value in inputs.items()}
                )
                losses = model.diffusion_loss(
                    inputs, ego.to(device), neighbors.to(device), mask.to(device)
                )
                for name in sums:
                    sums[name] += losses[name].item() * ego.size(0)
                count += ego.size(0)
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("validation loader is empty")
    result = {name: value / count for name, value in sums.items()}
    result["loss"] = (
        planning_weight * result["ego_planning_loss"] + result["neighbor_prediction_loss"]
    )
    return result
