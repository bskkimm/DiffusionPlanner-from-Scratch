"""Epoch-boundary checkpoint utilities for training experiments."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, *, epoch: int, identity: str) -> None:
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


def load_checkpoint(path, model, optimizer, *, identity: str) -> int:
    """Restore a trusted checkpoint after verifying its recipe/data identity.

    PyTorch pickle checkpoints must only be loaded from trusted sources.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise ValueError("checkpoint identity does not match the current recipe/data")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {**{name: value.detach() for name, value in losses.items()}, "loss": total.detach()}


def dataset_identity(data_dir, manifest, recipe: dict) -> str:
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
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
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
