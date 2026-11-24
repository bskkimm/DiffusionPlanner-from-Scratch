"""Practical single-process NPZ training (not the official distributed recipe)."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.augmentation import perturb_and_reframe
from diffusion_planner.data import NuPlanDataset, encode_future, normalize_observations
from diffusion_planner.tracking import Tracker
from diffusion_planner.training import (
    ExponentialMovingAverage,
    dataset_identity,
    estimate_progress,
    load_checkpoint,
    save_checkpoint,
    train_step,
    validate,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", help="optional disjoint validation split")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="JSON PlannerConfig overrides")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--planning-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--augment", action="store_true", help="apply ego perturbation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mlflow-uri", help="optional MLflow tracking URI")
    args = parser.parse_args(argv)
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.warmup_epochs < 2
        or not 0 <= args.ema_decay < 1
    ):
        parser.error("epochs, batch size, and learning rate must be positive")
    config = PlannerConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    if args.augment and config.future_len < 21:
        parser.error("augmentation requires at least 21 future steps at 10 Hz")
    source_hash = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for source in sorted(package_dir.rglob("*.py")):
        source_hash.update(str(source.relative_to(package_dir)).encode())
        source_hash.update(source.read_bytes())
    recipe = {
        "model": asdict(config),
        "source_sha256": source_hash.hexdigest(),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "planning_weight": args.planning_weight,
        "seed": args.seed,
        "device": args.device,
        "workers": 0,
        "optimizer": "AdamW",
        "warmup_epochs": args.warmup_epochs,
        "ema_decay": args.ema_decay,
        "augment": args.augment,
    }
    train_hashes: set[str] = set()
    if args.val_manifest:
        val_hashes: set[str] = set()
        recipe["validation_identity"] = dataset_identity(
            args.data_dir, args.val_manifest, {"split": "validation"}, files_out=val_hashes
        )
        identity = dataset_identity(args.data_dir, args.manifest, recipe, files_out=train_hashes)
        if train_hashes & val_hashes:
            raise ValueError("training and validation manifests contain identical files")
    else:
        identity = dataset_identity(args.data_dir, args.manifest, recipe)
    output = Path(args.output)
    checkpoint = output / "latest.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError("checkpoint exists; use --resume or a new output directory")
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError("cannot resume: latest.pt is missing")
    dataset = NuPlanDataset(
        args.data_dir,
        args.manifest,
        past_neighbors=config.agent_num,
        predicted_neighbors=config.predicted_neighbor_num,
        raw_futures=args.augment,
    )
    validation_loader = None
    if args.val_manifest:
        validation_dataset = NuPlanDataset(
            args.data_dir,
            args.val_manifest,
            past_neighbors=config.agent_num,
            predicted_neighbors=config.predicted_neighbor_num,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = DiffusionPlanner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs - 1
    )
    fixed = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 1.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, fixed], milestones=[args.warmup_epochs]
    )
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    start = (
        load_checkpoint(
            checkpoint, model, optimizer, identity=identity, scheduler=scheduler, ema=ema
        )
        if args.resume
        else 0
    )
    if args.epochs <= start:
        raise ValueError("epochs must exceed the checkpoint epoch")
    output.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    session = {
        "identity": identity,
        "recipe": recipe,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "code_revision": revision,
        "code_dirty": dirty,
        "initialization": "random" if not args.resume else "checkpoint",
        "precision": "float32",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resume_epoch": start,
        "requested_epochs": args.epochs,
    }
    (output / f"session-{uuid4().hex}.json").write_text(json.dumps(session, indent=2) + "\n")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    started = time.monotonic()
    wall_started = datetime.now(timezone.utc)
    tracker = Tracker(args.mlflow_uri)
    tracker.start(
        {
            "identity": identity,
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "resume_epoch": start,
            "code_revision": revision,
            "code_dirty": str(dirty),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "precision": "float32",
        }
    )
    status = "FAILED"
    try:
        for epoch in range(start, args.epochs):
            totals = {
                "loss": 0.0,
                "ego_planning_loss": 0.0,
                "neighbor_prediction_loss": 0.0,
                "gradient_norm": 0.0,
            }
            count = 0
            for inputs, ego, neighbors, mask in loader:
                inputs = {key: value.to(device) for key, value in inputs.items()}
                ego, neighbors, mask = ego.to(device), neighbors.to(device), mask.to(device)
                if args.augment:
                    inputs, ego, neighbors = perturb_and_reframe(inputs, ego, neighbors)
                    ego, _ = encode_future(ego, mask_padding=False)
                    neighbors, mask = encode_future(neighbors, mask_padding=True)
                inputs = normalize_observations(inputs)
                losses = train_step(
                    model,
                    optimizer,
                    inputs,
                    ego,
                    neighbors,
                    mask,
                    planning_weight=args.planning_weight,
                )
                ema.update(model)
                for name in totals:
                    totals[name] += losses[name].item() * ego.size(0)
                count += ego.size(0)
            validation = (
                validate(
                    model,
                    validation_loader,
                    device=device,
                    seed=args.seed + epoch,
                    planning_weight=args.planning_weight,
                )
                if validation_loader is not None
                else None
            )
            scheduler.step()
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                epoch=epoch + 1,
                identity=identity,
                scheduler=scheduler,
                ema=ema,
            )
            elapsed = time.monotonic() - started
            wall_elapsed = (datetime.now(timezone.utc) - wall_started).total_seconds()
            progress = estimate_progress(
                start_epoch=start,
                completed_epoch=epoch + 1,
                total_epochs=args.epochs,
                elapsed_seconds=elapsed,
                now=datetime.now(timezone.utc),
            )
            tracker.log(
                {
                    **{name: value / count for name, value in totals.items()},
                    **(
                        {f"validation_{name}": value for name, value in validation.items()}
                        if validation
                        else {}
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "active_seconds": elapsed,
                    "session_wall_seconds": wall_elapsed,
                    "remaining_seconds": progress["remaining_seconds"],
                },
                step=epoch + 1,
                tags={
                    "finish_utc": progress["finish"].isoformat(),
                    "estimate_provisional": str(progress["provisional"]),
                },
            )
            if validation is not None:
                print(f"validation_loss={validation['loss']:.6f}", flush=True)
            print(
                f"epoch {epoch + 1}/{args.epochs} loss={totals['loss'] / count:.6f} "
                f"active_seconds={elapsed:.1f} wall_seconds={wall_elapsed:.1f} "
                f"remaining_seconds={progress['remaining_seconds']:.1f} "
                f"finish={progress['finish'].isoformat()} provisional={progress['provisional']}",
                flush=True,
            )
        status = "FINISHED"
    finally:
        tracker.close(status=status)


if __name__ == "__main__":
    main()
