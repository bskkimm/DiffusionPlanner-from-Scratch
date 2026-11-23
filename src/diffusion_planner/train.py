"""Practical single-process NPZ training (not the official distributed recipe)."""

import argparse
import json
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.data import NuPlanDataset, normalize_observations
from diffusion_planner.tracking import Tracker
from diffusion_planner.training import (
    dataset_identity,
    estimate_progress,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="JSON PlannerConfig overrides")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--planning-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mlflow-uri", help="optional MLflow tracking URI")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    config = PlannerConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    recipe = {
        "model": asdict(config),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "planning_weight": args.planning_weight,
        "seed": args.seed,
        "device": args.device,
        "workers": 0,
        "optimizer": "AdamW",
    }
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
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = DiffusionPlanner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = load_checkpoint(checkpoint, model, optimizer, identity=identity) if args.resume else 0
    if args.epochs <= start:
        raise ValueError("epochs must exceed the checkpoint epoch")
    output.mkdir(parents=True, exist_ok=True)
    session = {
        "identity": identity,
        "recipe": recipe,
        "torch": torch.__version__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resume_epoch": start,
        "requested_epochs": args.epochs,
    }
    (output / f"session-{uuid4().hex}.json").write_text(json.dumps(session, indent=2) + "\n")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    started = time.monotonic()
    tracker = Tracker(args.mlflow_uri)
    tracker.start(
        {
            "identity": identity,
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "resume_epoch": start,
        }
    )
    status = "FAILED"
    try:
        for epoch in range(start, args.epochs):
            total = 0.0
            count = 0
            for inputs, ego, neighbors, mask in loader:
                inputs = normalize_observations(
                    {key: value.to(device) for key, value in inputs.items()}
                )
                losses = train_step(
                    model,
                    optimizer,
                    inputs,
                    ego.to(device),
                    neighbors.to(device),
                    mask.to(device),
                    planning_weight=args.planning_weight,
                )
                total += losses["loss"].item() * ego.size(0)
                count += ego.size(0)
            save_checkpoint(checkpoint, model, optimizer, epoch=epoch + 1, identity=identity)
            elapsed = time.monotonic() - started
            progress = estimate_progress(
                start_epoch=start,
                completed_epoch=epoch + 1,
                total_epochs=args.epochs,
                elapsed_seconds=elapsed,
                now=datetime.now(timezone.utc),
            )
            tracker.log(
                {
                    "loss": total / count,
                    "active_seconds": elapsed,
                    "remaining_seconds": progress["remaining_seconds"],
                },
                step=epoch + 1,
                tags={
                    "finish_utc": progress["finish"].isoformat(),
                    "estimate_provisional": str(progress["provisional"]),
                },
            )
            print(
                f"epoch {epoch + 1}/{args.epochs} loss={total / count:.6f} "
                f"active_seconds={elapsed:.1f} "
                f"remaining_seconds={progress['remaining_seconds']:.1f} "
                f"finish={progress['finish'].isoformat()} provisional={progress['provisional']}",
                flush=True,
            )
        status = "FINISHED"
    finally:
        tracker.close(status=status)


if __name__ == "__main__":
    main()
