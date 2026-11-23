"""Measure the practical CPU data and training paths; no automatic tuning."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.data import NuPlanDataset, normalize_observations
from diffusion_planner.training import train_step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("invalid batch size, warmup, or steps")
    start = time.perf_counter()
    config = PlannerConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    dataset = NuPlanDataset(
        args.data_dir,
        args.manifest,
        past_neighbors=config.agent_num,
        predicted_neighbors=config.predicted_neighbor_num,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = DiffusionPlanner(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    startup = time.perf_counter() - start

    def batches():
        while True:
            yield from loader

    def update(batch):
        inputs, ego, neighbors, mask = batch
        train_step(model, optimizer, normalize_observations(inputs), ego, neighbors, mask)

    # Fresh iterator avoids itertools.cycle caching batches and hiding disk reads.
    stream = batches()
    cached = next(stream)
    for _ in range(args.warmup):
        update(cached)
    results = {}
    for mode in ("loader", "cached_step", "end_to_end"):
        elapsed, wait, samples = 0.0, 0.0, 0
        for _ in range(args.steps):
            before = time.perf_counter()
            batch = cached if mode == "cached_step" else next(stream)
            after_load = time.perf_counter()
            if mode != "loader":
                update(batch)
            end = time.perf_counter()
            wait += after_load - before
            elapsed += end - before
            samples += batch[1].shape[0]
        results[mode] = {
            "seconds": elapsed,
            "loader_wait_seconds": wait,
            "samples": samples,
            "samples_per_second": samples / elapsed,
        }
    print(
        json.dumps(
            {
                "device": "cpu",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "model": asdict(config),
                "batch_size": args.batch_size,
                "workers": 0,
                "warmup_steps": args.warmup,
                "timed_steps": args.steps,
                "data_dir": str(Path(args.data_dir).resolve()),
                "cache": "OS cache not controlled; in-memory batch for cached_step",
                "startup_seconds": startup,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
