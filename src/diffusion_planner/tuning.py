"""Bounded CPU thread-count tuning against a correctness-checked FP32 baseline."""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.data import NuPlanDataset, normalize_observations
from diffusion_planner.training import train_step


def equivalent(reference: dict, candidate: dict, *, rtol=1e-4, atol=1e-6) -> bool:
    """Check outputs, losses, gradients and parameters after short learning."""
    for key in ("output", "losses"):
        if not torch.isfinite(candidate[key]).all() or not torch.allclose(
            reference[key], candidate[key], rtol=rtol, atol=atol
        ):
            return False
    for key in ("gradients", "parameters"):
        if reference[key].keys() != candidate[key].keys():
            return False
        for name, value in reference[key].items():
            other = candidate[key][name]
            if not torch.isfinite(other).all() or not torch.allclose(
                value, other, rtol=rtol, atol=atol
            ):
                return False
    return True


def candidate(data_dir, manifest, config, batch_size, threads, steps, result):
    """Run one isolated candidate using actual loss, gradients and updates."""
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    cfg = PlannerConfig(**config)
    dataset = NuPlanDataset(
        data_dir,
        manifest,
        past_neighbors=cfg.agent_num,
        predicted_neighbors=cfg.predicted_neighbor_num,
    )
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    inputs, ego, neighbors, mask = batch
    inputs = normalize_observations(inputs)
    model = DiffusionPlanner(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.eval()
    with torch.no_grad():
        encoded, _ = model(inputs)
        noisy = torch.randn(ego.size(0), cfg.participant_num, cfg.trajectory_dim)
        output = model.denoise(noisy, torch.full((ego.size(0),), 0.5), encoded, inputs)
    torch.manual_seed(314)
    losses = []
    # One warmup and at least two timed updates, all using the same data and recipe.
    train_step(model, optimizer, inputs, ego, neighbors, mask)
    start = time.perf_counter()
    for _ in range(steps):
        losses.append(train_step(model, optimizer, inputs, ego, neighbors, mask)["loss"].item())
    elapsed = time.perf_counter() - start
    torch.save(
        {
            "output": output.detach(),
            "losses": torch.tensor(losses),
            "gradients": {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            },
            "parameters": {
                name: value.detach().clone() for name, value in model.state_dict().items()
            },
            "seconds": elapsed,
            "samples": ego.size(0) * steps,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        result,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--child-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--child-result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.steps < 2 or args.max_seconds <= 0:
        parser.error("positive batch/time and at least two steps are required")
    cfg = asdict(
        PlannerConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    )
    if args.child_threads is not None:
        if args.child_result is None:
            parser.error("child result path is required")
        candidate(
            args.data_dir,
            args.manifest,
            cfg,
            args.batch_size,
            args.child_threads,
            args.steps,
            args.child_result,
        )
        return
    cpus = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    candidates = [1, 1] + ([2, 2] if cpus >= 2 else [])
    trials, reference, selected = [], None, 1
    deadline = time.monotonic() + args.max_seconds
    with tempfile.TemporaryDirectory(prefix="planner-tune-") as directory:
        for index, threads in enumerate(candidates):
            remaining = deadline - time.monotonic()
            entry = {"threads": threads}
            if remaining <= 0:
                entry["status"] = "time_limit"
                trials.append(entry)
                continue
            result = str(Path(directory) / f"{index}.pt")
            command = [
                sys.executable,
                "-m",
                "diffusion_planner.tuning",
                "--data-dir",
                args.data_dir,
                "--manifest",
                args.manifest,
                "--batch-size",
                str(args.batch_size),
                "--steps",
                str(args.steps),
                "--max-seconds",
                str(args.max_seconds),
                "--output",
                args.output,
                "--child-threads",
                str(threads),
                "--child-result",
                result,
            ]
            if args.config:
                command += ["--config", args.config]
            try:
                subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=remaining
                )
                data = torch.load(result, map_location="cpu", weights_only=True)
                entry.update(
                    status="measured",
                    seconds=data["seconds"],
                    samples=data["samples"],
                    peak_rss_kib=data["peak_rss_kib"],
                )
                if threads == 1 and reference is None:
                    reference = data
                    entry["correct"] = True
                elif reference is None:
                    entry["correct"] = False
                    entry["reason"] = "baseline was not measured"
                else:
                    entry["correct"] = equivalent(reference, data)

            except subprocess.TimeoutExpired:
                entry["status"] = "time_limit"
            except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
                entry.update(status="failed", error=str(exc)[-1000:])
            trials.append(entry)
    baseline = [item for item in trials if item["threads"] == 1 and item.get("correct")]
    alternatives = [item for item in trials if item["threads"] == 2 and item.get("correct")]
    if (
        len(baseline) == 2
        and len(alternatives) == 2
        and sum(item["seconds"] for item in alternatives)
        < 0.95 * sum(item["seconds"] for item in baseline)
    ):
        selected = 2
    report = {
        "selected_threads": selected,
        "selection": "best measured among tested candidates; "
        "baseline retained unless repeated trials are correct "
        "and mean time is at least 5% lower",
        "baseline_measured": reference is not None,
        "trials": trials,
        "device": "cpu",
        "precision": "float32",
        "model": cfg,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "cpu_affinity_count": cpus,
        "max_seconds": args.max_seconds,
        "limitations": "single cached batch, zero workers, no GPU or microbatch tuning",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
