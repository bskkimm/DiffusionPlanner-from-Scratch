"""Create a tiny offline NPZ dataset for smoke tests and diagnostics only."""

import argparse
import json
from pathlib import Path

import numpy as np

from diffusion_planner import PlannerConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/smoke-model.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    cfg = PlannerConfig(**json.loads(Path(args.config).read_text()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for split in ("train", "validation"):
        for index in range(2):
            ego = np.zeros((cfg.future_len, 3), dtype=np.float32)
            ego[:, 0] = np.arange(1, cfg.future_len + 1) * 0.5
            ego[:, 1] = rng.normal(0, 0.05, cfg.future_len)
            past = rng.normal(size=(cfg.agent_num, cfg.time_len, 11)).astype(np.float32)
            past[..., 2] = 1
            past[..., 3] = 0
            past[..., 6:8] = np.abs(past[..., 6:8]) + 1
            future = rng.normal(size=(cfg.predicted_neighbor_num, cfg.future_len, 3)).astype(
                np.float32
            )
            lanes = rng.normal(size=(cfg.lane_num, cfg.lane_len, 12)).astype(np.float32)
            routes = rng.normal(size=(cfg.route_num, cfg.lane_len, 12)).astype(np.float32)
            static = rng.normal(size=(cfg.static_objects_num, 10)).astype(np.float32)
            filename = f"{split}-{index}.npz"
            np.savez(
                output / filename,
                ego_current_state=np.array([0, 0, 1, 0, 5, 0, 0, 0, 0, 0], dtype=np.float32),
                ego_agent_future=ego,
                neighbor_agents_past=past,
                neighbor_agents_future=future,
                static_objects=static,
                lanes=lanes,
                lanes_speed_limit=np.ones((cfg.lane_num, 1), dtype=np.float32) * 10,
                lanes_has_speed_limit=np.ones((cfg.lane_num, 1), dtype=np.bool_),
                route_lanes=routes,
            )
        (output / f"{split}.json").write_text(
            json.dumps([f"{split}-{i}.npz" for i in range(2)]) + "\n"
        )
    print(f"Synthetic data written to {output}; never use it as real driving evidence.")


if __name__ == "__main__":
    main()
