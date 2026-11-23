"""Compute diagnostic ego displacement on preprocessed data, not nuPlan scores."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.data import NuPlanDataset, normalize_observations
from diffusion_planner.metrics import displacement_errors
from diffusion_planner.sampling import sample_ddim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="trusted training checkpoint")
    parser.add_argument("--config", help="JSON PlannerConfig overrides used by checkpoint")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = PlannerConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    dataset = NuPlanDataset(
        args.data_dir,
        args.manifest,
        past_neighbors=config.agent_num,
        predicted_neighbors=config.predicted_neighbor_num,
    )
    model = DiffusionPlanner(config).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    torch.manual_seed(args.seed)
    ades, fdes = [], []
    for inputs, ego, _, _ in DataLoader(dataset, batch_size=1, shuffle=False):
        inputs = normalize_observations(inputs)
        with torch.no_grad():
            encoded, output = model(inputs)
            current = output["current_states"]
            noise = torch.randn(1, config.participant_num, config.future_len + 1, 4)
            normalized = sample_ddim(model, encoded, inputs, current, noise, steps=args.steps)
            trajectory = model.normalizer.inverse(normalized[:, :, 1:])[:, 0]
            valid = torch.ones(ego.shape[:-1], dtype=torch.bool)
            ade, fde = displacement_errors(trajectory, ego, valid)
            ades.append(ade.item())
            fdes.append(fde.item())
    print(
        json.dumps(
            {
                "protocol": "diagnostic_ego_open_loop",
                "samples": len(ades),
                "seed": args.seed,
                "steps": args.steps,
                "ego_ade_m": sum(ades) / len(ades),
                "ego_fde_m": sum(fdes) / len(fdes),
            }
        )
    )


if __name__ == "__main__":
    main()
