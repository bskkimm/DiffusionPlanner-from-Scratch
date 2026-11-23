"""Compute diagnostic ego displacement on preprocessed data, not nuPlan scores."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.data import NuPlanDataset, normalize_observations
from diffusion_planner.guidance import make_collision_score
from diffusion_planner.metrics import displacement_errors
from diffusion_planner.sampling import sample_dpm_solver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="trusted training checkpoint")
    parser.add_argument("--config", help="JSON PlannerConfig overrides used by checkpoint")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ema", action="store_true", help="evaluate EMA weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--collision-guidance", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.temperature < 0 or args.guidance_scale < 0:
        parser.error("temperature and guidance scale must be nonnegative")
    config = PlannerConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    dataset = NuPlanDataset(
        args.data_dir,
        args.manifest,
        past_neighbors=config.agent_num,
        predicted_neighbors=config.predicted_neighbor_num,
    )
    model = DiffusionPlanner(config).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.ema and state.get("ema") is None:
        raise ValueError("checkpoint does not contain EMA weights")
    model.load_state_dict(state["ema"] if args.ema else state["model"])
    torch.manual_seed(args.seed)
    ades, fdes = [], []
    for inputs, ego, _, _ in DataLoader(dataset, batch_size=1, shuffle=False):
        dimensions = inputs["neighbor_agents_past"][:, : config.predicted_neighbor_num, -1, :][
            ..., [7, 6]
        ]
        inputs = normalize_observations(inputs)
        with torch.no_grad():
            encoded, output = model(inputs)
            current = output["current_states"]
            noise = (
                torch.randn(1, config.participant_num, config.future_len + 1, 4) * args.temperature
            )
            score = (
                make_collision_score(model.normalizer, dimensions, output["neighbor_current_mask"])
                if args.collision_guidance
                else None
            )
            normalized = sample_dpm_solver(
                model,
                encoded,
                inputs,
                current,
                noise,
                steps=args.steps,
                score_fn=score,
                score_scale=args.guidance_scale if score else 0.0,
            )
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
                "temperature": args.temperature,
                "collision_guidance": args.collision_guidance,
                "guidance_scale": args.guidance_scale if args.collision_guidance else 0.0,
                "weights": "ema" if args.ema else "model",
                "steps": args.steps,
                "ego_ade_m": sum(ades) / len(ades),
                "ego_fde_m": sum(fdes) / len(fdes),
            }
        )
    )


if __name__ == "__main__":
    main()
