# Diffusion Planner from Scratch

An independently implemented, architecture-faithful study of **Diffusion-Based Planning for Autonomous Driving with Flexible Guidance** (Zheng et al., ICLR 2025). The default configuration is the public official checkpoint configuration, while tests opt into reduced tensor sizes.

## What is implemented

- Agent-history, static-object, and lane encoders, including official type, speed-limit, traffic-light, and positional embeddings.
- Official MLP-Mixer and masked context-fusion transformer structure.
- Official route MLP-Mixer encoder and route-plus-timestep conditioning.
- Official DiT block order, adaLN-Zero modulation, agent identity embeddings, and final adaLN projection.
- Continuous VP-SDE, released `x_start` loss construction, state normalization, and current-state preservation.
- A DPM-Solver-compatible denoiser callback; the external solver is intentionally not vendored.
- `notebooks/diffusion_planner_walkthrough.ipynb`, a module-by-module architecture study with input/output traces.

This is architecture-faithful, not a complete nuPlan benchmark reproduction. It omits nuPlan's SDK-dependent raw-data preprocessing, released checkpoint loading, the upstream DPM-Solver dependency, distributed training, and closed-loop runners. The model defaults, component topology, tensor interfaces, loss construction, and initialization follow the official release.

## Install and test

```bash
python3 -m pip install -e ".[dev]"
pytest
```

The default released-checkpoint contract is:

```text
neighbor_agents_past: (batch, 32 agents, 21 history steps, 11 features)
static_objects:      (batch, 5 objects, 10 features)
lanes:               (batch, 70 lanes, 20 points, 12 features)
route_lanes:         (batch, 25 lanes, 20 points, 12 features)
ego_current_state:   (batch, 10 features)
future trajectory:   (batch, 11 participants, 80 steps, 4 features)
```

The 11 participants are ego followed by ten nearest neighbors. Each future state contains `(x, y, cos(yaw), sin(yaw))`, matching the paper's compact trajectory representation.

## Repository layout

```text
src/diffusion_planner/
  config.py                 # dimensions and diffusion hyperparameters
  diffusion.py              # VP schedule and forward noising
  guidance.py               # differentiable energy functions
  model/mixer.py            # token/channel MLP mixing
  model/encoder.py          # structured context encoders
  model/dit.py              # timestep-conditioned DiT denoiser
  model/planner.py          # loss, sampling, state constraint
notebooks/                  # step-by-step standalone explanation
tests/                      # shape, gradient, and notebook checks
```

## References

- [Paper](https://arxiv.org/abs/2501.15564)
- [Official implementation](https://github.com/ZhengYinan-AIR/Diffusion-Planner)
- [Project page](https://zhengyinan-air.github.io/Diffusion-Planner/)

```bibtex
@inproceedings{zheng2025diffusionbased,
  title={Diffusion-Based Planning for Autonomous Driving with Flexible Guidance},
  author={Yinan Zheng and Ruiming Liang and Kexin Zheng and Jinliang Zheng and Liyuan Mao and Jianxiong Li and Weihao Gu and Rui Ai and Shengbo Eben Li and Xianyuan Zhan and Jingjing Liu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
