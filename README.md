# Diffusion Planner from Scratch

An educational, independently implemented study of **Diffusion-Based Planning for Autonomous Driving with Flexible Guidance** (Zheng et al., ICLR 2025). The implementation follows the paper and the public official repository, but deliberately uses a small structured tensor interface so every module can be understood and tested on a laptop.

## What is implemented

- Agent-history and lane vector encoders with MLP-Mixer blocks.
- Masked transformer context fusion for agents, static objects, and lanes.
- Route-independent DiT-style trajectory denoiser with timestep conditioning and adaptive modulation.
- VP diffusion forward process and `x_start` denoising objective.
- Joint ego and neighbor trajectory generation with the current state clamped during sampling.
- Training-free differentiable collision and comfort guidance.
- `notebooks/diffusion_planner_walkthrough.ipynb`, which rebuilds the complete pipeline independently instead of importing this package.

This is not a nuPlan reproduction and does not claim the paper's benchmark results. It omits nuPlan preprocessing, DPM-Solver, normalization statistics, CARLA/nuPlan closed-loop runners, and production-scale training. Those boundaries keep the learning implementation runnable and make the tensor contracts explicit.

## Install and test

```bash
python3 -m pip install -e ".[dev]"
pytest
```

The default toy contract is:

```text
neighbor_agents_past: (batch, 4 neighbors, 8 history steps, 11 features)
static_objects:      (batch, 4 objects, 10 features)
lanes:               (batch, 8 lanes, 12 points, 12 features)
ego_current_state:   (batch, 4)             # x, y, cos(yaw), sin(yaw)
target:              (batch, 5 participants, 13 steps, 4 features)
```

The five participants are ego followed by four nearest neighbors. Each future state contains `(x, y, cos(yaw), sin(yaw))`, matching the paper's compact trajectory representation.

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
