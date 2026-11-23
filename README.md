# Diffusion Planner from Scratch

An independent PyTorch study of **Diffusion-Based Planning for Autonomous Driving with Flexible Guidance** (Zheng et al., ICLR 2025). The model uses dimensions documented for the public checkpoint, while tests use reduced tensor sizes. Numerical parity with the official model and paper results has not been established.

## Quick start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
jupyter notebook notebooks/diffusion_planner_walkthrough.ipynb
```

The notebook is the recommended entry point. It derives the architecture in execution order, shows the global tensor flow, and ends with an official-dimension synthetic shape trace.

## What is implemented

- Agent-history, static-object, and lane encoders, including official type, speed-limit, traffic-light, and positional embeddings.
- Official MLP-Mixer and masked context-fusion transformer structure.
- Official route MLP-Mixer encoder and route-plus-timestep conditioning.
- Official DiT block order, adaLN-Zero modulation, agent identity embeddings, and final adaLN projection.
- Continuous VP-SDE, released `x_start` loss construction, state normalization, and current-state preservation.
- An unguided second-order DPM-Solver++ sampler and denoiser callback; numerical verification is limited to a fixed operator fixture.
- `notebooks/diffusion_planner_walkthrough.ipynb`, a module-by-module architecture study with input/output traces.

> **Scope.** This is an incomplete implementation, not a complete nuPlan benchmark reproduction. It omits nuPlan's SDK-dependent raw-data preprocessing, released checkpoint loading, the upstream DPM-Solver dependency, distributed training, and closed-loop runners. The model is intended to follow the official release; operator and checkpoint parity remain unverified. See [the source audit](docs/paper-to-code.md) and [paper brief](docs/paper-brief.md).

## Learn the architecture

Open `notebooks/diffusion_planner_walkthrough.ipynb` and run its cells in order. It covers:

1. The global architecture map: scene encoders, context fusion, route/time conditioning, DiT denoising, and losses.
2. Module I/O cards for every utility and model component, including shape-changing operations, masks, and embeddings.
3. Mathematical derivations for normalization, VP-SDE noising, attention, adaLN, and the released `x_start` objective.
4. An executed tensor ledger over synthetic official-dimension inputs.

## Install and test

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

## Upstream dataset and preprocessing

The official Diffusion-Planner model is trained with [nuPlan v1.1](https://www.nuscenes.org/nuplan#download).
Create an account, accept the [nuPlan terms of use](https://www.nuscenes.org/terms-of-use), and manually
download all required archives from the official portal. The nuPlan devkit's
[dataset setup guide](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html) specifies the
archives and extraction layout. After extraction, the upstream preprocessing script needs paths such as:

```text
~/nuplan/dataset/nuplan-v1.1/splits/trainval
~/nuplan/dataset/maps
```

> **This repository does not download the dataset.** It does not include nuPlan SDK preprocessing. A practical single-process loader and training runner accept upstream-generated NPZ files.

To generate the data used by the official training pipeline, install the [nuPlan devkit](https://github.com/motional/nuplan-devkit)
and the [official Diffusion-Planner repository](https://github.com/ZhengYinan-AIR/Diffusion-Planner):

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
python3 -m pip install -e .
python3 -m pip install -r requirements.txt

cd ..
git clone https://github.com/ZhengYinan-AIR/Diffusion-Planner.git
cd Diffusion-Planner
python3 -m pip install -e .
python3 -m pip install -r requirements_torch.txt
```

In the upstream [`data_process.sh`](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/main/data_process.sh),
set the three variables to your extracted nuPlan directories and the desired generated-data output directory:

```bash
NUPLAN_DATA_PATH="$HOME/nuplan/dataset/nuplan-v1.1/splits/trainval"
NUPLAN_MAP_PATH="$HOME/nuplan/dataset/maps"
TRAIN_SET_PATH="/path/to/diffusion-planner-train-set"
```

Then run the upstream preprocessing script:

```bash
chmod +x data_process.sh
./data_process.sh
```

The script invokes `data_process.py` with those paths and `--total_scenarios 1000000`. Point the practical runner at the resulting NPZ files and a JSON list of relative paths:

```bash
python -m diffusion_planner.train --data-dir /path/to/generated-data \
  --manifest /path/to/train-list.json --output checkpoints/example --epochs 1
# To resume at an epoch boundary, keep all options the same and increase --epochs:
python -m diffusion_planner.train --data-dir /path/to/generated-data \
  --manifest /path/to/train-list.json --output checkpoints/example --epochs 2 --resume
```

This runner hashes sample contents and the recipe to reject changed data or settings on resume. It runs the actual loss and optimizer on CPU by default. It is a practical baseline, not the official distributed training recipe: it does not yet implement EMA, scheduling, augmentation, or official validation. Optional MLflow session logging is available but has not been integration-tested against a real MLflow server. Only load checkpoints you trust. The sampler in `sampling.py` includes an unguided DPM-Solver++ path checked against one fixed upstream operator fixture, plus a simpler deterministic alternative. Guided and checkpoint parity are unverified.

To compute diagnostic open-loop ego displacement on a separate NPZ split:

```bash
python -m diffusion_planner.evaluate --data-dir /path/to/generated-data \
  --manifest /path/to/validation-list.json --checkpoint checkpoints/example/latest.pt
```

This reports ADE/FDE in meters for a single sampled trajectory. It is not a nuPlan closed-loop score. Use the same `--config` file as training when overriding model dimensions.

Optional tracking uses `pip install -e ".[tracking]"` and `--mlflow-uri sqlite:///mlflow.db`; start the UI with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Each resumed session gets a separate run linked by the same data/recipe identity. MLflow is disabled by default.

To measure loader, cached optimizer step, and end-to-end CPU throughput on supplied preprocessed data, run `python -m diffusion_planner.benchmark --data-dir /path/to/generated-data --manifest /path/to/train-list.json`. This diagnostic does not tune settings, control the OS cache, or measure GPU memory.



### Released-checkpoint tensor contract

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
  data.py                   # upstream NPZ dataset and target encoding
  train.py, training.py     # practical runner and epoch checkpoints
  sampling.py               # unguided DPM-Solver++ and deterministic sampler
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
