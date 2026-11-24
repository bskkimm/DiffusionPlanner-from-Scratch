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
- An unguided second-order DPM-Solver++ sampler and denoiser callback; numerical verification is limited to fixed CPU operator fixtures.
- `notebooks/diffusion_planner_walkthrough.ipynb`, a module-by-module architecture study with input/output traces.

> **Scope.** This is an incomplete implementation, not a complete nuPlan benchmark reproduction. It omits nuPlan's SDK-dependent raw-data preprocessing, verified released checkpoint compatibility, the upstream DPM-Solver dependency, distributed training, and closed-loop runners. The model is intended to follow the official release; operator and checkpoint parity remain unverified. See [the source audit](docs/paper-to-code.md), [paper brief](docs/paper-brief.md), and [diagnostic results manifest](reports/results.json).

## Learn the architecture

Open `notebooks/diffusion_planner_walkthrough.ipynb` and run its cells in order. It covers:

1. The global architecture map: scene encoders, context fusion, route/time conditioning, DiT denoising, and losses.
2. Module I/O cards for every utility and model component, including shape-changing operations, masks, and embeddings.
3. Mathematical derivations for normalization, VP-SDE noising, attention, adaLN, and the released `x_start` objective.
4. An executed tensor ledger over synthetic official-dimension inputs.

## Install and test

Supported CPU environments are Python 3.11 and 3.12. Install `requirements/cpu-py311.lock` or `requirements/cpu-py312.lock`, then install the project with `python -m pip install --no-deps --no-build-isolation .`. The locks do not cover GPU or MLflow dependencies. Python 3.10 support was dropped because its newest compatible filelock version has known advisories and the fixed version requires Python 3.11 or later.

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

This runner hashes sample contents and the recipe to reject changed data or settings on resume. It runs the actual loss and optimizer on CPU by default. It is a practical baseline, not the official distributed training recipe: it implements EMA and an epoch-based warmup, with optional `--augment` ego-state perturbation and `--val-manifest` diagnostic loss; official closed-loop validation remains unavailable. The augmentation expects 10 Hz raw futures with at least 21 steps and has only been compared on synthetic CPU fixtures. Optional MLflow session logging has been checked against a local SQLite store with MLflow 3.14. Only load checkpoints you trust. The sampler in `sampling.py` includes an unguided DPM-Solver++ path checked against three fixed upstream CPU operator fixtures, plus a simpler deterministic alternative. The solver supports released-style collision guidance and experimental caller-provided energy guidance. The collision path matches a non-degenerate upstream CPU fixture; exact ties, checkpoint parity and real-data behavior remain unverified.

To compute diagnostic open-loop ego displacement on a separate NPZ split:

```bash
python -m diffusion_planner.evaluate --data-dir /path/to/generated-data \
  --manifest /path/to/validation-list.json --checkpoint checkpoints/example/latest.pt --ema
```

This reports ADE/FDE in meters for a single sampled trajectory; add `--collision-guidance` to enable the released-style collision score. It is not a nuPlan closed-loop score. Use the same `--config` file as training when overriding model dimensions. `--official-checkpoint` maps keys from the pinned upstream model; its mapping was tested on synthetic upstream weights, not an actual released checkpoint.

Optional tracking uses `pip install -e ".[tracking]"` and `--mlflow-uri sqlite:///mlflow.db`; start the UI with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Each resumed session gets a separate run linked by the same data/recipe identity. MLflow is disabled by default.

To measure loader, cached optimizer step, and end-to-end CPU throughput on supplied preprocessed data, run `python -m diffusion_planner.benchmark --data-dir /path/to/generated-data --manifest /path/to/train-list.json`. This diagnostic does not tune settings, control the OS cache, or measure GPU memory. An optional bounded CPU thread tuner can be run with `python -m diffusion_planner.tuning --data-dir /path/to/generated-data --manifest /path/to/train-list.json --max-seconds 60 --output tuning.json`. It compares one and two CPU threads over repeated isolated FP32 trials, retaining one thread unless correctness holds and mean measured time improves by at least 5%. It uses a single cached batch and does not tune GPU, microbatching, or data loading.



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
