# Command guide

Commands assume the repository root and Python 3.11. The CPU locks cover Linux/Python 3.11 and 3.12; GPU and MLflow dependencies are separate and unverified by those locks. Python 3.10 is unsupported because its compatible filelock versions have known advisories.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/cpu-py311.lock
python -m pip install --no-deps --no-build-isolation .
OMP_NUM_THREADS=1 python -m pytest -q
```

The repository does not download nuPlan. To use real data, first obtain it
under its terms and generate upstream NPZ files. The JSON manifest must be a
nonempty ordered list of relative NPZ paths below `--data-dir`. Train and
validation manifests should identify disjoint scenarios. The runner rejects identical
file contents across splits; different files representing the same scenario still
require dataset-level split checks. The loader expects
upstream observation keys and raw `(x, y, yaw)` future labels. Map positions
and trajectories use ego-frame meters, yaw is radians, and samples are at 10 Hz.
Padding is all zero. Full upstream extraction and simulator integration are not
implemented here.

## Recipes

- **Paper-model:** `configs/paper-model.json` contains the released model
  dimensions. Paper Appendix C reports a global batch of 2048, 500 epochs,
  five warmup epochs, and eight A100 GPUs. This repository's single-process
  runner does not implement that distributed recipe or prove its results.
- **Practical:** choose `configs/paper-model.json` and an authorized local
  dataset, batch size, epoch count and device. `--augment` enables the
  source-informed 10 Hz perturbation/interpolation path. Start with small,
  explicitly authorized experiments; batch size is not automatically tuned.
- **Smoke:** `configs/smoke-model.json` is reduced and not a paper result.
  The tests generate temporary synthetic NPZs and run the full training,
  resume, evaluation and tuning interfaces without external data.

To generate an offline smoke fixture without downloading nuPlan:

```bash
python scripts/create_synthetic_fixture.py --output /tmp/diffusion-smoke
python -m diffusion_planner.train --data-dir /tmp/diffusion-smoke \
  --manifest /tmp/diffusion-smoke/train.json --val-manifest /tmp/diffusion-smoke/validation.json \
  --config configs/smoke-model.json --output /tmp/diffusion-smoke-run --epochs 1 --augment
```

These randomly generated inputs are for testing only and are not evidence of driving performance.

```bash
python -m diffusion_planner.train --data-dir /path/to/npz \
  --manifest /path/to/train.json --val-manifest /path/to/validation.json \
  --config configs/paper-model.json --output checkpoints/example --epochs 1 --batch-size 1 --augment
# Resume from the last completed epoch using the same recipe and data.
python -m diffusion_planner.train --data-dir /path/to/npz \
  --manifest /path/to/train.json --val-manifest /path/to/validation.json \
  --config configs/paper-model.json --output checkpoints/example --epochs 2 --batch-size 1 --augment --resume
```

The runner checks a SHA-256 identity over ordered split files and recipe before
resuming, writes per-session configuration records and saves an atomic
checkpoint at each epoch boundary. Exact continuation has been tested with a
synthetic CPU fixture and zero loader workers. Checkpoints use pickle: load only
trusted files. The `--epochs` value is the desired total, not extra epochs.

```bash
python -m diffusion_planner.evaluate --data-dir /path/to/npz \
  --manifest /path/to/validation.json --config configs/paper-model.json \
  --checkpoint checkpoints/example/latest.pt --ema --collision-guidance
python -m diffusion_planner.benchmark --data-dir /path/to/npz \
  --manifest /path/to/train.json --config configs/paper-model.json
python -m diffusion_planner.tuning --data-dir /path/to/npz \
  --manifest /path/to/train.json --config configs/paper-model.json \
  --max-seconds 60 --output tuning.json
```

Evaluation reports diagnostic single-sample open-loop displacement, not
nuPlan's closed-loop score. Collision guidance and checkpoint compatibility
have only limited CPU fixtures, and exact box ties can differ from upstream.
Benchmarks are CPU paths; the tuner considers only one and two CPU threads,
uses repeated isolated trials, and retains the baseline unless correctness and
a measured speed gain are established. Its time limit applies to trial
execution, including child-process startup. Results are not universal.

To opt into MLflow, install `.[tracking]`, pass
`--mlflow-uri sqlite:///mlflow.db` to the training command, and inspect with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`. Each resumed session starts
a new run linked by its recipe/data identity. Logging failures are visible.

The notebook is `notebooks/diffusion_planner_walkthrough.ipynb`; it runs on
CPU with synthetic data and empty stored outputs. Its final illustrations and
their visual checks are deferred; see `docs/notebook-figures.md`.
