# Source and implementation audit

This is an initial code audit, not a claim of paper or checkpoint parity. The
paper is https://arxiv.org/abs/2501.15564; the official code was inspected at
`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df` from the local checkout. The
paper and supplement could not be independently fetched during this audit.

| Area | Local implementation | Official source examined | Evidence / gap |
| --- | --- | --- | --- |
| Context encoding | `src/diffusion_planner/model/encoder.py` | `diffusion_planner/model/module/encoder.py` | Shape tests exist; numerical parity unverified. |
| Route conditioning and denoising | `src/diffusion_planner/model/dit.py` | `diffusion_planner/model/module/dit.py`, `decoder.py` | Shape tests exist; checkpoint parity unverified. |
| VP noising and objective | `src/diffusion_planner/diffusion.py`, `model/planner.py` | `model/diffusion_utils/sde.py`, `loss.py` | Loss/backward smoke test exists; fixed-input parity unverified. |
| Training data | `src/diffusion_planner/data.py` | `utils/dataset.py`, `train_epoch.py` | NPZ loading, participant slicing and target encoding are implemented; real samples and source parity remain unverified. The neighbor mask is computed before heading conversion. |
| Observation normalization | `normalizer.py` | `normalization.json`, `utils/normalizer.py`, `train_epoch.py` | Caller must supply matching statistics and normalize inputs; no training runner wires this up. |
| Training | `train.py`, `training.py` | `train_epoch.py`, `train_predictor.py` | Practical single-process AdamW training and epoch resume are tested synthetically; official EMA, scheduling, augmentation and distributed recipe are missing. |
| Sampling | Denoiser callback and `sampling.py` | `model/diffusion_utils/sampling.py` | Practical deterministic sampler differs from upstream second-order DPM-Solver++; parity is unverified. |
| Evaluation | `metrics.py`, `evaluate.py` | `planner/planner.py` and simulator configuration | Diagnostic single-sample ego ADE/FDE only; no official closed-loop evaluation or measured planning metrics. |
| Notebook | `notebooks/diffusion_planner_walkthrough.ipynb` | Model modules above | Static syntax test only; fresh-process execution and matched-weight output/gradient parity remain open. |

The official pipeline normalizes observations after any augmentation, while
future `(x, y, cos(yaw), sin(yaw))` states use the separate state normalizer.
Zero padding and the order of these operations matter. Frame, units, full
objective weights, splits and metrics require further source/paper verification.
Synthetic tests cannot establish planning performance.
