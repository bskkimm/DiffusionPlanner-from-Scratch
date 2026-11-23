# Source and implementation audit

This is an initial code audit, not a claim of paper or checkpoint parity. The
paper is https://arxiv.org/abs/2501.15564; the official code was inspected at
`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df` from the local checkout. The paper HTML and its appendix were retrieved on 2026-11-23.

| Area | Local implementation | Official source examined | Evidence / gap |
| --- | --- | --- | --- |
| Context encoding | `src/diffusion_planner/model/encoder.py` | `diffusion_planner/model/module/encoder.py` | Reduced CPU training/evaluation context encoder fixtures match upstream with mapped weights and padded inputs; official checkpoint and full-dimension parity remain unverified. |
| Route conditioning and denoising | `src/diffusion_planner/model/dit.py` | `diffusion_planner/model/module/dit.py`, `decoder.py` | Shape tests exist; checkpoint parity unverified. |
| VP noising and objective | `src/diffusion_planner/diffusion.py`, `model/planner.py` | `model/diffusion_utils/sde.py`, `loss.py` | Loss/backward smoke test exists; fixed-input parity unverified. |
| Training data | `src/diffusion_planner/data.py` | `utils/dataset.py`, `train_epoch.py` | NPZ loading, participant slicing and target encoding are implemented; real samples and source parity remain unverified. The neighbor mask is computed before heading conversion. |
| Observation normalization | `normalizer.py` | `normalization.json`, `utils/normalizer.py`, `train_epoch.py` | Caller must supply matching statistics and normalize inputs; no training runner wires this up. |
| Training | `train.py`, `training.py` | `train_epoch.py`, `train_predictor.py` | Practical single-process AdamW training and epoch resume are tested synthetically; EMA and epoch warmup are included; official augmentation, distributed recipe and model parity remain missing. |
| Sampling | `sampling.py` | `model/diffusion_utils/sampling.py` | Unguided second-order DPM-Solver++ matches three fixed CPU operator fixtures; checkpoint and guided parity are unverified. A simpler deterministic alternative is also included. |
| Evaluation | `metrics.py`, `evaluate.py` | `planner/planner.py` and simulator configuration | Diagnostic single-sample ego ADE/FDE only; no official closed-loop evaluation or measured planning metrics. |
| Notebook | `notebooks/diffusion_planner_walkthrough.ipynb` | Model modules above | Fresh-process CPU execution and matching package loss/gradient tests pass; real-data and official-checkpoint parity remain open. |

The official pipeline normalizes observations after any augmentation, while
future `(x, y, cos(yaw), sin(yaw))` states use the separate state normalizer.
Zero padding and the order of these operations matter. Frame, units, full
objective weights, splits and metrics require further source/paper verification.
Synthetic tests cannot establish planning performance.

## Paper anchors and deliberate differences

- Sections 3.2 and 4.1–4.2 describe VP diffusion, joint trajectory generation, context encoders, route conditioning and DiT. The local model implements an x-start objective; numerical parity with official operators remains unverified.
- Section 4.3 and Appendix C.3 describe differentiable guidance energies. `guidance.py` is not integrated into the practical sampler, so that sampler cannot reproduce guided results.
- Appendix C.1 uses nuPlan training data, 1 million scenarios, 100 m map context and two seconds of history. It perturbs the current ego state, applies quintic interpolation, transforms into ego coordinates, and normalizes with x mean 10 and scale 20. The practical loader expects already preprocessed data and does not implement the augmentation.
- Appendix C.1 reports 500 epochs, batch 2048 on eight A100 GPUs, five warmup epochs and AdamW at 5e-4. The practical runner uses a different single-process configuration, with EMA and epoch warmup but no official augmentation or distributed training. These differences can materially affect learned behavior.
- Appendix C.2 uses DPM-Solver++ and low-temperature sampling. The unguided solver uses second-order multistep updates and log-SNR spacing; three fixed CPU operator fixtures match the released solver. Low-temperature and guided checkpoint parity remain unverified.
- Section 5 evaluates closed-loop driving; diagnostic ADE/FDE in this repository cannot establish those results.

- The draft initially used stochastic depth in Mixer residuals instead of the released MLP dropout and used a different stochastic-depth mask generator. Both were corrected. With identical reduced weights, inputs, masks, mode and random seed, context encoder output differed from upstream by at most `4.8e-7` in the tested CPU run (PyTorch 2.9.1 and timm 1.0.10). This is a local operator check, not full-model or checkpoint parity.
