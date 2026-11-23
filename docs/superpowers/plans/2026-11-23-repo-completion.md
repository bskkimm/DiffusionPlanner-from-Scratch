# Diffusion Planner repository completion implementation plan

**Goal:** Complete and verify a source-grounded, teachable planner with honest fidelity and reproducibility evidence.

**Architecture:** Keep the existing plain-PyTorch model, and add distinct data, training, sampling and evaluation interfaces. Share recipe construction across training and benchmarks. Document unverified parity and external-data gates.

**Tech stack:** Python, PyTorch, pytest, Jupyter; optional MLflow and nuPlan SDK.

**Specification:** `docs/paper-brief.md`, `docs/paper-to-code.md`, and pinned playbook in `AGENTS.md`.

## Tasks and gates

- [ ] Verify paper and supplement, upstream configuration and operator-level differences; expand paper-to-code and deviation tables with exact citations.
- [ ] Implement and test upstream data contract, heading conversion, masks, observation normalization and split identity; keep synthetic data clearly distinct.
- [ ] Implement reproducible training, validation, atomic checkpoint/resume, RNG and provenance checks; test interrupted versus uninterrupted training.
- [ ] Implement inference sampler and evaluation interfaces; test numerical behavior, preserving the current-state constraint. Do not label synthetic metrics official.
- [ ] Add optional MLflow, active/wall ETA and disabled-mode tests; establish reproducible dependencies and recipes.
- [ ] Add loader, training-step and end-to-end benchmarks and bounded, correctness-gated tuning.
- [ ] Complete offline notebook walkthrough, execute it in a fresh process and verify model/output/gradient parity. Prepare figure inventory; defer final illustrations and visual checks.
- [ ] Run scientific/integration/packaging checks and current-head hosted CI; obtain current-head scientific and engineering review, address findings, and merge only after applicable gates pass.
- [ ] Run real-data, GPU and official simulator evaluations only when prerequisites and compute authorization are available; report them separately.
