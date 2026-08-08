# Notebook Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visual, executable explanation of every Diffusion Planner notebook unit and its tensor I/O evolution.

**Architecture:** Use portable Markdown ASCII diagrams and compact I/O cards inside the existing standalone notebook. Add a final runtime tensor ledger that derives shapes from the synthetic batch, then protect the visual contract through the existing JSON/compile test.

**Tech Stack:** Jupyter Notebook JSON, Markdown, Python, PyTorch, pytest.

## Global Constraints

- Keep the notebook dependency-free beyond its existing PyTorch dependency.
- Use Markdown diagrams and tables rather than rendering libraries or image assets.
- Do not alter model behavior, default dimensions, or the existing executable architecture derivation.
- The ledger is illustrative shape instrumentation, not a trained-model evaluation or nuPlan preprocessing implementation.

---

### Task 1: Add Visual Documentation Contract

**Files:**
- Modify: `tests/test_notebook.py:5-31`
- Modify: `notebooks/diffusion_planner_walkthrough.ipynb`

**Interfaces:**
- Consumes: Notebook Markdown cells joined by `test_notebook_is_valid_and_compiles`.
- Produces: Stable Markdown markers `ARCHITECTURE MAP`, `MODULE I/O CARD`, and `EXECUTED TENSOR LEDGER`.

- [ ] **Step 1: Add failing marker assertions**

```python
assert "ARCHITECTURE MAP" in markdown
assert "MODULE I/O CARD" in markdown
assert "EXECUTED TENSOR LEDGER" in markdown
```

- [ ] **Step 2: Verify the test fails**

Run: `python3 -m pytest tests/test_notebook.py::test_notebook_is_valid_and_compiles -v`

Expected: FAIL because the marker text is missing.

- [ ] **Step 3: Add the visual Markdown cells and ledger code described in Tasks 2 and 3**

- [ ] **Step 4: Verify the test passes**

Run: `python3 -m pytest tests/test_notebook.py::test_notebook_is_valid_and_compiles -v`

Expected: PASS.

### Task 2: Add Architecture Map and Module I/O Cards

**Files:**
- Modify: `notebooks/diffusion_planner_walkthrough.ipynb:13-1050`

**Interfaces:**
- Consumes: Official dimension symbols already used by the notebook: `B`, `P=11`, `T=80`, and `D=192`.
- Produces: A top-level `ARCHITECTURE MAP` and a `MODULE I/O CARD` before each explanatory unit.

- [ ] **Step 1: Replace the current text-only pipeline heading with an annotated architecture map**

Show every data path, concatenation, mask/position flow, route-plus-timestep conditioning, DiT denoising, and loss output with tensor shapes.

- [ ] **Step 2: Insert I/O cards before every notebook unit**

Cover configuration, embeddings/drop path, VP-SDE, normalization, MLP/Mixer, agent/static/lane encoders, fusion, route encoder, timestep embedding, DiT block/final layer, planner loss, guidance, and the end-to-end trace. Each card uses this form:

```text
MODULE I/O CARD: <unit>
input:  <shape and meaning>
transform: <shape-changing operation, mask, or embedding>
output: <shape and meaning>
```

- [ ] **Step 3: Preserve existing source code and prose equations**

Do not change class/function semantics or remove mathematical derivations.

### Task 3: Add Executed Tensor Ledger

**Files:**
- Modify: `notebooks/diffusion_planner_walkthrough.ipynb:1004-1050`

**Interfaces:**
- Consumes: The synthetic official-dimension inputs and instantiated `DiffusionPlanner` from the existing final notebook cell.
- Produces: Printed `EXECUTED TENSOR LEDGER` entries for context tokens, route condition, noisy trajectory, denoiser output, and loss targets.

- [ ] **Step 1: Extend the final trace cell**

Add a ledger header and print shape labels that expose the model path:

```python
print("EXECUTED TENSOR LEDGER")
print("context encoding:", tuple(context["encoding"].shape))
print("route condition:", tuple(context["route_encoding"].shape))
print("denoiser x_start:", tuple(denoised.shape))
```

- [ ] **Step 2: Run the complete notebook test**

Run: `python3 -m pytest tests/test_notebook.py -v`

Expected: PASS with valid JSON and compilable code cells.

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add notebooks/diffusion_planner_walkthrough.ipynb tests/test_notebook.py docs/superpowers/plans/2026-08-08-notebook-visualization.md
git commit -m "docs: visualize notebook tensor flow"
```
