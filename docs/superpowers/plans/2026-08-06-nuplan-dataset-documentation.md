# nuPlan Dataset Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a README guide for manually downloading nuPlan v1.1 and invoking the official Diffusion-Planner preprocessing pipeline.

**Architecture:** Keep this repository model-only. The README links to official nuPlan and upstream Diffusion-Planner resources, describes the upstream data paths and command, and clearly states that no downloader, SDK preprocessing, loader, or training runner is included here.

**Tech Stack:** Markdown, pytest.

## Global Constraints

- Do not download dataset files locally.
- Do not vendor nuPlan devkit or upstream preprocessing code.
- Do not claim that this standalone implementation consumes generated upstream training data.
- Use nuPlan v1.1 `trainval` database and `maps` paths in examples.
- Keep all external setup and preprocessing commands attributed to the official upstream projects.

---

### Task 1: Document Dataset Access and Upstream Preprocessing

**Files:**
- Modify: `README.md:17-36`
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: nuPlan's official dataset setup URL and the upstream `data_process.sh` interface: `--data_path`, `--map_path`, `--save_path`, and `--total_scenarios`.
- Produces: A `Dataset and preprocessing` README section containing manual access steps, expected paths, upstream setup commands, preprocessing configuration, and explicit project boundaries.

- [ ] **Step 1: Add a failing documentation-contract test**

Add this test to `tests/test_notebook.py`:

```python
from pathlib import Path


def test_readme_documents_nuplan_preprocessing() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## Dataset and preprocessing" in readme
    assert "nuplan-v1.1/splits/trainval" in readme
    assert "~/nuplan/dataset/maps" in readme
    assert "data_process.sh" in readme
    assert "No dataset is downloaded by this repository" in readme
```

- [ ] **Step 2: Run the documentation-contract test to verify it fails**

Run: `pytest tests/test_notebook.py::test_readme_documents_nuplan_preprocessing -v`

Expected: FAIL because the README does not yet contain the dataset section.

- [ ] **Step 3: Add the README section**

Insert a `## Dataset and preprocessing` section after `## Install and test` that includes:

```markdown
The official model is trained with nuPlan v1.1. Create an account, accept the
terms, and manually download all required archives from the official nuPlan
download portal. Extract them so the train/validation databases and maps are
available at paths such as:

```text
~/nuplan/dataset/nuplan-v1.1/splits/trainval
~/nuplan/dataset/maps
```

No dataset is downloaded by this repository. This implementation does not
include nuPlan SDK preprocessing, a data loader, or a training runner.
```

Then provide the upstream nuPlan devkit and Diffusion-Planner installation commands, followed by the exact `data_process.sh` variables and invocation:

```bash
NUPLAN_DATA_PATH="$HOME/nuplan/dataset/nuplan-v1.1/splits/trainval"
NUPLAN_MAP_PATH="$HOME/nuplan/dataset/maps"
TRAIN_SET_PATH="/path/to/diffusion-planner-train-set"

python data_process.py \
  --data_path "$NUPLAN_DATA_PATH" \
  --map_path "$NUPLAN_MAP_PATH" \
  --save_path "$TRAIN_SET_PATH" \
  --total_scenarios 1000000
```

Link the nuPlan download/setup page, nuPlan terms, upstream Diffusion-Planner repository, and the exact upstream preprocessing script. State that users must configure the same values in upstream `data_process.sh` before running `./data_process.sh`, and that the generated data requires the upstream training pipeline rather than this repository.

- [ ] **Step 4: Run the documentation-contract test to verify it passes**

Run: `pytest tests/test_notebook.py::test_readme_documents_nuplan_preprocessing -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest`

Expected: PASS.

- [ ] **Step 6: Commit the documentation update**

```bash
git add README.md tests/test_notebook.py
git commit -m "docs: add nuplan dataset setup"
```
