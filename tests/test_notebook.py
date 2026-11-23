import json
from pathlib import Path


def test_notebook_is_valid_and_compiles() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "diffusion_planner_walkthrough.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert notebook["nbformat"] == 4
    compile(code, str(path), "exec")
    assert "AgentFusionEncoder" in code
    assert "LaneFusionEncoder" in code
    assert "RouteEncoder" in code
    assert "TrajectoryDiT" in code
    assert "diffusion_loss" in code
    assert "from diffusion_planner" not in code
    assert "import diffusion_planner" not in code
    assert "inspect.getsource" not in code
    assert "Full architecture pipeline" in markdown
    assert "ARCHITECTURE MAP" in markdown
    assert "MODULE I/O CARD" in markdown
    assert "EXECUTED TENSOR LEDGER" in code
    assert "CONDITIONING INPUTS" in markdown
    assert "Mathematical formulation" in markdown
    assert "q_t(\\mathbf{x}_t" in markdown
    assert "\\operatorname{adaLN}" in markdown
    assert "\\mathcal{L}_{ego}" in markdown
    assert all(character == "\n" or ord(character) >= 32 for character in markdown)
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    for module in (
        "Mlp",
        "MixerBlock",
        "AgentFusionEncoder",
        "LaneFusionEncoder",
        "RouteEncoder",
        "DiTBlock",
    ):
        assert any(f"class {module}" in cell for cell in code_cells)


def test_readme_documents_nuplan_preprocessing() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    assert "## Upstream dataset and preprocessing" in readme
    assert "nuplan-v1.1/splits/trainval" in readme
    assert "~/nuplan/dataset/maps" in readme
    assert "data_process.sh" in readme
    assert "This repository does not download the dataset" in readme


def test_notebook_model_matches_package_outputs_and_gradients():
    import torch
    from test_model import config, inputs

    from diffusion_planner import DiffusionPlanner

    path = Path(__file__).parents[1] / "notebooks" / "diffusion_planner_walkthrough.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("config = PlannerConfig()"):
            break
        exec(compile(source, str(path), "exec"), namespace)
    cfg = config()
    package_model = DiffusionPlanner(cfg).train()
    notebook_model = namespace["DiffusionPlanner"](cfg).train()
    notebook_model.load_state_dict(package_model.state_dict())
    batch = inputs(cfg, batch=1)
    ego = torch.randn(1, cfg.future_len, 4)
    neighbors = torch.randn(1, cfg.predicted_neighbor_num, cfg.future_len, 4)
    mask = torch.zeros(1, cfg.predicted_neighbor_num, cfg.future_len, dtype=torch.bool)
    torch.manual_seed(99)
    left = package_model.diffusion_loss(batch, ego, neighbors, mask)
    (left["ego_planning_loss"] + left["neighbor_prediction_loss"]).backward()
    torch.manual_seed(99)
    right = notebook_model.diffusion_loss(batch, ego, neighbors, mask)
    (right["ego_planning_loss"] + right["neighbor_prediction_loss"]).backward()
    for key in left:
        torch.testing.assert_close(left[key], right[key])
    for (name, parameter), (other_name, other) in zip(
        package_model.named_parameters(), notebook_model.named_parameters()
    ):
        assert name == other_name
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad)


def test_notebook_executes_in_fresh_cpu_process():
    import os
    import subprocess
    import sys

    path = Path(__file__).parents[1] / "notebooks" / "diffusion_planner_walkthrough.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells)
    script = "\n\n".join("".join(cell["source"]) for cell in code_cells)
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""),
        timeout=120,
        check=True,
    )
    assert "synthetic future shape:" in process.stdout
