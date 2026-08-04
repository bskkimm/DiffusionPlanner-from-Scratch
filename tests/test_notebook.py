import json
from pathlib import Path


def test_notebook_is_valid_and_compiles() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "diffusion_planner_walkthrough.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert notebook["nbformat"] == 4
    compile(code, str(path), "exec")
    assert "AgentFusionEncoder" in code
    assert "LaneFusionEncoder" in code
    assert "RouteEncoder" in code
    assert "TrajectoryDiT" in code
    assert "diffusion_loss" in code
