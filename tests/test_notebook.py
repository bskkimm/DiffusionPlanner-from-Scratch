import json
from pathlib import Path


def test_notebook_is_valid_and_compiles() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "diffusion_planner_walkthrough.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
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
    assert "CONDITIONING INPUTS" in markdown
    assert "Mathematical formulation" in markdown
    assert "q_t(\\mathbf{x}_t" in markdown
    assert "\\operatorname{adaLN}" in markdown
    assert "\\mathcal{L}_{ego}" in markdown
    assert all(character == "\n" or ord(character) >= 32 for character in markdown)
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for module in ("Mlp", "MixerBlock", "AgentFusionEncoder", "LaneFusionEncoder", "RouteEncoder", "DiTBlock"):
        assert any(f"class {module}" in cell for cell in code_cells)
