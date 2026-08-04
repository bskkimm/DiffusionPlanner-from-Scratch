import json
from pathlib import Path


def test_notebook_is_standalone_and_compiles() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "diffusion_planner_walkthrough.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert notebook["nbformat"] == 4
    assert "from diffusion_planner" not in code
    assert "import diffusion_planner" not in code
    compile(code, str(path), "exec")
