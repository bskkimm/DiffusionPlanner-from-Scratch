import json
import re
from pathlib import Path

from diffusion_planner import PlannerConfig

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_local_links_and_recipes():
    for document in (ROOT / "README.md", ROOT / "COMMAND_GUIDE.md", ROOT / "docs/paper-to-code.md"):
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            assert (document.parent / path).exists(), (document, target)
    for name in ("paper-model", "smoke-model"):
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        PlannerConfig(**config)


def test_results_manifest_does_not_claim_unmeasured_results():
    result = json.loads((ROOT / "reports/results.json").read_text())
    assert result["paper_results_reproduced"] is False
    assert result["real_dataset"] is None
    assert result["official_closed_loop_metrics"] is None
