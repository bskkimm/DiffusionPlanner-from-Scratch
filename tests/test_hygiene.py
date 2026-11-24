"""Repository hygiene checks run without external services."""

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def repository_files():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    return [ROOT / name for name in sorted(set(names)) if (ROOT / name).is_file()]


def test_no_large_files_private_keys_or_conflict_markers():
    private_headers = [
        b"-----BEGIN " + kind + b"PRIVATE KEY-----" for kind in (b"", b"RSA ", b"EC ", b"OPENSSH ")
    ]
    for path in repository_files():
        data = path.read_bytes()
        assert len(data) <= 2_000_000, path
        assert not any(header in data for header in private_headers), path
        if path.suffix in {".py", ".md", ".toml", ".yml", ".yaml", ".json"}:
            lines = data.decode("utf-8").splitlines()
            assert all(line == line.rstrip() for line in lines), path
            assert all(not line.startswith("<" * 7 + " ") for line in lines), path
            assert all(not line.startswith(">" * 7 + " ") for line in lines), path


def test_structured_files_parse_and_notebook_outputs_are_empty():
    try:
        import tomllib
    except ImportError:
        tomllib = None  # Python 3.10 TOML is checked on the 3.11 CI job.
    for path in repository_files():
        if path.suffix in {".json", ".ipynb"}:
            content = json.loads(path.read_text(encoding="utf-8"))
            if path.suffix == ".ipynb":
                for cell in content["cells"]:
                    if cell["cell_type"] == "code":
                        assert cell["execution_count"] is None and cell["outputs"] == []
        elif path.suffix in {".yaml", ".yml"}:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        elif path.suffix == ".toml" and tomllib is not None:
            tomllib.loads(path.read_text(encoding="utf-8"))
