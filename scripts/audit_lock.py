"""Audit the CPU lock, normalizing PyTorch's wheel-only local version suffix.

The vulnerability API recognizes the PyPI release (for example 2.15.0), not
its official +cpu wheel variant. This checks published release advisories; it
does not verify wheel hashes or prove absence of vulnerabilities.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def normalized_lock(text: str) -> str:
    normalized, count = re.subn(r"(?m)^(torch==[0-9]+(?:\.[0-9]+)+)\+cpu$", r"\1", text)
    if count != 1:
        raise ValueError("expected exactly one pinned PyTorch CPU wheel")
    return normalized


def audit(path_to_lock: Path) -> None:
    lock = path_to_lock.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="planner-audit-") as directory:
        path = Path(directory) / "requirements.txt"
        path.write_text(normalized_lock(lock), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--disable-pip",
                "--no-deps",
                "-r",
                str(path),
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
        )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"audit could not complete: {result.stderr}")
    report = json.loads(result.stdout)
    skipped = [item for item in report["dependencies"] if "skip_reason" in item]
    findings = [item for item in report["dependencies"] if item.get("vulns")]
    if skipped or findings or result.returncode != 0:
        raise RuntimeError(f"audit findings={findings}, skipped={skipped}")
    print(f"{path_to_lock}: audited {len(report['dependencies'])} versions; no advisories or skips")


def main() -> None:
    for path_to_lock in (Path("requirements/cpu-py311.lock"), Path("requirements/cpu-py312.lock")):
        audit(path_to_lock)


if __name__ == "__main__":
    main()
