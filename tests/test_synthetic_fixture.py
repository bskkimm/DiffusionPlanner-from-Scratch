import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_generator_produces_disjoint_valid_splits(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/create_synthetic_fixture.py"),
            "--output",
            str(tmp_path),
            "--config",
            str(ROOT / "configs/smoke-model.json"),
        ],
        check=True,
        cwd=ROOT,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
    )
    train = json.loads((tmp_path / "train.json").read_text())
    val = json.loads((tmp_path / "validation.json").read_text())
    assert set(train).isdisjoint(val)
    with np.load(tmp_path / train[0], allow_pickle=False) as sample:
        assert sample["ego_agent_future"].shape == (21, 3)
        assert sample["neighbor_agents_past"].shape == (3, 4, 11)
