import random

import pytest
import torch

from diffusion_planner.training import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_rng(tmp_path):
    torch.manual_seed(123)
    random.seed(123)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, identity="data-a")
    expected_output = model(torch.ones(1, 2)).detach().clone()
    expected_random = torch.rand(2)
    expected_python = random.random()
    with torch.no_grad():
        model.weight.zero_()
    assert load_checkpoint(path, model, optimizer, identity="data-a") == 1
    torch.testing.assert_close(model(torch.ones(1, 2)), expected_output)
    torch.testing.assert_close(torch.rand(2), expected_random)
    assert random.random() == expected_python
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, model, optimizer, identity="data-b")


def test_training_step_updates_planner():
    from test_model import config, inputs

    from diffusion_planner import DiffusionPlanner
    from diffusion_planner.training import train_step

    torch.manual_seed(10)
    cfg = config()
    model = DiffusionPlanner(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.decoder.final_layer.proj[-1].weight.detach().clone()
    losses = train_step(
        model,
        optimizer,
        inputs(cfg),
        torch.randn(2, cfg.future_len, 4),
        torch.randn(2, cfg.predicted_neighbor_num, cfg.future_len, 4),
        torch.zeros(2, cfg.predicted_neighbor_num, cfg.future_len, dtype=torch.bool),
    )
    assert all(torch.isfinite(value) for value in losses.values())
    assert not torch.equal(before, model.decoder.final_layer.proj[-1].weight)


def test_resume_matches_uninterrupted_training(tmp_path):
    from test_model import config, inputs

    from diffusion_planner import DiffusionPlanner
    from diffusion_planner.training import train_step

    torch.set_num_threads(1)
    cfg = config()
    torch.manual_seed(42)
    batch = inputs(cfg, batch=1)
    ego = torch.randn(1, cfg.future_len, 4)
    neighbors = torch.randn(1, cfg.predicted_neighbor_num, cfg.future_len, 4)
    mask = torch.zeros(1, cfg.predicted_neighbor_num, cfg.future_len, dtype=torch.bool)
    model = DiffusionPlanner(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_step(model, optimizer, batch, ego, neighbors, mask)
    save_checkpoint(tmp_path / "epoch.pt", model, optimizer, epoch=1, identity="same")
    train_step(model, optimizer, batch, ego, neighbors, mask)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    resumed = DiffusionPlanner(cfg)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    load_checkpoint(tmp_path / "epoch.pt", resumed, resumed_optimizer, identity="same")
    train_step(resumed, resumed_optimizer, batch, ego, neighbors, mask)
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_dataset_identity_detects_changed_sample(tmp_path):
    import json

    from diffusion_planner.training import dataset_identity

    (tmp_path / "a.npz").write_bytes(b"one")
    manifest = tmp_path / "list.json"
    manifest.write_text(json.dumps(["a.npz"]))
    first = dataset_identity(tmp_path, manifest, {"lr": 0.1})
    (tmp_path / "a.npz").write_bytes(b"two")
    assert dataset_identity(tmp_path, manifest, {"lr": 0.1}) != first
    assert dataset_identity(tmp_path, manifest, {"lr": 0.2}) != first


def test_training_cli_runs_and_resumes(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from dataclasses import asdict

    import numpy as np
    from test_model import config

    cfg = config()
    np.savez(
        tmp_path / "sample.npz",
        ego_current_state=np.ones(10, dtype="float32"),
        ego_agent_future=np.ones((cfg.future_len, 3), dtype="float32"),
        neighbor_agents_past=np.ones((cfg.agent_num, cfg.time_len, 11), dtype="float32"),
        neighbor_agents_future=np.ones(
            (cfg.predicted_neighbor_num, cfg.future_len, 3), dtype="float32"
        ),
        static_objects=np.ones((cfg.static_objects_num, 10), dtype="float32"),
        lanes=np.ones((cfg.lane_num, cfg.lane_len, 12), dtype="float32"),
        lanes_speed_limit=np.ones((cfg.lane_num, 1), dtype="float32"),
        lanes_has_speed_limit=np.ones((cfg.lane_num, 1), dtype="bool"),
        route_lanes=np.ones((cfg.route_num, cfg.lane_len, 12), dtype="float32"),
    )
    (tmp_path / "list.json").write_text(json.dumps(["sample.npz"]))
    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg)))
    cmd = [
        sys.executable,
        "-m",
        "diffusion_planner.train",
        "--data-dir",
        str(tmp_path),
        "--manifest",
        str(tmp_path / "list.json"),
        "--config",
        str(tmp_path / "config.json"),
        "--output",
        str(tmp_path / "run"),
        "--epochs",
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run(cmd + ["1"], check=True, env=env, capture_output=True, text=True)
    subprocess.run(cmd + ["2", "--resume"], check=True, env=env, capture_output=True, text=True)
    assert (tmp_path / "run" / "latest.pt").exists()
    assert len(list((tmp_path / "run").glob("session-*.json"))) == 2
