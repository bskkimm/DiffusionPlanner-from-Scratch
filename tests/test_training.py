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
    continuous = cmd.copy()
    continuous[continuous.index(str(tmp_path / "run"))] = str(tmp_path / "continuous")
    subprocess.run(continuous + ["2"], check=True, env=env, capture_output=True, text=True)
    resumed_state = torch.load(tmp_path / "run" / "latest.pt", weights_only=False)
    continuous_state = torch.load(tmp_path / "continuous" / "latest.pt", weights_only=False)
    for name, value in resumed_state["model"].items():
        torch.testing.assert_close(value, continuous_state["model"][name], rtol=0, atol=0)
    for name, value in resumed_state["ema"].items():
        torch.testing.assert_close(value, continuous_state["ema"][name], rtol=0, atol=0)
    assert resumed_state["scheduler"] == continuous_state["scheduler"]
    evaluation = subprocess.run(
        [
            sys.executable,
            "-m",
            "diffusion_planner.evaluate",
            "--data-dir",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "list.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--checkpoint",
            str(tmp_path / "run" / "latest.pt"),
            "--steps",
            "2",
            "--ema",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert json.loads(evaluation.stdout)["protocol"] == "diagnostic_ego_open_loop"
    assert json.loads(evaluation.stdout)["weights"] == "ema"
    benchmark = subprocess.run(
        [
            sys.executable,
            "-m",
            "diffusion_planner.benchmark",
            "--data-dir",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "list.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--warmup",
            "0",
            "--steps",
            "1",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert set(json.loads(benchmark.stdout)["results"]) == {"loader", "cached_step", "end_to_end"}
    tuning = tmp_path / "tuning.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "diffusion_planner.tuning",
            "--data-dir",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "list.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--max-seconds",
            "30",
            "--output",
            str(tuning),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    report = json.loads(tuning.read_text())
    assert report["baseline_measured"]
    assert report["trials"][0]["status"] == "measured"


def test_evaluation_cli_requires_explicit_checkpoint():
    import os
    import subprocess
    import sys

    process = subprocess.run(
        [sys.executable, "-m", "diffusion_planner.evaluate", "--help"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert "--checkpoint" in process.stdout


def test_progress_eta_uses_current_session_remaining_work():
    from datetime import datetime, timezone

    from diffusion_planner.training import estimate_progress

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    progress = estimate_progress(
        start_epoch=4, completed_epoch=6, total_epochs=10, elapsed_seconds=20, now=now
    )
    assert progress["remaining_seconds"] == 40
    assert progress["finish"].isoformat() == "2026-01-01T00:00:40+00:00"
    assert progress["provisional"] is False


def test_benchmark_requires_real_data_arguments():
    import os
    import subprocess
    import sys

    process = subprocess.run(
        [sys.executable, "-m", "diffusion_planner.benchmark", "--help"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert "--manifest" in process.stdout


def test_checkpoint_restores_scheduler_and_ema(tmp_path):
    from diffusion_planner.training import ExponentialMovingAverage

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    expected = ema.state_dict()["weight"].clone()
    optimizer.step()
    scheduler.step()
    save_checkpoint(
        tmp_path / "state.pt", model, optimizer, epoch=1, identity="a", scheduler=scheduler, ema=ema
    )
    with torch.no_grad():
        ema.shadow["weight"].zero_()
    optimizer.param_groups[0]["lr"] = 5.0
    load_checkpoint(
        tmp_path / "state.pt", model, optimizer, identity="a", scheduler=scheduler, ema=ema
    )
    assert optimizer.param_groups[0]["lr"] == 0.5
    assert scheduler.last_epoch == 1
    torch.testing.assert_close(ema.shadow["weight"], expected)
