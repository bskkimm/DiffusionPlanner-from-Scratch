import sys
from types import SimpleNamespace

from diffusion_planner.tracking import Tracker


def test_disabled_tracker_needs_no_mlflow():
    tracker = Tracker(None)
    tracker.start({"seed": 3})
    tracker.log({"loss": 1.0}, step=1, tags={"status": "running"})
    tracker.close()


def test_enabled_tracker_records_epoch_metrics(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        set_tracking_uri=lambda value: calls.append(("uri", value)),
        set_experiment=lambda value: calls.append(("experiment", value)),
        start_run=lambda: calls.append(("start",)),
        log_params=lambda values: calls.append(("params", values)),
        log_metrics=lambda values, step: calls.append(("metrics", values, step)),
        set_tags=lambda values: calls.append(("tags", values)),
        end_run=lambda status: calls.append(("end", status)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    tracker = Tracker("sqlite:///test.db")
    tracker.start({"seed": 3})
    tracker.log({"loss": 0.5}, step=1, tags={"finish": "tomorrow"})
    tracker.close()
    assert ("metrics", {"loss": 0.5}, 1) in calls
    assert ("end", "FINISHED") in calls


def test_sqlite_tracking_round_trip_when_mlflow_installed(tmp_path):
    import pytest

    mlflow = pytest.importorskip("mlflow", reason="optional MLflow dependency not installed")
    uri = "sqlite:///" + str(tmp_path / "mlflow.db")
    tracker = Tracker(uri)
    tracker.start({"seed": 7})
    tracker.log({"loss": 0.5}, step=1, tags={"finish_utc": "2026-01-01T00:00:00+00:00"})
    tracker.close()
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("diffusion-planner")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.metrics["loss"] == 0.5
    assert runs[0].info.status == "FINISHED"
