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
