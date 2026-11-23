"""Optional MLflow session logging; each resumed session has its own run."""


class Tracker:
    def __init__(self, uri: str | None):
        self.uri = uri
        self.mlflow = None

    def start(self, params: dict):
        if self.uri is None:
            return
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("install the tracking extra to enable MLflow") from exc
        self.mlflow = mlflow
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment("diffusion-planner")
        mlflow.start_run()
        mlflow.log_params(params)

    def log(self, metrics: dict, *, step: int, tags: dict):
        if self.mlflow is not None:
            self.mlflow.log_metrics(metrics, step=step)
            self.mlflow.set_tags(tags)

    def close(self, status="FINISHED"):
        if self.mlflow is not None:
            self.mlflow.end_run(status=status)
            self.mlflow = None
