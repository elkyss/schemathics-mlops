"""MLflow experiment tracking: one experiment per mode, comparable params/metrics on every run."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow

from bl_ranking.data.source import Snapshot


class ExperimentTracker:
    def __init__(self, tracking_uri: str, prefix: str = "bl-ranking"):
        self.tracking_uri, self.prefix = tracking_uri, prefix

    @contextmanager
    def run(self, mode: str, tags: dict[str, str]) -> Iterator[str]:
        """An MLflow run in experiment ``{prefix}/{mode}``. Exceptions mark the run FAILED and propagate."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        mlflow.set_experiment(f"{self.prefix}/{mode}")
        with mlflow.start_run(tags=tags) as run:
            yield run.info.run_id

    def log_snapshot(self, s: Snapshot) -> None:
        mlflow.log_params({"data_table": s.table, "data_rows": s.rows, "data_min_session_dt": s.min_session_dt,
                           "data_max_session_dt": s.max_session_dt})
        mlflow.set_tag("data_snapshot_sha256", s.sha256)

    def log_params(self, params: dict) -> None:
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in metrics.items()})

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path)

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)
