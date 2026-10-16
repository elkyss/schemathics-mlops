import json
from datetime import date

import mlflow
import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.slow


@pytest.fixture
def pipeline(tmp_path, tmp_mlflow, synthetic_csv, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    from bl_ranking.config import Settings
    from bl_ranking.data.source import DuckDBSource
    from bl_ranking.models.registry import ModelRegistryManager
    from bl_ranking.payout.backends import FallbackBackend
    from bl_ranking.training.pipeline import TrainingPipeline
    from bl_ranking.training.tracking import ExperimentTracker

    settings = Settings(_env_file=None, warehouse_dir=tmp_path / "wh", mlflow_tracking_uri=tmp_mlflow,
                        model_name="bl_ranker_test", api_url="http://127.0.0.1:9")  # nothing listening
    source = DuckDBSource(tmp_path / "wh" / "bl.duckdb", "bl.bl_full_data")
    source.ingest_csv(synthetic_csv)
    return TrainingPipeline(settings, source, ExperimentTracker(tmp_mlflow),
                            ModelRegistryManager(MlflowClient(), "bl_ranker_test"), FallbackBackend(),
                            min_rows=100)


def test_train_test_run_logs_metrics_and_researcher_log(pipeline):
    res = pipeline.run("train_test")
    run = MlflowClient().get_run(res.run_id)
    assert run.info.status == "FINISHED" and "sold_overall_f1" in run.data.metrics
    assert run.data.params["payout_backend"] == "fallback" and "data_snapshot_sha256" in run.data.tags
    arts = [a.path for a in MlflowClient().list_artifacts(res.run_id, "researcher_log")]
    assert arts and arts[0].endswith(".log")
    assert res.version is None


def test_production_lifecycle_gate(pipeline):
    first = pipeline.run("production", as_of=date(2026, 1, 14))
    assert first.gate.decision == "no_champion" and first.promoted and first.version == "1"
    mv = MlflowClient().get_model_version("bl_ranker_test", "1")
    assert mv.tags["data_end_date"] == "2026-01-14" and "Gate" in mv.description

    second = pipeline.run("production")  # a new week the champion never saw → real gate
    assert second.gate.decision in {"pass", "fail"} and second.version == "2"
    assert "sold_overall_f1" in second.gate.champion

    third = pipeline.run("production")  # identical snapshot (e.g. weekly job, table not refreshed)
    assert third.gate.decision == "skipped_unchanged_data" and third.version is None and not third.promoted


def test_too_little_new_data_is_registered_but_not_promoted(pipeline):
    """Final review: a skipped gate must not auto-promote — the new version waits at @challenger."""
    pipeline.run("production", as_of=date(2026, 1, 10))
    newer = pipeline.run("production", as_of=date(2026, 1, 14))  # 4 days newer: no unseen full week
    assert newer.gate.decision == "skipped_no_new_data" and newer.version == "2" and not newer.promoted
    assert str(pipeline.registry.get("champion").version) == "1"
    assert str(pipeline.registry.get("challenger").version) == "2"


def test_last_success_written(pipeline):
    pipeline.run("production", as_of=date(2026, 1, 14))
    rec = json.loads((pipeline.settings.warehouse_dir / "last_success.json").read_text())
    assert rec["mode"] == "production" and rec["version"] == "1"


def test_validation_failure_marks_run_failed(pipeline):
    from bl_ranking.data.source import DataValidationError

    pipeline.min_rows = 10**9
    with pytest.raises(DataValidationError):
        pipeline.run("train_test")
    runs = mlflow.search_runs(experiment_names=["bl-ranking/train_test"], output_format="list")
    assert runs[0].info.status == "FAILED"
