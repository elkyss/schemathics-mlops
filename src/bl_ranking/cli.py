"""`bl` — one entry point for data loading, training, model lifecycle and serving.

Heavy modules (the researcher's code loads NameDataset at import) are imported inside the commands, so
`bl --help` and argument errors stay instant.
"""
from __future__ import annotations

import fcntl
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import typer

from bl_ranking.config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
DRIVE_URL = "https://drive.google.com/drive/folders/1pX2IxzcJEV4Fl520ZkeyT935bNhlotY3"


def _console_logging() -> None:
    """Our pipeline's INFO lines on the console; the researcher's report stays in its own log file (artifact)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.name.startswith("bl_ranking") or r.levelno >= logging.WARNING)


def _settings() -> Settings:
    return Settings()


def _registry(settings: Settings):
    import mlflow
    from mlflow import MlflowClient

    from bl_ranking.models.registry import ModelRegistryManager

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)
    return ModelRegistryManager(MlflowClient(), settings.model_name)


def _reload_api(settings: Settings) -> None:
    import httpx

    headers = {"X-Admin-Token": settings.admin_token.get_secret_value()} if settings.admin_token else {}
    try:
        r = httpx.post(f"{settings.api_url}/admin/reload", headers=headers, timeout=60)
        typer.echo(f"API reloaded: {r.json()}" if r.is_success else f"API reload failed: HTTP {r.status_code}")
    except httpx.HTTPError:
        typer.echo(f"API not reachable at {settings.api_url}; it will load the new champion on start.")


@app.command("load-data")
def load_data(csv: Annotated[Path | None, typer.Option(help="CSV to load (default: DATA_CSV)")] = None):
    """Load the raw CSV into the training table."""
    settings = _settings()
    csv = csv or settings.data_csv
    if not csv.exists():
        typer.echo(f"CSV not found at {csv}; download bl_full_data.csv from the Drive link in README "
                   f"({DRIVE_URL}) into data/.", err=False)
        raise typer.Exit(1)
    from bl_ranking.data.source import get_source

    rows = get_source(settings).ingest_csv(csv)
    typer.echo(f"loaded {rows} rows into {settings.table_name}")


@app.command()
def train(mode: Annotated[str, typer.Option(help="train_test | production")] = "production",
          as_of: Annotated[str | None, typer.Option(help="only use sessions up to this date (YYYY-MM-DD)")] = None,
          trigger: Annotated[str, typer.Option(help="manual | schedule")] = "manual"):
    """Run the training pipeline (production = train on all data, register, gate, promote)."""
    if mode not in ("train_test", "production"):
        raise typer.BadParameter("mode must be train_test or production")
    settings = _settings()
    settings.warehouse_dir.mkdir(parents=True, exist_ok=True)
    lock = (settings.warehouse_dir / "train.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        typer.echo("another training run is already running (warehouse/train.lock); exiting")
        raise typer.Exit(2) from None
    try:
        _console_logging()
        from bl_ranking.data.source import get_source
        from bl_ranking.payout.backends import resolve_backend
        from bl_ranking.training.pipeline import TrainingPipeline
        from bl_ranking.training.tracking import ExperimentTracker

        pipeline = TrainingPipeline(settings, get_source(settings), ExperimentTracker(settings.mlflow_tracking_uri),
                                    _registry(settings), resolve_backend(settings))
        result = pipeline.run(mode, as_of=date.fromisoformat(as_of) if as_of else None, trigger=trigger)
    except Exception as exc:
        typer.echo(f"training failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    finally:
        lock.close()
    gate = f", gate={result.gate.decision} ({result.gate.reason})" if result.gate else ""
    typer.echo(f"run {result.run_id}: version={result.version}{gate}, promoted={result.promoted}")


@app.command()
def promote(version: Annotated[str | None, typer.Option(help="default: current @challenger")] = None,
            force: bool = False, reason: Annotated[str | None, typer.Option()] = None):
    """Move @champion to a version (and hot-reload the API)."""
    settings = _settings()
    registry = _registry(settings)
    if version is None:
        challenger = registry.get("challenger")
        if challenger is None:
            typer.echo("no @challenger to promote")
            raise typer.Exit(1)
        version = str(challenger.version)
    if force and not reason:
        raise typer.BadParameter("--force requires --reason (it is recorded on the version)")
    from bl_ranking.models.registry import PromotionRefused

    try:
        registry.promote(version, force=force, reason=reason or "manual")
    except PromotionRefused as exc:
        typer.echo(f"refused: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"@champion → v{version}")
    _reload_api(settings)


@app.command()
def rollback():
    """Move @champion back to the version it replaced (and hot-reload the API)."""
    settings = _settings()
    from bl_ranking.models.registry import PromotionRefused

    try:
        version = _registry(settings).rollback()
    except PromotionRefused as exc:
        typer.echo(f"refused: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"@champion → v{version}")
    _reload_api(settings)


@app.command()
def status():
    """Champion version, model age and the last successful training run."""
    settings = _settings()
    champion = _registry(settings).get("champion")
    if champion is None:
        typer.echo("champion: none")
    else:
        age_h = (datetime.now(UTC).timestamp() * 1000 - champion.creation_timestamp) / 3_600_000
        tags = champion.tags
        typer.echo(f"champion: v{champion.version} ({tags.get('payout_backend')}, data to {tags.get('data_end_date')}, "
                   f"{age_h:.1f} h old, gate={tags.get('gate')})")
    last = settings.warehouse_dir / "last_success.json"
    typer.echo(f"last_success: {json.loads(last.read_text()) if last.exists() else 'never'}")


@app.command()
def scheduler():
    """Run weekly production training on SCHEDULE_CRON in SCHEDULER_TZ (default: Sunday 05:00 Asia/Jerusalem)."""
    import subprocess
    import sys

    from bl_ranking.scheduler import run_forever

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = _settings()
    typer.echo(f"scheduler: '{settings.schedule_cron}' in {settings.scheduler_tz}")
    run_forever(settings.schedule_cron, settings.scheduler_tz, lambda: subprocess.call(
        [sys.executable, "-m", "bl_ranking.cli", "train", "--mode", "production", "--trigger", "schedule"]))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Serve /v1/predict (one worker per container; scale with replicas)."""
    import uvicorn

    uvicorn.run("bl_ranking.serving.app:build_app", factory=True, host=host, port=port, workers=1)


if __name__ == "__main__":
    app()
