"""The training pipeline the weekly schedule runs.

train_test : snapshot → validate → researcher's train/test evaluation → metrics + researcher log in MLflow.
production : snapshot → validate → [gate evaluation if the champion never saw the newest week] → train on all
             data → package bl_ranker → register as @challenger with a model card → promote to @champion if the
             gate allows → ask the serving API to hot-reload.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import httpx
import joblib
import mlflow
from catboost import CatBoostClassifier

from bl_ranking.adapters import TrainAdapter
from bl_ranking.config import Settings
from bl_ranking.data.source import DataSource, Snapshot, validate_snapshot
from bl_ranking.models.pyfunc import API_RECORD, _build_payout_model, log_bl_ranker
from bl_ranking.models.registry import CHAMPION, ModelRegistryManager, PromotionRefused
from bl_ranking.payout.backends import DeadlineRegressor, PayoutBackend, patch_module
from bl_ranking.training.gate import GateResult, evaluate_gate, score_on_holdout, should_run_gate
from bl_ranking.training.tracking import ExperimentTracker

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
GATE_HOLDOUT_DAYS = 7  # the researcher's train_test holdout (bl_models_train.py:235)


@dataclass
class PipelineResult:
    run_id: str
    version: str | None = None
    gate: GateResult | None = None
    promoted: bool = False


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _upstream_commit() -> str:
    path = ROOT / "reference" / "UPSTREAM_COMMIT"
    return path.read_text().strip()[:12] if path.exists() else "unknown"


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("catboost", "tabpfn-client", "tabpfn", "pandas", "scikit-learn", "names-dataset", "mlflow"):
        try:
            out[f"lib_{pkg}"] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


class TrainingPipeline:
    def __init__(self, settings: Settings, source: DataSource, tracker: ExperimentTracker,
                 registry: ModelRegistryManager, backend: PayoutBackend, min_rows: int = 1000):
        self.settings, self.source, self.tracker, self.registry, self.backend = (
            settings, source, tracker, registry, backend)
        self.min_rows = min_rows
        patch_module(sys.modules[TrainAdapter.__mro__[1].__module__], backend)

    # ------------------------------------------------------------------------------------------------------
    def run(self, mode: Literal["train_test", "production"], as_of: date | None = None,
            trigger: str = "manual") -> PipelineResult:
        work = self.settings.warehouse_dir / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        tags = {"mode": mode, "trigger": trigger, "payout_backend": self.backend.name, "git_sha": _git_sha(),
                "upstream_commit": _upstream_commit()}
        with self.tracker.run(mode, tags) as run_id:
            snap = self.source.snapshot(work / "data", as_of=as_of)
            self.tracker.log_snapshot(snap)
            validate_snapshot(snap, min_rows=self.min_rows)
            result = self._train_test(snap, work, run_id) if mode == "train_test" else \
                self._production(snap, work, run_id)
        self._write_last_success(mode, result.version)
        return result

    def _adapter(self, snap: Snapshot, out: Path, train_test: bool) -> TrainAdapter:
        out.mkdir(parents=True, exist_ok=True)
        return TrainAdapter(str(snap.path.parent) + "/", snap.path.name, str(out) + "/", train_test=train_test,
                            holdout_days=self.settings.holdout_days_prod)

    def _log_training(self, adapter: TrainAdapter, holdout_days: int) -> None:
        cb = adapter.cb_model_.get_params()
        self.tracker.log_params({
            "payout_backend": self.backend.name, "holdout_days": holdout_days,
            "cb_depth": cb.get("depth"), "cb_n_estimators": cb.get("n_estimators"),
            "cb_random_seed": cb.get("random_seed"), "cb_eval_metric": cb.get("eval_metric"),
            "tabpfn_context_rows": len(adapter.tfm_context_["x"]), "tabpfn_ignore_pretraining_limits": True,
            "n_features": len(adapter.tfm_context_["columns"]),
            **{f"rows_{k}": v for k, v in adapter.row_counts_.items()}, **_versions(),
        })
        self.tracker.log_artifact(adapter.log_path_, "researcher_log")

    def _train_test(self, snap: Snapshot, work: Path, run_id: str) -> PipelineResult:
        adapter = self._adapter(snap, work / "train_test", train_test=True)
        adapter.fit_()
        self._log_training(adapter, GATE_HOLDOUT_DAYS)
        self.tracker.log_metrics(adapter.metrics_)
        return PipelineResult(run_id=run_id)

    def _production(self, snap: Snapshot, work: Path, run_id: str) -> PipelineResult:
        data_end = date.fromisoformat(snap.max_session_dt[:10])
        gate = self._gate(snap, work, data_end)
        self.tracker.set_tags({"gate": gate.decision})
        self.tracker.log_metrics(gate.challenger, prefix="gate_challenger_")
        if gate.champion:
            self.tracker.log_metrics(gate.champion, prefix="gate_champion_")

        predictors = work / "predictors"
        adapter = self._adapter(snap, predictors, train_test=False)
        adapter.fit_()
        self._log_training(adapter, self.settings.holdout_days_prod)
        if self.backend.name == "api":
            adapter.model_tfm_.save_model(predictors / API_RECORD)  # server-side fit id: serving never refits

        info = log_bl_ranker(predictors, self.backend.name, self.settings.tabpfn_deadline_s)
        version = self.registry.register(info.model_uri, tags={
            "payout_backend": self.backend.name, "data_end_date": data_end.isoformat(), "gate": gate.decision,
            "snapshot_sha256": snap.sha256, "git_sha": _git_sha(), "upstream_commit": _upstream_commit(),
            "run_id": run_id}, description=self._model_card(snap, gate), model_id=info.model_id, run_id=run_id)
        promoted = self._maybe_promote(version, gate)
        return PipelineResult(run_id=run_id, version=version, gate=gate, promoted=promoted)

    # ------------------------------------------------------------------------------------------------------
    def _gate(self, snap: Snapshot, work: Path, data_end: date) -> GateResult:
        champion = self.registry.get(CHAMPION)
        if champion is None:
            return GateResult("no_champion", "first model — nothing to compare against", {}, None)
        champ_end = champion.tags.get("data_end_date")
        if not should_run_gate(data_end, date.fromisoformat(champ_end) if champ_end else None, GATE_HOLDOUT_DAYS):
            return GateResult("skipped_no_new_data",
                              f"champion v{champion.version} already trained on data up to {champ_end}", {}, None)
        challenger = self._adapter(snap, work / "gate", train_test=True)
        challenger.fit_()
        cb, payout = self._champion_models(champion.version)
        champ_metrics = score_on_holdout(cb, payout, challenger.x_test_all_days_, challenger.y_test_all_days_)
        return evaluate_gate(challenger.metrics_, champ_metrics, self.settings.gate_max_regression)

    def _champion_models(self, version: str):
        """Load only the champion's model files (not the serving pyfunc: that would load NameDataset again)."""
        model_dir = Path(mlflow.artifacts.download_artifacts(f"models:/{self.registry.name}/{version}"))
        predictors = model_dir / "artifacts" / "predictors"
        backend = self.registry.version(version).tags.get("payout_backend", "fallback")
        context = joblib.load(predictors / "payout_tfm_context.joblib")
        payout = _build_payout_model(predictors, context, backend, deadline_s=60.0)
        cb = CatBoostClassifier(allow_writing_files=False).load_model(str(predictors / "CB_bl_lead.cbm"))
        return cb, payout.primary if isinstance(payout, DeadlineRegressor) else payout

    def _maybe_promote(self, version: str, gate: GateResult) -> bool:
        if not (self.settings.auto_promote and gate.allows_promotion):
            log.warning("v%s stays @challenger: gate=%s (%s)", version, gate.decision, gate.reason)
            return False
        try:
            self.registry.promote(version, reason=f"auto: gate={gate.decision}")
        except PromotionRefused as exc:
            log.warning("v%s not promoted: %s", version, exc)
            return False
        self._reload_api()
        return True

    def _reload_api(self) -> None:
        headers = {"X-Admin-Token": self.settings.admin_token.get_secret_value()} if self.settings.admin_token else {}
        try:
            httpx.post(f"{self.settings.api_url}/admin/reload", headers=headers, timeout=60).raise_for_status()
        except Exception as exc:  # the API may not be running (first deploy, batch environment)
            log.info("API reload skipped: %s", exc)

    def _model_card(self, snap: Snapshot, gate: GateResult) -> str:
        def fmt(m):
            return ", ".join(f"{k}={v:.3f}" for k, v in m.items() if k in ("sold_overall_f1", "payout_mae")) or "—"

        return "\n".join([
            "### bl_ranker — Business Loans brand ranking",
            "CatBoost P(lead) × TabPFN payout → brands ranked by expected payout.",
            f"- **Data:** `{snap.table}` {snap.min_session_dt} → {snap.max_session_dt}, {snap.rows} rows, "
            f"snapshot sha256 `{snap.sha256[:12]}`",
            f"- **Payout backend:** {self.backend.name}; production holdout days: {self.settings.holdout_days_prod}",
            f"- **Gate:** {gate.decision} — {gate.reason}",
            f"  - challenger: {fmt(gate.challenger)}; champion: {fmt(gate.champion or {})}",
            f"- **Code:** git `{_git_sha()}`, researcher scripts upstream `{_upstream_commit()}` (unchanged)",
        ])

    def _write_last_success(self, mode: str, version: str | None) -> None:
        self.settings.warehouse_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.warehouse_dir / "last_success.json").write_text(json.dumps(
            {"mode": mode, "version": version, "finished_at": datetime.now(UTC).isoformat(timespec="seconds")}))
