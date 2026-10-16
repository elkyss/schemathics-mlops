"""``bl_ranker``: the researcher's predictor + both models packaged as one MLflow pyfunc.

Loaded once per process: CatBoost from ``CB_bl_lead.cbm``; the payout model from the versioned TabPFN context
(api: the persisted server-side fit id, no re-upload; local: OSS TabPFN fitted once; fallback: brand means).
Each request then runs the original preprocessing → P(lead) × payout → ranking with no disk or model loading.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
from catboost import CatBoostClassifier

from bl_ranking.adapters import LoadedModels, make_predict_adapter
from bl_ranking.payout.backends import (
    DEGRADED,
    ApiBackend,
    BrandMeanRegressor,
    DeadlineRegressor,
    LocalBackend,
    load_api_model,
)

API_RECORD = "tabpfn_api_model.json"
EMPTY_RESULT = {"expected_payout": 0, "prob_lead": 0}  # what the original returns when no row survives preprocessing
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Synthetic person — logged with the model as an example request (never a real lead's data).
EXAMPLE_REQUEST: dict[str, Any] = {
    "session_dt": "2026-01-06 19:24:22", "conversion_dt": "2026-01-06 19:26:10",
    "register_date": "2026-01-06 19:26:07", "campaign_id": 120227360861540306,
    "page": "top10us.com/app/business-loans-v2", "auto_city": "Fort Lauderdale", "auto_country": "United States",
    "auto_state": "Florida", "device_type": "mobile", "sub1": "01121993", "sub2": "01121993 Ad set",
    "sub3": 1513124082, "business_type": "C Corporation", "credit_score": "Very Poor - Under 550",
    "industry": "construction", "loan_amount": "$25,000 - $49,999", "loan_reason": "Equipment purchase",
    "monthly_revenue": "$20,000 - $49,999", "time_in_business": "2+ years", "fname": "Alex", "lname": "Example",
    "cellphone": 7865550100,
}


@dataclass
class RankResult:
    ranking: dict[str, dict[str, float]]
    degraded: bool


def _build_payout_model(predictors_dir: Path, context: dict, payout_backend: str, deadline_s: float):
    fallback = BrandMeanRegressor().fit(context["x"], context["y"])
    if payout_backend == "fallback":
        return fallback
    if payout_backend == "api":
        ApiBackend().set_token(os.environ.get("TABPFN_TOKEN", ""))
        record = predictors_dir / API_RECORD
        if record.exists():
            primary = load_api_model(record, context)
        else:  # no persisted fit (e.g. trained with another backend): one fit at load time, free of credits
            primary = ApiBackend().regressor(ignore_pretraining_limits=True).fit(context["x"], context["y"])
    elif payout_backend == "local":
        primary = LocalBackend().regressor(ignore_pretraining_limits=True).fit(context["x"], context["y"])
    else:
        raise ValueError(f"unknown payout backend {payout_backend!r}")
    return DeadlineRegressor(primary, fallback, deadline_s)


class BLRankerModel(mlflow.pyfunc.PythonModel):
    """Input: post-funnel user dicts. Output: the original ranked-brand dict per user."""

    def load_context(self, context) -> None:
        cfg = context.model_config or {}
        self._init(Path(context.artifacts["predictors"]), cfg.get("payout_backend", "fallback"),
                   float(os.environ.get("TABPFN_DEADLINE_S", cfg.get("deadline_s", 1.2))))

    @classmethod
    def from_dir(cls, predictors_dir: Path, payout_backend: str, deadline_s: float = 1.2) -> BLRankerModel:
        model = cls()
        model._init(Path(predictors_dir), payout_backend, deadline_s)
        return model

    def _init(self, predictors_dir: Path, payout_backend: str, deadline_s: float) -> None:
        self.predictors_path = str(predictors_dir) + "/"  # the original concatenates paths as strings
        self.payout_backend = payout_backend
        context = joblib.load(predictors_dir / "payout_tfm_context.joblib")
        cb_model = CatBoostClassifier(allow_writing_files=False).load_model(
            str(predictors_dir / "CB_bl_lead.cbm"), format="cbm")
        self._loaded = LoadedModels(_build_payout_model(predictors_dir, context, payout_backend, deadline_s),
                                    context["columns"], cb_model)
        # Warm the request path (imports, NameDataset, CatBoost) without spending TabPFN API credits.
        fallback_only = LoadedModels(BrandMeanRegressor().fit(context["x"], context["y"]), context["columns"],
                                     cb_model)
        make_predict_adapter(self.predictors_path, dict(EXAMPLE_REQUEST), fallback_only).predict_()

    def rank(self, user: dict) -> RankResult:
        DEGRADED.set(False)
        result = make_predict_adapter(self.predictors_path, dict(user), self._loaded).predict_()
        if result == EMPTY_RESULT:
            result = {}
        ranking = {str(brand): {k: float(v) for k, v in vals.items()} for brand, vals in result.items()}
        return RankResult(ranking=ranking, degraded=DEGRADED.get())

    def predict(self, context, model_input: list[dict[str, Any]], params=None) -> list[dict[str, Any]]:
        return [self.rank(user).ranking for user in model_input]


def log_bl_ranker(predictors_dir: Path, payout_backend: str, deadline_s: float = 1.2) -> mlflow.models.model.ModelInfo:
    """Log the pyfunc into the active run. No input_example: validating it would run a prediction in the trainer
    (NameDataset load, possibly API credits). The synthetic example request is logged as a plain artifact."""
    mlflow.log_text(json.dumps(EXAMPLE_REQUEST, indent=2), "example_request.json")
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=BLRankerModel(),
        artifacts={"predictors": str(predictors_dir)},
        model_config={"payout_backend": payout_backend, "deadline_s": deadline_s},
        code_paths=[str(PROJECT_ROOT / "src" / "bl_ranking")],
        pip_requirements=str(PROJECT_ROOT / "requirements.txt"),
    )
