from pathlib import Path

import pytest

from tests.fixtures.synthetic import make_synthetic_csv

ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = ROOT / "data" / "bl_full_data.csv"


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    return make_synthetic_csv(tmp_path_factory.mktemp("synthetic") / "bl_full_data.csv")


# A post-funnel request shaped like the example in bl_exp_payout_predictor.py, with a synthetic person.
EXAMPLE_USER = {
    "session_dt": "2026-01-06 19:24:22", "conversion_dt": "2026-01-06 19:26:10",
    "register_date": "2026-01-06 19:26:07", "campaign_id": 120240633562020306,
    "page": "top10us.com/app/business-loans-v2", "auto_city": "Houston", "auto_country": "United States",
    "auto_state": "Texas", "device_type": "mobile", "sub1": 1122003, "sub2": "01122003 Ad set",
    "sub3": 1511143042, "business_type": "C Corporation", "credit_score": "Very Poor - Under 550",
    "industry": "construction", "loan_amount": "$25,000 - $49,999", "loan_reason": "Equipment purchase",
    "monthly_revenue": "$20,000 - $49,999", "time_in_business": "2+ years", "fname": "Alex", "lname": "Doe",
    "cellphone": 7135550100,
}


@pytest.fixture
def tmp_mlflow(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment("tests")  # reset the process-global active experiment from earlier tests
    yield uri
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="session")
def trained_synthetic_dir(synthetic_csv, tmp_path_factory):
    """Production-mode artifacts from the original training code on synthetic data (fallback payout backend)."""
    import sys

    from bl_ranking.adapters import TrainAdapter
    from bl_ranking.payout.backends import FallbackBackend, patch_module

    patch_module(sys.modules[TrainAdapter.__mro__[1].__module__], FallbackBackend())
    out = tmp_path_factory.mktemp("predictors")
    TrainAdapter(str(synthetic_csv.parent) + "/", synthetic_csv.name, str(out) + "/", train_test=False,
                 holdout_days=0).fit_()
    return out
