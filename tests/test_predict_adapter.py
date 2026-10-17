import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.conftest import EXAMPLE_USER

pytestmark = pytest.mark.slow

USER_B = dict(EXAMPLE_USER, credit_score="Excellent (720+)", monthly_revenue="$100,000 +",
              loan_amount="$100,000+", device_type="desktop", auto_state="Florida", auto_city="Miami")


@pytest.fixture(scope="module")
def ranker(trained_synthetic_dir):
    from bl_ranking.models.pyfunc import BLRankerModel

    return BLRankerModel.from_dir(trained_synthetic_dir, payout_backend="fallback", deadline_s=1.2)


def test_rank_matches_original_predict(trained_synthetic_dir, ranker, tmp_path):
    """Train-serve skew test: the served ranking equals the researcher's own predictor on the same artifacts."""
    from bl_ranking.original import bl_exp_payout_predictor as orig
    from bl_ranking.payout.backends import FallbackBackend, patch_module

    patch_module(sys.modules[orig.__name__], FallbackBackend())
    orig.user_data = dict(EXAMPLE_USER)
    expected = orig.BLPayoutModelsPredict(str(trained_synthetic_dir) + "/", dict(EXAMPLE_USER),
                                          str(tmp_path)).predict_()
    result = ranker.rank(dict(EXAMPLE_USER))
    assert result.ranking == expected
    assert result.ranking and result.degraded is False


def test_rank_output_shape(ranker):
    ranking = ranker.rank(dict(EXAMPLE_USER)).ranking
    ranks = sorted(v["rank"] for v in ranking.values())
    assert ranks == [float(i) for i in range(1, len(ranks) + 1)]
    assert all(isinstance(v["expected_payout"], float) for v in ranking.values())


def test_concurrent_users_get_own_ranking(ranker):
    expected = {"a": ranker.rank(dict(EXAMPLE_USER)).ranking, "b": ranker.rank(dict(USER_B)).ranking}
    assert expected["a"] != expected["b"]
    order = ["a", "b"] * 12
    users = {"a": EXAMPLE_USER, "b": USER_B}
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda u: ranker.rank(dict(users[u])).ranking, order))
    assert all(r == expected[u] for r, u in zip(results, order, strict=True))


def test_not_a_lead_maps_to_empty(ranker, monkeypatch):
    from bl_ranking.predict_adapter import PredictAdapter

    monkeypatch.setattr(PredictAdapter, "predict_", lambda self: {"expected_payout": 0, "prob_lead": 0})
    assert ranker.rank(dict(EXAMPLE_USER)).ranking == {}


def test_pyfunc_roundtrip_through_mlflow(trained_synthetic_dir, tmp_mlflow):
    import mlflow

    from bl_ranking.models.pyfunc import BLRankerModel, log_bl_ranker

    with mlflow.start_run():
        info = log_bl_ranker(trained_synthetic_dir, payout_backend="fallback")
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    ranker = loaded.unwrap_python_model()
    assert isinstance(ranker, BLRankerModel)
    direct = BLRankerModel.from_dir(trained_synthetic_dir, payout_backend="fallback").rank(dict(EXAMPLE_USER))
    assert ranker.rank(dict(EXAMPLE_USER)).ranking == direct.ranking
    assert loaded.predict([dict(EXAMPLE_USER)]) == [direct.ranking]


def test_serve_backend_override(trained_synthetic_dir, monkeypatch):
    """Ops switch: serve a model trained with one payout backend through another (e.g. api outage → local)."""
    from types import SimpleNamespace

    from bl_ranking.models.pyfunc import BLRankerModel

    monkeypatch.setenv("SERVE_PAYOUT_BACKEND", "fallback")
    model = BLRankerModel()
    model.load_context(SimpleNamespace(artifacts={"predictors": str(trained_synthetic_dir)},
                                       model_config={"payout_backend": "api", "deadline_s": 2.0}))
    assert model.payout_backend == "fallback"
    assert model.rank(dict(EXAMPLE_USER)).ranking


def test_sparse_survey_returns_empty_not_crash(ranker):
    """~5% of real users answer <5 of 8 survey questions; the original drops them and then crashes in
    additional_features on the empty frame. Serving must answer with an empty ranking instead of a 500."""
    sparse = dict(EXAMPLE_USER, credit_score=None, industry=None, loan_amount=None, loan_reason=None)
    result = ranker.rank(sparse)
    assert result.ranking == {} and result.reason == "insufficient_survey_answers"


def test_log_model_works_when_installed_as_a_package(trained_synthetic_dir, tmp_mlflow, monkeypatch, tmp_path):
    """On Databricks the code runs from an installed wheel: there is no repo root with src/ or requirements.txt."""
    import mlflow

    from bl_ranking.models import pyfunc

    monkeypatch.setattr(pyfunc, "PROJECT_ROOT", tmp_path / "no-repo-here")
    with mlflow.start_run():
        info = pyfunc.log_bl_ranker(trained_synthetic_dir, payout_backend="fallback")
    local = mlflow.artifacts.download_artifacts(info.model_uri)
    from pathlib import Path

    assert (Path(local) / "code" / "bl_ranking" / "original" / "bl_models_train.py").exists()
