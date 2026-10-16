import time

import pytest
from fastapi.testclient import TestClient

from bl_ranking.config import Settings
from bl_ranking.models.pyfunc import RankResult
from bl_ranking.serving.app import create_app
from tests.conftest import EXAMPLE_USER

FAKE_RANKING = {"fundera / nerdwallet": {"rank": 1.0, "expected_payout": 42.31},
                "businessloans.com": {"rank": 2.0, "expected_payout": 28.05}}


class FakeRanker:
    def __init__(self, ranking=FAKE_RANKING, degraded=False):
        self.ranking, self.degraded, self.calls = ranking, degraded, 0

    def rank(self, user):
        self.calls += 1
        assert user["fname"] == EXAMPLE_USER["fname"]
        return RankResult(ranking=self.ranking, degraded=self.degraded)


def _client(tmp_path, rankers, admin_token=None):
    versions = iter(rankers)
    settings = Settings(_env_file=None, warehouse_dir=tmp_path, admin_token=admin_token)
    app = create_app(settings, loader=lambda: next(versions))
    client = TestClient(app)
    client.__enter__()
    for _ in range(200):
        if client.get("/ready").status_code == 200:
            break
        time.sleep(0.01)
    return client


@pytest.fixture
def fake(tmp_path):
    ranker = FakeRanker()
    client = _client(tmp_path, [(ranker, "7", "api"), (FakeRanker(), "8", "api")])
    yield client, ranker
    client.__exit__(None, None, None)


def test_predict_returns_brand_dict_and_headers(fake):
    client, _ = fake
    r = client.post("/v1/predict", json=EXAMPLE_USER)
    assert r.status_code == 200 and r.json() == FAKE_RANKING
    assert r.headers["X-Model-Version"] == "7" and r.headers["X-Payout-Backend"] == "api"
    assert r.headers["X-Degraded"] == "false" and r.headers["X-Cache-Hit"] == "false"
    assert len(r.headers["X-Request-ID"]) == 32


def test_predict_missing_register_date_422(fake):
    client, _ = fake
    body = {k: v for k, v in EXAMPLE_USER.items() if k != "register_date"}
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 422 and "register_date" in r.text


def test_predict_bad_cellphone_422(fake):
    client, _ = fake
    assert client.post("/v1/predict", json=dict(EXAMPLE_USER, cellphone="not-a-phone")).status_code == 422
    assert client.post("/v1/predict", json={k: v for k, v in EXAMPLE_USER.items() if k != "cellphone"}
                       ).status_code == 422


def test_numeric_string_cellphone_accepted(fake):
    client, _ = fake
    assert client.post("/v1/predict", json=dict(EXAMPLE_USER, cellphone="7135550100")).status_code == 200


def test_empty_ranking_returns_empty_dict(tmp_path):
    client = _client(tmp_path, [(FakeRanker(ranking={}), "7", "api")])
    r = client.post("/v1/predict", json=EXAMPLE_USER)
    assert r.status_code == 200 and r.json() == {}


def test_cache_hit_second_call(fake):
    client, ranker = fake
    first = client.post("/v1/predict", json=EXAMPLE_USER)
    second = client.post("/v1/predict", json=EXAMPLE_USER)
    assert first.headers["X-Cache-Hit"] == "false" and second.headers["X-Cache-Hit"] == "true"
    assert second.json() == first.json() and ranker.calls == 1


def test_degraded_answer_not_cached(tmp_path):
    ranker = FakeRanker(degraded=True)
    client = _client(tmp_path, [(ranker, "7", "api")])
    r1 = client.post("/v1/predict", json=EXAMPLE_USER)
    client.post("/v1/predict", json=EXAMPLE_USER)
    assert r1.headers["X-Degraded"] == "true" and ranker.calls == 2


def test_reload_swaps_version(fake):
    client, _ = fake
    assert client.get("/version").json()["model_version"] == "7"
    assert client.post("/admin/reload").json()["model_version"] == "8"
    assert client.get("/version").json()["model_version"] == "8"
    assert client.post("/v1/predict", json=EXAMPLE_USER).headers["X-Model-Version"] == "8"


def test_reload_requires_admin_token_when_set(tmp_path):
    client = _client(tmp_path, [(FakeRanker(), "7", "api"), (FakeRanker(), "8", "api")], admin_token="s3cret")
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_ready_and_health(fake):
    client, _ = fake
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["model_version"] == "7" and "model_age_hours" in body


def test_metrics_exposed(fake):
    client, _ = fake
    client.post("/v1/predict", json=EXAMPLE_USER)
    text = client.get("/metrics").text
    for name in ("bl_request_latency_seconds", "bl_requests_total", "bl_payout_fallbacks_total",
                 "bl_cache_hits_total", "bl_model_info"):
        assert name in text


def test_not_ready_returns_503(tmp_path):
    def slow_loader():
        time.sleep(5)
        return FakeRanker(), "7", "api"

    app = create_app(Settings(_env_file=None, warehouse_dir=tmp_path), loader=slow_loader)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/predict", json=EXAMPLE_USER).status_code == 503


@pytest.mark.slow
def test_predict_unseen_categories_still_ranks(trained_synthetic_dir, tmp_path):
    from bl_ranking.models.pyfunc import BLRankerModel

    ranker = BLRankerModel.from_dir(trained_synthetic_dir, payout_backend="fallback")
    client = _client(tmp_path, [(ranker, "1", "fallback")])
    body = dict(EXAMPLE_USER, auto_city="Atlantis", page="top10us.com/app/never-seen", sub1=1121993,
                industry="space_mining", credit_score=None)
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 200 and len(r.json()) >= 1
    assert all({"rank", "expected_payout"} == set(v) for v in r.json().values())


def test_empty_reason_header(tmp_path):
    class SparseRanker(FakeRanker):
        def rank(self, user):
            return RankResult(ranking={}, degraded=False, reason="insufficient_survey_answers")

    client = _client(tmp_path, [(SparseRanker(), "7", "api")])
    r = client.post("/v1/predict", json=EXAMPLE_USER)
    assert r.status_code == 200 and r.json() == {} and r.headers["X-Empty-Reason"] == "insufficient_survey_answers"


@pytest.mark.parametrize("field,value", [("page", None), ("page", ""), ("session_dt", "garbage"),
                                         ("register_date", "not-a-date")])
def test_client_errors_are_422_not_500(fake, field, value):
    """Final review: null page / unparseable timestamps reached CatBoost and returned 500."""
    client, ranker = fake
    r = client.post("/v1/predict", json=dict(EXAMPLE_USER, **{field: value}))
    assert r.status_code == 422 and field in r.text and ranker.calls == 0
