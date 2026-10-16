import pytest

from bl_ranking.config import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("TABPFN_TOKEN", "TABPFN_BACKEND", "HOLDOUT_DAYS_PROD", "TABPFN_DEADLINE_S", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(var, raising=False)


def test_defaults():
    s = Settings(_env_file=None)
    assert s.tabpfn_backend == "auto"
    assert s.holdout_days_prod == 0
    assert s.tabpfn_deadline_s == 2.0
    assert s.mlflow_tracking_uri == "http://localhost:5001"
    assert s.model_name == "bl_ranker"


def test_token_is_secret(monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "abc-secret")
    s = Settings(_env_file=None)
    assert s.tabpfn_token.get_secret_value() == "abc-secret"
    assert "abc-secret" not in repr(s)


def test_env_override(monkeypatch):
    monkeypatch.setenv("TABPFN_BACKEND", "local")
    monkeypatch.setenv("HOLDOUT_DAYS_PROD", "7")
    s = Settings(_env_file=None)
    assert s.tabpfn_backend == "local" and s.holdout_days_prod == 7


def test_databricks_env_switches_uris(monkeypatch):
    monkeypatch.setenv("ENV", "databricks")
    s = Settings(_env_file=None)
    assert s.mlflow_tracking_uri == "databricks" and s.registry_uri == "databricks-uc"
    assert s.model_name == "main.bl.bl_ranker" and s.table_name == "main.bl.bl_full_data"
    assert s.experiment_prefix == "/Shared/bl-ranking"


def test_databricks_explicit_values_win(monkeypatch):
    monkeypatch.setenv("ENV", "databricks")
    monkeypatch.setenv("MODEL_NAME", "prod.bl.bl_ranker")
    assert Settings(_env_file=None).model_name == "prod.bl.bl_ranker"


def test_local_registry_follows_tracking():
    s = Settings(_env_file=None)
    assert s.registry_uri == s.mlflow_tracking_uri and s.experiment_prefix == "bl-ranking"
