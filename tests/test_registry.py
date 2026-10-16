import mlflow
import pytest
from mlflow import MlflowClient

from bl_ranking.models.registry import ModelRegistryManager, PromotionRefused


class _Dummy(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: list[str], params=None) -> list[str]:
        return model_input


def _versions(registry, backends):
    out = []
    for backend in backends:
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(name="model", python_model=_Dummy())
        out.append(registry.register(info.model_uri, tags={"payout_backend": backend}))
    return out


def test_register_sets_challenger(tmp_mlflow):
    r = ModelRegistryManager(MlflowClient(), "bl_ranker_test")
    (v1,) = _versions(r, ["api"])
    assert str(r.get("challenger").version) == v1 and r.get("champion") is None


def test_promote_then_rollback(tmp_mlflow):
    r = ModelRegistryManager(MlflowClient(), "bl_ranker_test")
    v1, v2, v3 = _versions(r, ["api", "api", "api"])
    r.promote(v1)
    r.promote(v2, reason="weekly")
    assert str(r.get("champion").version) == v2
    assert r.rollback() == v1 and str(r.get("champion").version) == v1


def test_rollback_without_previous_raises(tmp_mlflow):
    r = ModelRegistryManager(MlflowClient(), "bl_ranker_test")
    (v1,) = _versions(r, ["api"])
    r.promote(v1)
    with pytest.raises(PromotionRefused):
        r.rollback()


def test_fallback_cannot_replace_tabpfn_champion(tmp_mlflow):
    r = ModelRegistryManager(MlflowClient(), "bl_ranker_test")
    v1, v2 = _versions(r, ["api", "fallback"])
    r.promote(v1)
    with pytest.raises(PromotionRefused, match="fallback"):
        r.promote(v2)
    r.promote(v2, force=True, reason="api outage drill")
    assert str(r.get("champion").version) == v2


def test_fallback_can_be_first_champion(tmp_mlflow):
    r = ModelRegistryManager(MlflowClient(), "bl_ranker_test")
    (v1,) = _versions(r, ["fallback"])
    r.promote(v1)
    assert str(r.get("champion").version) == v1
