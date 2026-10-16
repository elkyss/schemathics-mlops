import logging
import sys

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def adapter_mod():
    from bl_ranking import adapters
    from bl_ranking.payout.backends import FallbackBackend, patch_module

    patch_module(sys.modules[adapters.TrainAdapter.__mro__[1].__module__], FallbackBackend())
    return adapters


def _adapter(adapter_mod, csv, out, **kw):
    return adapter_mod.TrainAdapter(str(csv.parent) + "/", csv.name, str(out) + "/", **kw)


def test_train_test_on_synthetic_with_fallback(adapter_mod, synthetic_csv, tmp_path):
    a = _adapter(adapter_mod, synthetic_csv, tmp_path, train_test=True)
    a.fit_()
    assert {"sold_overall_f1", "sold_overall_precision", "payout_mae", "payout_mape"} <= a.metrics_.keys()
    assert any(k.startswith("sold_day_") for k in a.metrics_)
    assert a.log_path_.exists() and "CB accuracy report" in a.log_path_.read_text()
    assert not (tmp_path / "CB_bl_lead.cbm").exists()  # train_test writes no artifacts
    assert len(a.x_test_all_days_) > 0


def test_production_holdout_zero_trains_all_rows(adapter_mod, synthetic_csv, tmp_path):
    a = _adapter(adapter_mod, synthetic_csv, tmp_path, train_test=False, holdout_days=0)
    a.fit_()
    assert a.row_counts_["train"] == a.row_counts_["after_survey_filter"]
    for name in ("CB_bl_lead.cbm", "payout_tfm_context.joblib", "all_clients.csv"):
        assert (tmp_path / name).exists()
    assert a.model_tfm_ is not None and set(a.tfm_context_) == {"x", "y", "columns"}


def test_holdout_seven_reproduces_original_split(adapter_mod, synthetic_csv, tmp_path):
    a = _adapter(adapter_mod, synthetic_csv, tmp_path, train_test=False, holdout_days=7)
    a.bl_preprocessing()
    original_cls = adapter_mod.TrainAdapter.__mro__[1]
    b = original_cls(str(synthetic_csv.parent) + "/", synthetic_csv.name, str(tmp_path / "o") + "/", train_test=False)
    x_train_orig, *_ = b.bl_preprocessing()
    assert a.row_counts_["train"] == len(x_train_orig)
    assert a.row_counts_["train"] < a.row_counts_["after_survey_filter"]


def test_gender_lookup_cached_and_identical(adapter_mod, synthetic_csv, tmp_path):
    a = _adapter(adapter_mod, synthetic_csv, tmp_path, train_test=True)
    original = adapter_mod.TrainAdapter.__mro__[1].detect_gender_with_confidence
    assert a.detect_gender_with_confidence("Maria") == original(a, "Maria")
    assert "Maria" in a._gender_cache


def test_logger_keeps_existing_root_handlers(adapter_mod, synthetic_csv, tmp_path):
    marker = logging.StreamHandler()
    logging.getLogger().addHandler(marker)
    try:
        _adapter(adapter_mod, synthetic_csv, tmp_path, train_test=True)
        assert marker in logging.getLogger().handlers
    finally:
        logging.getLogger().removeHandler(marker)


def test_relative_output_path_finds_log(adapter_mod, synthetic_csv, tmp_path, monkeypatch):
    """The CLI passes relative paths (warehouse/runs/...); FileHandler stores absolute ones."""
    monkeypatch.chdir(tmp_path)
    a = _adapter(adapter_mod, synthetic_csv, "relative_out", train_test=True)
    assert a.log_path_.exists()
