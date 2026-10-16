import pytest

from bl_ranking.training.metrics import classification_metrics, payout_metrics


def test_classification_metrics_class_one():
    m = classification_metrics([1, 0, 1, 0], [1, 0, 0, 0], prefix="sold_overall")
    assert m == {"sold_overall_f1": pytest.approx(2 / 3), "sold_overall_precision": 1.0, "sold_overall_recall": 0.5}


def test_classification_metrics_no_positive_predictions_is_zero_not_error():
    assert classification_metrics([1, 0], [0, 0], prefix="d")["d_precision"] == 0.0


def test_payout_metrics_overall():
    m = payout_metrics([10, 20], [12, 18])
    assert m["payout_mae"] == pytest.approx(2.0)
    assert m["payout_mape"] == pytest.approx((0.2 + 0.1) / 2)


def test_payout_metrics_skips_empty_threshold():
    m = payout_metrics([10, 12], [6, 7])
    assert "payout_mae_thr_5" in m and "payout_mae_thr_20" not in m and "payout_mape_thr_10" not in m
    assert m["payout_mae"] == pytest.approx(4.5)
