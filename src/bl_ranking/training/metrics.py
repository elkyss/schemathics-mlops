"""The researcher's accuracy report (classification per day, payout MAPE/MAE per threshold) as MLflow metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, precision_recall_fscore_support

PAYOUT_THRESHOLDS = (5, 10, 15, 20)


def classification_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    return {f"{prefix}_f1": float(f[0]), f"{prefix}_precision": float(p[0]), f"{prefix}_recall": float(r[0])}


def payout_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    out = {"payout_mape": float(mean_absolute_percentage_error(y_true, y_pred)),
           "payout_mae": float(mean_absolute_error(y_true, y_pred))}
    for t in PAYOUT_THRESHOLDS:
        mask = y_pred >= t
        if mask.any():  # the original raises on an empty slice; we record nothing instead
            out[f"payout_mape_thr_{t}"] = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))
            out[f"payout_mae_thr_{t}"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return out
