"""Promotion gate: a new model must not be worse than the champion on data neither has trained on.

The comparison is only fair if the champion never saw the holdout week. With fresh weekly data that holds
(the champion was trained a week earlier); re-running on the same data does not, so the gate is skipped then.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from bl_ranking.training.metrics import holdout_classification_metrics, payout_metrics

F1, MAE = "sold_overall_f1", "payout_mae"
Decision = Literal["pass", "fail", "skipped_no_new_data", "skipped_unchanged_data", "no_champion"]


@dataclass
class GateResult:
    decision: Decision
    reason: str
    challenger: dict
    champion: dict | None

    @property
    def allows_promotion(self) -> bool:
        """Auto-promote only after a real comparison (or when there is nothing to compare against). A skipped gate
        leaves the new version at @challenger for `bl promote`."""
        return self.decision in ("pass", "no_champion")


def should_run_gate(snapshot_max: date, champion_data_end: date | None, holdout_days: int = 7) -> bool:
    return champion_data_end is not None and snapshot_max - timedelta(days=holdout_days) >= champion_data_end


def evaluate_gate(challenger: dict, champion: dict, max_regression: float) -> GateResult:
    problems = []
    if champion[F1] - challenger[F1] > max_regression * champion[F1]:
        problems.append(f"f1 dropped {champion[F1]:.3f} → {challenger[F1]:.3f}")
    if challenger[MAE] - champion[MAE] > max_regression * champion[MAE]:
        problems.append(f"payout mae rose {champion[MAE]:.2f} → {challenger[MAE]:.2f}")
    if problems:
        return GateResult("fail", "; ".join(problems) + f" (tolerance {max_regression:.0%})", challenger, champion)
    return GateResult("pass", f"within {max_regression:.0%} of champion on f1 and payout mae", challenger, champion)


def score_on_holdout(cb_model, payout_model, x_test_all_days, y_test_all_days) -> dict:
    metrics = holdout_classification_metrics(cb_model, x_test_all_days, y_test_all_days)
    paid = (y_test_all_days["payout"] > 0).to_numpy()
    x_paid = x_test_all_days[paid].drop("split_day", axis=1)
    metrics.update(payout_metrics(y_test_all_days["payout"][paid], payout_model.predict(x_paid)))
    return metrics
