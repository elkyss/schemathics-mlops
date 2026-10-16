from datetime import date

import pandas as pd

from bl_ranking.training.gate import evaluate_gate, score_on_holdout, should_run_gate

CHAMP = {"sold_overall_f1": 0.47, "payout_mae": 2.3}


def test_gate_skipped_when_champion_saw_holdout():
    assert not should_run_gate(date(2026, 2, 11), date(2026, 2, 11))


def test_gate_runs_on_new_week():
    assert should_run_gate(date(2026, 2, 11), date(2026, 2, 4))


def test_gate_needs_a_champion():
    assert not should_run_gate(date(2026, 2, 11), None)


def test_gate_fails_on_f1_drop():
    r = evaluate_gate({"sold_overall_f1": 0.40, "payout_mae": 2.3}, CHAMP, 0.05)
    assert r.decision == "fail" and "f1" in r.reason


def test_gate_fails_on_mae_increase():
    r = evaluate_gate({"sold_overall_f1": 0.47, "payout_mae": 2.6}, CHAMP, 0.05)
    assert r.decision == "fail" and "mae" in r.reason


def test_gate_passes_within_tolerance():
    assert evaluate_gate({"sold_overall_f1": 0.46, "payout_mae": 2.35}, CHAMP, 0.05).decision == "pass"


def test_score_on_holdout_keys():
    class CB:
        def predict(self, x):
            return (x["f"] > 1).astype(int).to_numpy()

    class Pay:
        def predict(self, x):
            return x["f"].to_numpy() * 10.0

    x = pd.DataFrame({"f": [1, 2, 3], "split_day": [1, 1, 2]})
    y = pd.DataFrame({"sold_to_client": [0, 1, 1], "payout": [0.0, 20.0, 31.0], "split_day": [1, 1, 2]})
    m = score_on_holdout(CB(), Pay(), x, y)
    assert m["sold_overall_f1"] == 1.0 and m["payout_mae"] == 0.5 and "sold_day_2_f1" in m


def test_only_real_comparisons_allow_promotion():
    from bl_ranking.training.gate import GateResult

    assert GateResult("pass", "", {}, {}).allows_promotion
    assert GateResult("no_champion", "", {}, None).allows_promotion
    assert not GateResult("skipped_no_new_data", "", {}, None).allows_promotion
    assert not GateResult("fail", "", {}, {}).allows_promotion
