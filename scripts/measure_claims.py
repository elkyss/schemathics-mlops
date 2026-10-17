"""Reproduce the numbers quoted in README / Part 2 that are not load-test outputs.

- traffic: sessions/day and registered sessions/day (registered = a prediction is needed)
- sparse surveys: share of registered sessions with < 5 of 8 survey answers — the original predictor drops them in
  impute_survey_columns (dropna thresh=5) and then crashes in additional_features
- request overhead: our request path with the fallback payout model (validation-free rank(): original preprocessing,
  CatBoost, ranking) — i.e. everything except TabPFN

Usage: python scripts/measure_claims.py [--predictors DIR]   →   docs/measurements/claims.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SURVEY = ["credit_score", "industry", "loan_amount", "loan_reason", "monthly_revenue", "time_in_business",
          "device_type", "business_type"]  # impute_survey_columns' list (bl_exp_payout_predictor.py)


def traffic_and_sparse(csv: Path) -> dict:
    d = pd.read_csv(csv, usecols=["session_id", "session_dt", "register_date", *SURVEY], low_memory=False)
    ts = pd.to_datetime(d["session_dt"], errors="coerce")
    days = (ts.max() - ts.min()).days + 1
    reg = d[d["register_date"].notna()].drop_duplicates("session_id")
    # the original lower-cases and blanks 1-char answers ("-") before filling; dropna(thresh=5) runs before that,
    # on the raw values, so NaN is the only thing that counts as missing here
    sparse = reg[SURVEY].notna().sum(axis=1) < 5
    weeks_ago = (ts.max() - ts.loc[reg.index]).dt.days // 7
    weekly = sparse.groupby(weeks_ago).mean().round(4)
    return {"days": days, "sessions_per_day": round(d["session_id"].nunique() / days),
            "sparse_survey_share_by_weeks_ago": {int(k): float(v) for k, v in weekly.head(4).items()},
            "registered_sessions_per_day": round(len(reg) / days),
            "predictions_per_month_at_30_days": round(len(reg) / days * 30),
            "registered_sessions": len(reg), "sparse_survey_sessions": int(sparse.sum()),
            "sparse_survey_share": round(float(sparse.mean()), 4)}


def request_overhead(predictors: Path, n: int = 200) -> dict:
    from bl_ranking.models.pyfunc import BLRankerModel
    from load_test.payloads import build_payloads

    ranker = BLRankerModel.from_dir(predictors, payout_backend="fallback")
    users = build_payloads(ROOT / "data" / "bl_full_data.csv", n=n, repeat_rate=0.0, seed=11)
    times = []
    for user in users:
        start = time.perf_counter()
        ranker.rank(dict(user))
        times.append(time.perf_counter() - start)
    times.sort()
    return {"n": n, "p50_ms": round(statistics.median(times) * 1000, 1),
            "p95_ms": round(times[int(0.95 * (n - 1))] * 1000, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictors", type=Path, help="a production predictors dir (for the request overhead)")
    args = ap.parse_args()
    out = {"traffic_and_sparse_surveys": traffic_and_sparse(ROOT / "data" / "bl_full_data.csv")}
    if args.predictors:
        out["request_overhead_without_tabpfn"] = request_overhead(args.predictors)
    path = ROOT / "docs" / "measurements" / "claims.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
