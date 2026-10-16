"""Step 1 baseline: run the researcher's ORIGINAL training code unchanged and measure it.

Wraps class methods at runtime with timers (the source file is not edited), so we get
per-stage wall time, peak RSS and TabPFN API credit usage for one train_test run.

Usage: .venv/bin/python scripts/step1_run_original.py [--production]
"""
import argparse
import functools
import json
import os
import resource
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src" / "bl_ranking" / "original"))

t0 = time.perf_counter()
import bl_models_train as orig  # noqa: E402  (module-level NameDataset() load happens here)

import_s = time.perf_counter() - t0

import tabpfn_client  # noqa: E402

TIMINGS: dict[str, float] = {"import_module_incl_NameDataset": import_s}


def timed(cls, name):
    fn = getattr(cls, name)

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        s = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            TIMINGS[name] = TIMINGS.get(name, 0.0) + time.perf_counter() - s

    setattr(cls, name, wrapper)


for m in ["import_preprocess", "impute_survey_columns", "credit_score_to_numeric",
          "credit_loan_amount_to_numeric", "monthly_revenue_to_numeric", "time_in_business_to_num",
          "time_features", "additional_features", "split_by_time",
          "catbosot_model_sold_to_client", "tabpfn_regression_payout",
          "accuracy_classification_model", "accuracy_cont_payout_prediction", "save_models"]:
    timed(orig.BLPayoutModelsFit, m)


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3  # macOS: bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", action="store_true")
    args = ap.parse_args()
    out = ROOT / "artifacts" / ("step1_prod" if args.production else "step1_train_test")
    out.mkdir(parents=True, exist_ok=True)

    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    usage_before = tabpfn_client.get_api_usage()
    print(f"peak RSS after import: {peak_rss_gb():.2f} GB", flush=True)

    s = time.perf_counter()
    model = orig.BLPayoutModelsFit(str(ROOT / "data") + "/", "bl_full_data.csv", str(out) + "/",
                                   train_test=not args.production)
    model.fit_()
    TIMINGS["fit_total"] = time.perf_counter() - s

    report = {
        "mode": "production" if args.production else "train_test",
        "timings_s": {k: round(v, 2) for k, v in TIMINGS.items()},
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "api_usage_before": usage_before,
        "api_usage_after": tabpfn_client.get_api_usage(),
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    print(json.dumps(report, indent=2))
    (out / "step1_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
