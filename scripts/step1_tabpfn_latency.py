"""Step 1: TabPFN serving-latency options on the REAL production context (10 query rows = 1 user x ~10 brands).

  A  original style : fit(context) + predict per request          (what bl_exp_payout_predictor does)
  B  fit once       : predict only, estimator kept in memory
  C  persisted fit  : save_model() -> load_model() -> predict      (survives restarts, no re-upload)
  D  local OSS      : tabpfn package, fit_with_cache, CPU / MPS

Each API predict costs ~10k credits (free tier 20M/month), so repetitions are capped.
Usage: .venv/bin/python scripts/step1_tabpfn_latency.py [--api-reps 5] [--skip-local]
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import joblib
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
CTX = ROOT / "artifacts" / "step1_prod" / "payout_tfm_context.joblib"


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "p50_ms": round(statistics.median(xs) * 1000),
            "max_ms": round(xs[-1] * 1000), "min_ms": round(xs[0] * 1000)}


def bench(fn, reps):
    out = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        out.append(time.perf_counter() - s)
    return out


def api_bench(args, x, y, q, res):
    import tabpfn_client as tc
    tc.set_access_token(os.environ["TABPFN_TOKEN"])
    res["usage_before"] = tc.get_api_usage()

    def per_request():
        m = tc.TabPFNRegressor(ignore_pretraining_limits=True)
        m.fit(x, y)
        m.predict(q)
    res["A_api_fit_plus_predict_per_request"] = stats(bench(per_request, args.api_reps))

    s = time.perf_counter()
    m = tc.TabPFNRegressor(ignore_pretraining_limits=True)
    m.fit(x, y)
    res["api_single_fit_s"] = round(time.perf_counter() - s, 2)
    res["B_api_predict_only"] = stats(bench(lambda: m.predict(q), args.api_reps))

    rec = ROOT / "artifacts" / "step1_prod" / "tabpfn_api_model.json"
    m.save_model(rec)
    s = time.perf_counter()
    m2 = tc.TabPFNRegressor.load_model(rec)
    res["C_load_model_s"] = round(time.perf_counter() - s, 3)
    res["C_api_predict_after_load"] = stats(bench(lambda: m2.predict(q), max(2, args.api_reps // 2)))
    res["persisted_record_bytes"] = rec.stat().st_size
    res["usage_after"] = tc.get_api_usage()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-reps", type=int, default=5)
    ap.add_argument("--skip-local", action="store_true")
    ap.add_argument("--skip-api", action="store_true", help="local backend only (no credits)")
    args = ap.parse_args()

    ctx = joblib.load(CTX)
    x, y = ctx["x"], ctx["y"]
    q = x.iloc[-10:]
    res = {}

    out_path = ROOT / "artifacts" / ("step1_tabpfn_latency_local.json" if args.skip_api
                                     else "step1_tabpfn_latency.json")
    if not args.skip_api:
        api_bench(args, x, y, q, res)
    if not args.skip_local:
        import torch
        from tabpfn import TabPFNRegressor as LocalReg
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for dev in devices:
            try:
                s = time.perf_counter()
                lm = LocalReg(ignore_pretraining_limits=True, fit_mode="fit_with_cache", device=dev)
                lm.fit(x, y)
                fit_s = time.perf_counter() - s
                lm.predict(q)  # warm-up
                res[f"D_local_{dev}"] = {"fit_s": round(fit_s, 2), **stats(bench(lambda: lm.predict(q), 15))}
            except Exception as e:  # licence / device problems are findings, not crashes
                res[f"D_local_{dev}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    print(json.dumps(res, indent=2))
    out_path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
