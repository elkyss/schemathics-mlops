"""Baseline: the researcher's predictor exactly as given, one call per request.

Each call constructs BLPayoutModelsPredict (new log file), loads the artifacts and fits TabPFN again — what the
original serving path would do. Runs with the local or fallback backend so it spends no API credits; the API's
per-request fit+predict cost is in docs/measurements/step1.md (p50 8.9 s).

Usage: python scripts/bench_original_predictor.py --predictors DIR [--backend local|fallback] [--n 20]
"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictors", required=True, type=Path)
    ap.add_argument("--backend", default="local", choices=["local", "fallback"])
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    from bl_ranking.original import bl_exp_payout_predictor as orig
    from bl_ranking.payout.backends import FallbackBackend, LocalBackend, patch_module
    from load_test.payloads import build_payloads

    patch_module(sys.modules[orig.__name__], LocalBackend() if args.backend == "local" else FallbackBackend())
    payloads = build_payloads(ROOT / "data" / "bl_full_data.csv", n=args.n, repeat_rate=0.0, seed=7)
    times, crashes = [], 0
    with tempfile.TemporaryDirectory() as logs:
        for user in payloads:
            start = time.perf_counter()
            orig.user_data = user  # the original reads the module-global (bl_exp_payout_predictor.py:58)
            try:
                orig.BLPayoutModelsPredict(str(args.predictors) + "/", user, logs).predict_()
            except ValueError:  # <5 survey answers: the original crashes on the empty frame
                crashes += 1
                continue
            times.append(time.perf_counter() - start)
    times.sort()
    p99 = times[int(0.99 * (len(times) - 1))]
    result = {"name": f"original predictor per request ({args.backend} TabPFN)", "n": len(times),
              "crashes": crashes, "p50_ms": round(statistics.median(times) * 1000), "p99_ms": round(p99 * 1000),
              "max_ms": round(times[-1] * 1000)}
    out = ROOT / "docs" / "measurements" / f"baseline_original_{args.backend}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
