"""Run one headless Locust scenario against a running API and record p50/p95/p99 into a results JSON.

Example:  python -m load_test.run_scenario --name "ours, fallback, stress" --users 20 --run-time 60s
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

RESULTS = Path("docs/measurements/load_results.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--spawn-rate", type=float, default=5)
    ap.add_argument("--run-time", default="60s")
    ap.add_argument("--wait", default="0.0,0.05", help="per-user wait between requests, seconds (min,max)")
    ap.add_argument("--max-requests", type=int, default=0)
    args = ap.parse_args()

    version = httpx.get(f"{args.host}/version", timeout=10).json()
    with tempfile.TemporaryDirectory() as tmp:
        prefix, summary = Path(tmp) / "run", Path(tmp) / "summary.json"
        env = dict(os.environ, WAIT=args.wait, MAX_REQUESTS=str(args.max_requests), SUMMARY_JSON=str(summary))
        subprocess.run([sys.executable, "-m", "locust", "-f", "load_test/locustfile.py", "--headless",
                        "--host", args.host, "-u", str(args.users), "-r", str(args.spawn_rate),
                        "--run-time", args.run_time, "--csv", str(prefix), "--only-summary"],
                       env=env, check=False)
        rows = list(csv.DictReader((Path(str(prefix) + "_stats.csv")).open()))
        agg = next(r for r in rows if r["Name"] == "Aggregated")
        counts = json.loads(summary.read_text()) if summary.exists() else {}
    n = int(agg["Request Count"])
    result = {
        "name": args.name, "payout_backend": version["payout_backend"], "model_version": version["model_version"],
        "users": args.users, "wait_s": args.wait, "requests": n, "failures": int(agg["Failure Count"]),
        "rps": round(float(agg["Requests/s"]), 2),
        **{f"{p}_ms": float(agg[f"{q}%"]) for p, q in (("p50", "50"), ("p95", "95"), ("p99", "99"), ("max", "100"))},
        "degraded_share": round(counts.get("degraded", 0) / n, 3) if n else None,
        "cache_hit_share": round(counts.get("cache_hit", 0) / n, 3) if n else None,
    }
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    results = [r for r in results if r["name"] != args.name] + [result]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
