"""Step 1: payout accuracy of LOCAL TabPFN vs the API, using the original train_test code unchanged.

The original module's `TabPFNRegressor` / `set_access_token` names are swapped at runtime for the
local OSS estimator (no source edits, no API credits). Compare the payout MAPE/MAE in the
researcher log with the API baseline (MAPE 0.054, MAE 2.34 from step1_run_original.py).
"""
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src" / "bl_ranking" / "original"))

import bl_models_train as orig  # noqa: E402
from tabpfn import TabPFNRegressor as LocalTabPFNRegressor  # noqa: E402


def local_regressor(**kwargs):
    return LocalTabPFNRegressor(device="cpu", fit_mode="fit_with_cache", **kwargs)


orig.TabPFNRegressor = local_regressor
orig.set_access_token = lambda token: None

out = ROOT / "artifacts" / "step1_local_train_test"
out.mkdir(parents=True, exist_ok=True)
s = time.perf_counter()
orig.BLPayoutModelsFit(str(ROOT / "data") + "/", "bl_full_data.csv", str(out) + "/", train_test=True).fit_()
print(f"fit_total_s={time.perf_counter() - s:.1f}")
log = next(out.glob("*.log")).read_text().splitlines()
for line in log:
    if "Payout mean_absolute" in line or "thr:" in line:
        print(line.split(" - INFO - ")[-1])
