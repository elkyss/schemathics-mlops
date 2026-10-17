# BL brand ranking — productized with MLflow

The researcher's two scripts (`bl_models_train.py`, `bl_exp_payout_predictor.py`) turned into a production ML system:
a weekly, MLflow-tracked training pipeline with a gated model registry and rollback, and a low-latency API that
ranks lender brands for a user waiting on the landing page.

**The model logic is untouched.** The original scripts sit in [`src/bl_ranking/original/`](src/bl_ranking/original)
byte-identical to `DYK-Schemathics/Candidates@dev` (`ae1fce1`), checked by a test against
[`reference/SHA256SUMS`](reference/SHA256SUMS). Everything around them subclasses or wraps them; no function
body is copied or edited.

> Part 2 write-up: [`docs/part2.pdf`](docs/part2.pdf) · Latency report: [`docs/latency.md`](docs/latency.md) ·
> Design + decisions: [`docs/PLAN.md`](docs/PLAN.md)

## Engineering highlights

| | What | Evidence |
|---|---|---|
| ⚡ | **TabPFN served in the most efficient way:** the original re-fit TabPFN on every request (8.9 s). Fitted once in training and reused when serving: 0.29 s/request on local TabPFN; our own code adds ~15 ms | [`docs/measurements/step1.md`](docs/measurements/step1.md), [`docs/latency.md`](docs/latency.md) |
| 🛡️ | **The promotion gate works in practice:** a local-TabPFN retrain whose payout MAE rose 3.80 → 4.00 was refused and left at `@challenger`; the gate only compares models on a week neither has seen | `tests/test_gate.py`, `tests/test_pipeline.py` |
| 🔒 | **The researcher's code is untouched and proven equivalent:** byte-identical originals (sha256 test); a train-serve parity test shows the API returns exactly what their predictor returns | `tests/test_integrity.py`, `tests/test_predict_adapter.py` |
| 🐞 | **Bugs in the original found and handled without editing it:** production never trained on the last 7 days; a global `user_data`; a crash on ~5 % of users (14 of 300) with sparse survey answers | [Findings](#findings-in-the-original-code-handled-without-changing-it) |
| 📏 | **Measured, then corrected:** the first API latency (0.9 s) came from server-cached repeated rows; real users always send new rows → 1.5–1.8 s. Reported, and the design adjusted | [`docs/latency.md`](docs/latency.md) |
| ✅ | **Tested like a reviewer:** `make demo` from a fresh clone caught two submission-breaking bugs (a `.gitignore` rule hiding a code package; a macOS-only lockfile); a unit test caught a daylight-saving bug a week before Israel's clocks change | `tests/test_integrity.py`, `tests/test_scheduler.py` |
| 💸 | **Cost-aware:** API fits are free, so they're persisted and reused; the connection warm-up uses a free call; queued calls that time out are cancelled; load tests capped to protect the quota (each API prediction ≈ 10k of 20M monthly credits) | `payout/backends.py`, `tests/test_backends.py` |
| 🧠 | **Resource-aware:** local TabPFN measured at ~5.5 GB per process, so every `make` target picks a backend that fits the reviewer's Docker memory, and says why | `scripts/backend.sh`, `tests/test_backend_sh.py` |
| 🔐 | **Privacy by default:** data and `.env` never committed; no personal data in logs or the inference log; load-test traffic de-identified; the use of gender inferred from names is flagged as a fair-lending risk | `tests/test_sinks.py`, `tests/test_payloads.py` |

117 automated tests, ruff and a Docker build run in CI on every push.

---

## Requirements map

| Brief requirement | Where it is solved | How to see it |
|---|---|---|
| Productize both scripts, **don't change functions/logic** | `src/bl_ranking/original/` (verbatim) + `adapters.py`, `predict_adapter.py` (subclasses, plumbing only) | `pytest tests/test_integrity.py`; train-serve parity test `tests/test_predict_adapter.py::test_rank_matches_original_predict` |
| Data in a table; model reads from it | `bl load-data` → DuckDB table `bl.bl_full_data` (Delta/UC table on Databricks); each run snapshots it byte-faithfully for the unchanged code | `make load-data`; `tests/test_source.py::test_real_data_parity` |
| Training: **train/test mode** | `bl train --mode train_test` → MLflow experiment `bl-ranking/train_test` with per-day metrics | `make train-test`, MLflow UI <http://localhost:5001> |
| Training: **production mode** (all data, registers artifacts) | `bl train --mode production` → registered model `bl_ranker` (`@challenger` → gate → `@champion`) | `make train`, MLflow → Models |
| **Weekly schedule, Sunday 05:00** | `scheduler` container (`bl scheduler`, cron `0 5 * * 0`, `SCHEDULER_TZ`, default Asia/Jerusalem); Databricks Jobs cron in `databricks/databricks.yml` | `docker compose logs scheduler` → "next scheduled training at …" |
| Technical params in MLflow, **runs comparable** | Same params/metrics on every run: CatBoost settings, TabPFN context size + backend, data range/rows/snapshot sha256, row counts per filter, library versions, git + upstream commit | MLflow → compare runs |
| **Researcher log as artifact** | `researcher_log/log_file_bl_train_*.log` on every run | MLflow run → Artifacts |
| **Versioned artifacts, identify serving version, rollback** | MLflow Model Registry versions + aliases; model card per version; `/version` endpoint; `X-Model-Version` header; `bl rollback` + API hot reload (no restart) | `make rollback`, `curl localhost:8000/version` |
| **TabPFN productized efficiently** | Fit once in training; the hosted fit's id is persisted with the model (762-byte record) and reused by serving — no re-upload, no per-request refit; deadline + fallback; optional local backend | `docs/measurements/step1.md`, `docs/latency.md` |
| **Endpoint**: dict in → ranked brand dict out | `POST /v1/predict` — body is exactly the original output; metadata in headers | `make predict` |
| Synchronous, user-facing → **minimize latency** | Everything loaded once; ~15 ms of our own code per request; TabPFN deadline 2 s → fallback; response cache | `docs/latency.md` |
| **Load simulation**, median + tail latency | Locust scenarios + report generator | `make load-test`, [`docs/latency.md`](docs/latency.md) |
| Runnable locally (Docker OK); code on `main`; README | `docker compose` stack + `make demo` | [Quickstart](#quickstart) |
| Logs and results written to the platform | Researcher log → MLflow artifact; inference log table (`inference_log`, SQLite locally / Delta on Databricks) | `warehouse/inference.db` in the `warehouse` volume |
| Part 2: ≤ 2-page document | [`docs/part2.pdf`](docs/part2.pdf) | — |

The two versions of the brief differ: the repo README asks for Databricks tables and outputs; the Google Doc allows
any local setup. This repo runs **locally first** with Databricks-shaped interfaces (table source, MLflow/UC
registry, inference-log table, Jobs cron), and `ENV=databricks` + [`databricks/databricks.yml`](databricks/databricks.yml)
moves it to Databricks. The bundle is written and YAML-validated but **not deployed** (no workspace was available).

---

## Quickstart

**Needs:** Docker Desktop with **≥ 8 GB memory** (Settings → Resources), `make`, ~3 GB disk for the image.
The default payout model, local TabPFN, holds ~5.5 GB per process, so running it in Docker (API + trainer
side by side) needs **~20 GB**. `make demo` checks Docker's memory and, below that, uses the hosted API
(with a token) or the fallback, saying so — and so does every other `make` target (`scripts/backend.sh`).
To choose yourself, set `TABPFN_BACKEND=api|fallback|local` in `.env`.

```bash
git clone https://github.com/elkyss/schemathics-mlops && cd schemathics-mlops
```

1. **Data** (not in git — it contains personal data): download `bl_full_data.csv` from the
   [assignment Drive folder](https://drive.google.com/drive/folders/1pX2IxzcJEV4Fl520ZkeyT935bNhlotY3) into `data/`.
2. **Optional TabPFN token:** `cp .env.example .env` and set `TABPFN_TOKEN` (free Prior Labs account; accept the
   TabPFN-3.5 licence on ux.priorlabs.ai). The default payout model is **local TabPFN on CPU**: the token is used
   once to download its weights and no API credits are spent. `TABPFN_BACKEND=api` switches to the hosted model
   (`make demo` then spends ~8 hosted predictions). Without a token the payout model is the fallback (mean payout
   per brand) and everything else works the same.
3. Run the whole story:

```bash
make demo
```

It starts MLflow (<http://localhost:5001>), the API (<http://localhost:8000>) and the scheduler, loads the data,
runs the researcher's train/test evaluation, trains a production model on data up to 2026-02-04 (v1 → champion),
predicts, retrains on all data (the gate compares v2 with v1 on the week v1 never saw → v2 promoted), predicts
again, and rolls back to v1 **without restarting the API**. First run: ~3 min image build + ~6 min of training.

Individual steps:

| Command | What it does |
|---|---|
| `make up` / `make down` | start / stop the stack (`docker compose down -v` also wipes MLflow + warehouse volumes) |
| `make load-data` | CSV → training table |
| `make train-test` | researcher evaluation → MLflow metrics + log |
| `make train [AS_OF=YYYY-MM-DD]` | production run → register → gate → promote → API hot reload |
| `make predict` | rank brands for [`load_test/example.json`](load_test/example.json) (synthetic person) |
| `make rollback` / `make promote` / `make status` | model lifecycle |
| `make load-test` | 60 s realistic Locust run against the API, recorded in `docs/measurements/` |
| `make test` / `make lint` | tests (no real data, no API credits) / ruff, inside the image |

---

## Architecture

```
            weekly cron (Sun 05:00)              data/bl_full_data.csv
                    │                                   │ bl load-data
                    ▼                                   ▼
 ┌──────────────── bl train ──────────────────┐   bl.bl_full_data (DuckDB · Delta on Databricks)
 │ snapshot table → validate                  │◄──────────────┘
 │ [gate: challenger vs champion, same week]  │
 │ original training code (wrapped)           │──► MLflow tracking: params · metrics · researcher log
 │ package pyfunc bl_ranker                   │──► MLflow registry: bl_ranker vN @challenger
 │ promote if gate allows → POST /admin/reload│──► @champion (+ previous_champion tag for rollback)
 └────────────────────────────────────────────┘
                                                        │ models:/bl_ranker@champion (loaded once)
 landing page ── POST /v1/predict ──► API (FastAPI) ────┘
                 ◄── ranked brand dict ── original preprocessing → CatBoost P(lead) × TabPFN payout → rank
                                          │ deadline 2 s → fallback payout · response cache · /metrics
                                          └─► inference_log (no personal data)
```

| Concern | Local (this repo) | Databricks (`ENV=databricks`) |
|---|---|---|
| Data table | DuckDB `bl.bl_full_data`, all columns text | UC Delta `main.bl.bl_full_data` (`SparkSource`) |
| Tracking | MLflow server in compose (SQLite + proxied artifacts) | Databricks MLflow, `/Shared/bl-ranking/*` |
| Registry | MLflow registry `bl_ranker` + aliases | Unity Catalog `main.bl.bl_ranker` + aliases |
| Schedule | `scheduler` container | Jobs `quartz_cron_expression: "0 0 5 ? * SUN"` |
| Serving | FastAPI container | Model Serving endpoint on the same pyfunc |
| Results | `inference_log` (SQLite WAL) | Delta table |

Code map: `src/bl_ranking/` — `config.py` (all settings, env-overridable) · `data/` (table source, snapshot,
validation, inference log) · `adapters.py` / `predict_adapter.py` (wrappers around the originals) ·
`payout/backends.py` (TabPFN api/local/fallback, deadline) · `models/` (pyfunc, registry) · `training/`
(pipeline, gate, tracking, metrics) · `serving/` (API, schema, cache, telemetry) · `scheduler.py` · `cli.py` (`bl`).

---

## Training and MLflow

- **Every run** snapshots the table (sha256 logged), validates it (required columns, row count, date range,
  registered share, survey null rates — a bad extract fails the run with the reason) and logs comparable params,
  metrics and the researcher's log.
- **train_test** reproduces the researcher's evaluation exactly (real data: F1(lead) 0.474, payout MAPE 5.4 %,
  MAE $2.34) and turns the text report into metrics: `sold_overall_*`, `sold_day_{1..7}_*`,
  `payout_mape|mae[_thr_{5,10,15,20}]`.
- **production** trains on **all** rows. The original split always holds out the last 7 days, even in
  production; `HOLDOUT_DAYS_PROD=7` restores that behaviour.
- **Gate:** before promoting, the pipeline retrains a train/test challenger and scores the current champion on the
  same newest week. Promotion is refused if F1 drops or payout MAE rises by more than `GATE_MAX_REGRESSION`
  (5 %). The comparison is only fair if the champion never saw that week, so the gate runs only when the snapshot
  is ≥ 7 days newer than the champion's data. Otherwise `gate=skipped_no_new_data` and the new version **waits at
  `@challenger`** (`make promote` after review). An identical snapshot (the table wasn't refreshed) registers
  nothing (`gate=skipped_unchanged_data`). The weekly job assumes an upstream ingestion refreshes the table
  (`bl load-data` locally, the `bl_load_data` job / a Delta ingestion on Databricks).
- **Registry:** each production run registers `bl_ranker` vN with tags (`payout_backend`, `data_end_date`,
  `gate`, `snapshot_sha256`, `git_sha`, `upstream_commit`, `run_id`) and a markdown **model card**.
  `AUTO_PROMOTE=false` leaves new versions at `@challenger` for human approval (`make promote`).
  A fallback-trained version never replaces a TabPFN champion without `--force --reason`.
  The champion is scored with **its own** payout backend: an api-trained champion makes the gate spend API
  credits, and a local challenger is likely refused against it (payout MAPE 8.9 % vs 5.4 %) — retrain the
  champion on the new backend, or promote deliberately, when switching backends.
- **Reliability:** one run at a time (`warehouse/train.lock`), non-zero exit + MLflow FAILED on errors,
  `warehouse/last_success.json`, `bl status` shows champion age.

## Serving

`POST /v1/predict` with the post-funnel dict (see [`load_test/example.json`](load_test/example.json)) returns the
original output unchanged, e.g. `{"xlt": {"rank": 1.0, "expected_payout": 198.5}, "businessloans.com": {...}}`.

| Header | Meaning |
|---|---|
| `X-Model-Version` | registry version that answered |
| `X-Payout-Backend` | `api` / `local` / `fallback` |
| `X-Degraded` | `true` when TabPFN missed the deadline and the fallback payout answered |
| `X-Cache-Hit` | served from the 10-minute response cache (repeats, refreshes, double submits) |
| `X-Empty-Reason` | why the ranking is `{}` (e.g. `insufficient_survey_answers`) |
| `X-Request-ID` | correlates with the JSON logs and the inference log |

Other endpoints: `GET /health`, `GET /ready` (503 until the model is loaded, with the reason), `GET /version`,
`POST /admin/reload` (atomic model swap; `X-Admin-Token` if `ADMIN_TOKEN` is set), `GET /metrics` (Prometheus:
latency histogram, requests by status, fallbacks, cache hits, model info).

**Payout backends** (`TABPFN_BACKEND`, default `local`; without a token `local`/`api` fall back automatically;
`SERVE_PAYOUT_BACKEND` switches at serve time without retraining, e.g. during an API outage):

| Backend | Latency (new rows) | Payout MAPE | Notes |
|---|---|---|---|
| `api` (the researcher's original choice) | 1.5–1.8 s | 5.4 % | hosted; ~10k credits per prediction: free tier ≈ 2,000/month vs ~28,000/month needed at ~940 registered sessions/day; data leaves the company |
| `local` (**default**) | ~0.21 s | 8.9 % | OSS TabPFN-3.5 on CPU; no per-call cost, data stays in-house; ~5.5 GB RAM per process (context cache); weights are non-commercial → production needs Prior Labs' commercial licence |
| `fallback` | ~0.015 s | — | mean payout per brand; safety net |

Load results ([`docs/latency.md`](docs/latency.md)): local (default) p50 0.28 s / p99 0.76 s at 5 concurrent
users (over the 0.4 s target under concurrency: scale out replicas); api p50 1.7 s with **25 % of requests answered by the fallback**
at the 2 s deadline (p99 2.1 s from a quota-capped n=40 run); fallback p99 84 ms; the original predictor took 5–9 s per request and crashes on ~5 % of users (14 of 300 sampled).

---

## Findings in the original code (handled without changing it)

1. Production mode never trains on the last 7 days (the split always runs) → TabPFN context is a week stale.
   Fixed by passing `days_for_test=0` in production (flag restores the original).
2. The predictor reads a **global** `user_data`, not `self.user_data` → set under a lock per request.
3. Per request it re-reads artifacts, opens a new log file and re-fits TabPFN through the API (8.9 s) → loaded once.
4. Users with < 5 of 8 survey answers (~5 % of traffic) are dropped and then **crash** `additional_features` →
   the service returns `{}` with `X-Empty-Reason: insufficient_survey_answers`.
5. `setup_bl_logger` clears all root-logger handlers → preserved by the wrapper.
6. The example request sends `sub1` as an int; the data has zero-padded strings (`"01121993"`), so the category
   never matches — passed through unchanged, flagged for the researcher.
7. The TabPFN context (last 1,000 paid rows) covers only 4 of 10 brands; payouts for the rest are extrapolated.
8. The model uses gender inferred from first names in a lending product — a fair-lending risk to review with the
   research team (logic untouched).

## Development without Docker

```bash
pyenv install 3.12.13 && ~/.pyenv/versions/3.12.13/bin/python -m venv .venv
.venv/bin/pip install -r requirements.in -e .     # requirements.txt is locked for the Linux image (make lock)
.venv/bin/pytest -m "not slow"                    # fast tests; -m "slow" loads NameDataset (~2.2 GB)
```

Test markers: `slow` (imports the original modules), `data` (needs the real CSV), `api` (spends TabPFN credits).
CI (GitHub Actions) runs lint, `pytest -m "not data and not api"` and a Docker build on every push.
