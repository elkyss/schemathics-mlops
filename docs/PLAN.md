# Schemathics MLOps assignment — BL brand ranking, productized with MLflow (v5, final — after review rounds 1-4)

## Context
Home assignment from Schemathics (Shaked Bar). Received 2026-10-16 07:00 UTC; **48h deadline ≈ 2026-10-18 07:00 UTC**.
Productize `bl_models_train.py` + `bl_exp_payout_predictor.py` (Candidates@dev, commit `ae1fce1`) with MLflow:
- training (both modes, weekly Sunday 05:00, comparable params, researcher log as an artifact, versioning + rollback)
- a synchronous low-latency endpoint
- a load simulation (p50/tail)
- a README with full docs
- a ≤2-page Part 2 doc

Constraints: don't change model functions/logic; code on `main`.

**Two versions of the brief exist:**
- The Google Doc says "runnable locally, Docker OK".
- The repo README additionally says: "Put [data] into the Databricks table… make model read from the table and write artifacts, logs and results to Databricks".

Decision: develop and run locally, with Databricks-shaped interfaces (data table, MLflow/UC registry, inference log table, bundle YAML). Moving to Databricks = switching to `ENV=databricks` + `databricks bundle deploy`.

User situation: no Databricks, no GPU, no paid services; Mac M1 Pro, 32 GB RAM, Docker 29; no `uv` installed (use pip/venv or install uv).

**Done so far:** `~/Projects/schemathics-mlops/` exists. The data was moved to `data/bl_full_data.csv` (128,443 lines). The 3 originals are in `reference/` (+ `UPSTREAM_COMMIT`, `SHA256SUMS`). The 2 scripts are also copied to `src/bl_ranking/original/`. No git yet.

## What the given code does (verified)
**Data:**
- 128,442 rows / 124,701 sessions, 2025-12-11 → 2026-02-11
- 65,718 rows have no register_date
- 35,206 rows have payout>0
- 10 brands + 'other' (2 noise brands from other verticals)
- `business_name` appears twice in the CSV header

**Training:** labels via `check_match` → filters → band→number maps → time/name features → `split_by_time(7)` → CatBoost(depth 8, 800 trees, F1) on 25 features including `client_name` → TabPFN on the last 1000 payout>0 rows (fit = context only) → train_test writes reports to the log; prod writes cbm + context joblib + all_clients.csv.

**Predictor:** same preprocessing → cross-join with brands (~10 rows) → **per call: load artifacts + remote TabPFN fit + remote predict** → P×payout → filter >0.01 → rank → dict.

**Issues found (documented; fixed only in wrappers):**
1. Prod mode trains without the last 7 days (the split always runs) → the context is 7 days stale.
2. Predictor reads the **global** `user_data` (pred:58).
3. A logger, a log file and a TabPFN refit are created per request.
4. `setup_bl_logger` clears the **root** handlers and replaces `warnings.showwarning`.
5. The example `sub1` is an int; training values are strings with a leading zero.
6. The empty-result schema is inconsistent.
7. Smaller items: `payout_adj` is unused, there's an odd time_in_business key, `session_day` = day of month, `cellphone.astype(int)` crashes if the phone is missing.

## Technical facts that shape the design (from review + spike)
- **Python ≥3.12 required.** The originals put newlines inside f-string braces (train:264-265, 305-314; PEP 701). Containers use **python:3.12-slim**.
- **TabPFN needs a Prior Labs account in every mode.** The spike (tabpfn 9.2.0) showed the local OSS weights download only after license acceptance, which needs `TABPFN_TOKEN` (a free account at ux.priorlabs.ai). Same token for the hosted API. → The user creates the account; the token goes in `.env` (gitignored), never in code or logs.
- **Hard-wired client.** Both files call `set_access_token(os.environ["TABPFN_TOKEN"])` and `tabpfn_client.TabPFNRegressor` at module level (train:8,288; pred:12,50).
  - Swap backends by **patching module attributes** (`orig.TabPFNRegressor = <backend>`, `orig.set_access_token = <backend auth>`). No function body is edited.
  - `tabpfn_client` stays installed because of the top-level import.
- **NameDataset** is loaded at import in both modules (train:5, pred:5): several GB of RAM and slow to start.
  - Serving imports only the predictor module and runs 1-2 workers.
  - Training and serving are separate containers.
  - The README states the Docker Desktop RAM need (~8 GB).
- **Global `user_data` vs concurrency.** Override `import_preprocess` as: take a module-level lock → set `pred_mod.user_data = self.user_data` → `super().import_preprocess()` → release.
  - The lock covers only that step (milliseconds), so there's no race.
  - No body is copied. Documented.

## Architecture
```
schemathics-mlops/
  reference/                    # upstream originals + UPSTREAM_COMMIT + SHA256SUMS (untouched)
  src/bl_ranking/
    original/                   # the 2 scripts, byte-identical (sha-checked) — the ONLY copy of the logic
    config.py                   # Settings (pydantic-settings), ENV=local|databricks, TABPFN_BACKEND=auto|api|local|fallback
    adapters.py                 # subclasses: split_by_time(days_for_test=0 in prod), logger isolation,
                                #   locked user_data, load-once models, metric capture via super() calls
    tabpfn_backends.py          # ApiBackend (tabpfn_client, fit once) | LocalBackend (OSS, fit_with_cache) | BrandMeanFallback
    data/source.py              # DataSource: local=DuckDB table `bl.bl_full_data` | databricks=spark.table(uc_name)
                                #   → writes a temp CSV snapshot → passes its path to the unchanged constructor; logs its sha
    data/sinks.py               # InferenceLog: local SQLite (WAL) table | Databricks Delta table (request digest, ranking, version, latency)
    training/pipeline.py        # TrainingPipeline (orchestrates adapter + tracking + registry)
    training/tracking.py        # ExperimentTracker (MLflow params/metrics/artifacts/tags)
    models/pyfunc.py            # BLRankerModel (models-from-code / code_paths; permissive input schema)
    models/registry.py          # ModelRegistryManager: register, alias champion/challenger, promote, rollback
    serving/app.py              # FastAPI /predict /health /ready /version
    cli.py                      # bl load-data | train --mode | promote | rollback | serve
  load_test/locustfile.py
  scheduler/crontab             # supercronic: 0 5 * * 0 bl train --mode production
  databricks/databricks.yml     # Asset Bundle: load-data job, weekly train job (quartz "0 0 5 ? * SUN"), serving endpoint
  docker-compose.yml            # mlflow (sqlite + artifact volume), trainer, scheduler, api
  tests/   docs/   data/ (gitignored)   pyproject.toml   Makefile   .env.example   README.md
```

### Local ↔ Databricks mapping (the migration story)
| Concern | Local (dev) | Databricks (`ENV=databricks`) |
|---|---|---|
| Data table | DuckDB `bl.bl_full_data` loaded by `bl load-data` | Delta/UC table, same `bl load-data` job |
| Tracking | MLflow server in compose | `MLFLOW_TRACKING_URI=databricks` |
| Registry | MLflow registry `bl_ranker` + aliases | UC model `main.bl.bl_ranker` + same aliases API |
| Schedule | supercronic container | Jobs cron in `databricks.yml` |
| Serving | FastAPI container | Model Serving endpoint on the same pyfunc (FastAPI optional) |
| Results | DuckDB `bl.inference_log` | Delta `bl.inference_log` |

## TabPFN strategy
- One `PayoutBackend` interface, chosen with `TABPFN_BACKEND`:
  - `auto`: `api` if a token is set and the API is reachable, else `fallback`
  - `api`: tabpfn_client, as the researcher uses it
  - `local`: OSS weights, downloaded **once** with the token at image build time, then offline
  - `fallback`: CatBoost P(lead) × the brand's historical mean payout; marks the response `degraded`
- **Efficiency wins regardless of backend:**
  - fit **once** at startup (not per request)
  - one batched predict for all ~10 brands
  - a timeout budget → fallback
  - the context is built in training and versioned with the model
- `api` vs `local` will be decided by **measured** numbers in the load test. API pros: Plus models, no local RAM/CPU, licensed for commercial use; cons: a network hop, quotas. Local pros: no network; cons: CPU latency, non-commercial weights.
- Output differs between backends → parity tests always compare the same backend on both sides.

## MLflow usage
- **Experiments:** `bl-ranking/train-test` and `bl-ranking/production`. Tags: git sha, upstream commit, mode, trigger, backend.
- **Params:** CatBoost hyperparameters, context size, backend, data range, row counts after each filter, label rate, brand count, library versions.
- **Data lineage:** the source table name + snapshot sha256 + schema. **No PII** gets logged (no `from_pandas` on raw data).
- **Metrics (train_test):** overall + `*_day_{1..7}` F1/precision/recall; payout MAE/MAPE overall and `*_thr_{5..20}`. Computed by wrapping the original methods (call `super()`, compute on the same inputs; payout predictions are read from `y_test_payout['pred_payout']` after `super()`, so there are no extra API calls).
- **Artifacts:** the researcher log (required) + the 3 model files.
- **Model:** pyfunc `BLRankerModel` (code_paths/models-from-code) with a signature, a **synthetic** input_example (no real name/phone) and pinned env (py3.12).
- **Registry:**
  - every prod run registers a version → `challenger`
  - gate passes (see below) + smoke predict OK → `champion` (auto when `AUTO_PROMOTE=true`)
  - `bl rollback` → previous version
  - `/version` shows what's serving

## Serving
FastAPI.
- **At startup:** load `@champion` → warm NameDataset → backend fit once → warm-up predict.
- **Per request:** validate (pydantic, coerce sub1/campaign_id to the training types and document it) → locked preprocess → CatBoost + payout backend → rank → write to the InferenceLog asynchronously (off the request path). **1 uvicorn worker per container** (NameDataset RAM; scale with replicas).
- Logs contain no PII (only a request hash).

### Caching (serving)
- **Response cache:**
  - key = sha256(canonical request JSON) + model version (a new `@champion` invalidates it automatically)
  - in-process TTL+LRU (10 min, bounded size)
  - catches retries, refreshes and double-submits; real hit rate is low because timestamps and names make requests unique
  - `cache_hit` is returned and logged to `inference_log` to measure the real hit rate
  - production with replicas → Redis (doc only)
  - no PII: the key is a hash, the value is rankings
- **Gender lookup cache:** `lru_cache` on `detect_gender_with_confidence` by first name (wraps `super()`, identical output) in both training and serving.
- **Load-once:** models, TabPFN context, NameDataset, brand list.
- **Load test** uses a realistic ~5% repeat-request mix and also reports cache-off numbers.

### Storage
| Data | Local | Databricks |
|---|---|---|
| Training data | DuckDB `warehouse/bl.duckdb` (all_varchar) | Delta/UC table |
| MLflow run data + registry | SQLite `mlflow.db` | managed |
| MLflow artifacts | `mlartifacts/` folder | managed (UC Volumes) |
| Inference log | SQLite in WAL mode `warehouse/inference.db` (concurrent readers + one writer across processes, so `bl monitor` can read while the API writes), batched background writes | Delta table |
Postgres/MinIO = the production swap (one URI change); described in the doc, below the cut line.

### Training efficiency
- Measure first (step 1), then cache the gender lookups and load NameDataset once.
- CatBoost uses all cores (settings unchanged); TabPFN is fit once per run.
- The data snapshot hash is logged; an optional "skip if data unchanged" check.
- Stage outputs are logged to MLflow so a failed register step doesn't need retraining.

## Production-readiness additions (review round 3, commercial lens)
**BUILD (before the cut):**
1. **Metric-gated promotion.**
   - The weekly job trains a train_test challenger on the newest snapshot (last 7 days held out); the **current champion is scored on the same holdout week**.
   - **Leak guard:** this comparison is only fair if the champion never saw that week. Run the gate only when `snapshot.max_date - 7d >= champion.data_end_date` (tagged on every version). Otherwise → `gate=skipped_no_new_data` tag + promotion is allowed on the smoke test only. With the static assignment dataset the gate is skipped; `--as-of <date>` simulates a "last week" run to demo it.
   - Gate: refuse if F1(sold_to_client) or payout MAE regresses more than a configurable % → the prod model is registered as `challenger` and only promoted when the gate passes.
   - `AUTO_PROMOTE=true` (default) or `false` (human approval); `bl promote --force --reason` writes an audit tag.
   - A fallback-trained version never replaces a TabPFN-trained champion. **With no token at all** (no champion exists) it may become champion, tagged `payout_backend=fallback` → the no-token reviewer path still works end to end.
2. **Hot reload.** `POST /admin/reload` loads `@champion` into a new object and swaps it atomically; `bl promote`/`bl rollback` call it → rollback takes effect in seconds without a restart. (Polling the alias = below the cut.)
3. **Observability.**
   - `prometheus_client` `/metrics`: latency histogram, errors, fallback count, cache hits, `model_version` gauge, model age.
   - Structured JSON logs with `request_id`.
   - *(below the cut)* `bl monitor` → drift report (PSI on survey bands/traffic source; rank-1 brand + expected-payout distributions) from inference_log vs. the training snapshot → MLflow artifact.
4. **SLO + deadline.**
   - p99 < 300 ms for the full request (to be revised after the step-1 measurements).
   - A per-request TabPFN deadline → fallback (`degraded=true`).
   - *(below the cut)* **Circuit breaker:** after N API failures, stay on the fallback for T seconds.
   - Tested in Locust with a slowed API backend.
5. **Scheduled-run reliability.** Non-zero exit on failure, a lockfile against overlapping runs, a `last_success` marker → `/ready` and `bl status` report model age; MLflow run status FAILED is visible.
6. **Reproducibility + CI.** Pinned lockfile (pip-compile); GitHub Actions: ruff + pytest on a small synthetic fixture (NameDataset mocked or marked slow). The adapter-vs-original parity test is labeled as the **train-serve skew test**.

**OPTIONAL (only if time remains):** shadow mode, `SHADOW_ALIAS=challenger` logs a second ranking into inference_log.

**DOC ONLY (Part 2):**
- **Feedback loop:** inference_log ⨝ later dispositions/payouts by session → realized revenue per session per model version (the real KPI) → retraining labels.
- **Rollout:** shadow → session-hash A/B with revenue/session as the primary metric plus guardrails.
- **Compliance:**
  - TabPFN API = per-request **data egress** of features to a third party → DPA/vendor review; the local or self-hosted commercial license avoids it.
  - **`gender` inferred from first name** in a business-loan marketplace = an ECOA/Reg B fair-lending risk → raise with research (logic untouched).
  - Non-commercial weights; PII at rest/retention; secret scanning.
- **Scaling:** NameDataset RAM per replica dominates the cost per pod; Redis for a shared cache; `/v1` API versioning; Postgres/MinIO backing stores; backfills (`--as-of`).
- **Vendor:** measured API latency, quotas, estimated cost per 1M predictions; production recommendation.

**Trim:** response cache = keep it simple, no polish. Locust: first compare the original per-request refit vs. ours on one backend; add a second backend if time allows. Databricks bundle stays below the cut.

## Build order (cut line marked)
0. `git init -b main`, `.gitignore` (data/, .env, mlruns/, artifacts/, *.log) and a private GitHub remote (user decides the name); never push PII (~15 min)
1. Python 3.12 env; the original runs locally with patched backends; measure RAM + latency of api/local/fallback; **check the free-tier quota/rate limits of the TabPFN API** (~1h)
2. DataSource (DuckDB `all_varchar`) + **dtype parity test** (snapshot vs `pd.read_csv` on the needed columns, train:66-70)
3. Adapters + MLflow training for both modes + registry/aliases
4. **Schedule + reliability**: `scheduler/crontab` (`0 5 * * 0`) + `supercronic -test` + launchd/cron instructions in the README; weekly job = train → gate → register → (auto)promote; failure exit codes, lockfile, `last_success`
5. pyfunc + FastAPI (sync `def` endpoint, 1 worker) with warm load + InferenceLog (SQLite WAL)
5b. `POST /admin/reload`, `/metrics`, JSON logs, per-request deadline → fallback
5c. Lockfile + GitHub Actions CI (ruff + pytest)
6. Locust: original-style per-request refit vs ours on one backend (local or fallback by default, to protect the API quota; a short capped API run) → p50/p95/p99
6b. **docker-compose (reviewer path)**: `mlflow server --serve-artifacts` (proxied `mlflow-artifacts:` so the host CLI and the containers resolve the same paths); scheduler `TZ` configurable (default `Asia/Jerusalem`, documented); `make load-data` fails clearly if the CSV is missing (README gives the Drive download step); mlflow, api, scheduler, trainer (CPU-only torch, python:3.12-slim); Makefile targets up/load-data/train/predict/load-test/rollback; API backend is the default in Docker; local weights download at runtime with the reviewer's own token, never baked into an image; Docker Desktop needs ≥8 GB RAM
6c. Minimal `databricks/databricks.yml` (load-data job + weekly train job, quartz `0 0 5 ? * SUN` with timezone) — written, not deployed (~45 min)
7. README (full runnable docs; states that the repo brief asks for Databricks, the Doc brief allows local, and the bundle is written but not deployed/validated against a workspace)
7b. ≤2-page Part 2 doc — its own 3h slot
7c. Fresh-clone acceptance test + share the repo (make it public at the end, or invite the reviewers) — 1h

— cut line —

8. Polling hot reload, circuit breaker, `bl monitor` drift report, second Locust backend, `databricks bundle validate` (if the CLI is available)
9. Broader unit tests (the sha test + dtype parity + one API contract test are already before the cut)
10. Extras (Postgres/MinIO, system metrics, parallel model calls, MPS for local TabPFN)

## Implementation rules from review round 2
- **Patch the module the methods actually run in.** Patch `sys.modules[BLPayoutModelsFit.__module__]` (and the predictor's equivalent), never a second alias; import the originals under one package path only (also inside MLflow `code_paths`).
- The lock around `super().import_preprocess()` is valid because pred:58 reads the global once and `pd.DataFrame([..])` copies it. The endpoint must be a sync `def`; the adapter overrides `setup_bl_logger`/`capture_warnings` so the root handlers survive.
- **No prediction cache.** Override `accuracy_cont_payout_prediction`: call `super()`, then read `y_test_payout['pred_payout']`, which the original writes into the passed frame (train:302) → MLflow metrics.
- **Fallback = `BrandMeanRegressor`** with the `fit/predict` interface, patched in as `TabPFNRegressor`; it fits the brand means from the same context x/y → no change to training logic. The metric wrapper records "n/a" for empty threshold slices (sklearn would raise at train:310-312). Runs are tagged `payout_backend`; `bl promote` refuses fallback-trained versions unless `--allow-fallback`.
- **NameDataset loads once per process.** Import the predictor module lazily inside `load_context`; log the model with a signature + synthetic example, no in-process validation (smoke test in a subprocess/API container).
- **`days_for_test=0` in prod is a flag** (`--holdout-days`, default 0; 7 = original behavior); verified that it puts all rows in train.
- **Mapping for "logs/results to Databricks"**: the researcher log stays a file handler with its original name → MLflow artifact; `inference_log` table → results; artifacts → MLflow/UC registry.
- Artifact dirs always end with `/` (the originals concatenate strings); `all_clients.csv` is read once at load time.

## Verification
1. `pytest`: originals' sha matches upstream; the adapter ranking equals the original `predict_` (same backend, global set); API contract test.
2. `bl load-data` → `bl train --mode train_test` → metrics in the MLflow UI; `--mode production` → gate (skipped on static data, tag visible; `--as-of` run shows the real gate) → `champion`; `curl /predict` with the example → ranked dict; `bl rollback` → `/version` changes.
3. `supercronic -test scheduler/crontab`; one manual run of the scheduled command.
4. Load test table saved to `docs/latency.md`.
4b. `/metrics` shows the latency histogram, fallback count and model version; forcing a TabPFN timeout returns `degraded=true` within the deadline; `bl rollback` → `/version` changes within seconds (no restart).
4c. A `--as-of` production run shows the gate PASS/FAIL tags in MLflow; CI is green on GitHub.
5. **Reviewer acceptance test**: fresh `git clone` into an empty dir → follow only the README → `make up/load-data/train/predict/load-test/rollback` all work.
6. `databricks bundle validate` (if the CLI is available; otherwise YAML lint only).

## Process
Round 4 (final gatekeeper): GO WITH CHANGES → all 8 applied (gate leak guard, no-token path, SQLite WAL inference log + 1 worker, --serve-artifacts, scheduler TZ, scope trimmed, minimal databricks.yml above the cut, stale lines removed + verification added).
Round 3 (commercial lens) applied: gate, hot reload, observability, SLO/circuit breaker, run reliability, CI, compliance/rollout docs.
Review round 1 (gaps + risks) and round 2 (verifying claims against the code + both briefs) have been applied. Round 2's three blockers are resolved: the schedule is moved before the cut, a dtype parity test is added, git is step 0. → Build.
