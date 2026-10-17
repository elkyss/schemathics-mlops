# BL Brand Ranking — MLOps Productization Implementation Plan

> **Update 2026-10-17 (supersedes the backend lines below):** the default payout backend is now **`local`**
> (`TABPFN_BACKEND=api` by config; `auto` = local; no token → fallback). Local TabPFN needs ~5.5 GB per process,
> so `scripts/backend.sh` (used by every `make` target) switches to api/fallback on Docker hosts under ~18 GiB.
> The SLO and measurements below are unchanged; see README and docs/latency.md for the current state.


**Goal:** Wrap the researcher's two unchanged scripts into a scheduled, MLflow-tracked training pipeline with a metric-gated model registry and a low-latency `/v1/predict` endpoint, runnable locally with Docker and deployable to Databricks by configuration.

**Architecture:**
- **Logic:** the original modules are the only copy of the model logic. Thin subclasses ("adapters") change only plumbing, and module-attribute patching swaps the TabPFN backend.
- **Training:** a `TrainingPipeline` snapshots the data table, trains (train_test for the gate, production for the model), logs to MLflow, packages a pyfunc `bl_ranker`, registers it and promotes `@champion` through a leak-guarded gate.
- **Serving:** FastAPI loads `@champion` once (TabPFN via the persisted API model id), answers with the ranked brand dict, and writes an inference log in the background.

**Tech Stack:**
- Python 3.12.13, pandas 2.3, catboost 1.2, names-dataset 3.3, tabpfn-client 0.6.1, tabpfn 9.2 (OSS, optional)
- mlflow 3.17, duckdb 1.5, fastapi 0.143 + uvicorn, prometheus-client, cachetools, typer, pydantic-settings
- locust, pytest, ruff, pip-tools; Docker Compose, supercronic, Databricks Asset Bundles

**Spec:** `docs/PLAN.md` (v6). Executed task by task with TDD; per-task rulings and deviations are summarised in the PR/commit history. Measurements: `docs/measurements/step1.md`. Briefs: `reference/README.md`, `reference/BRIEF_google_doc.md`.

## Global Constraints

- Files in `src/bl_ranking/original/` stay **byte-identical** to `reference/` (sha256 in `reference/SHA256SUMS`, upstream commit `ae1fce1`). No function body from them is copied anywhere.
- Python **3.12** (the originals use PEP 701 multi-line f-strings). Containers use `python:3.12-slim`, CPU-only torch.
- Secrets only from `.env` / environment (`TABPFN_TOKEN`). Never log the token, raw names, phones or addresses. MLflow `input_example` is synthetic.
- `data/`, `.env`, `warehouse/`, `mlruns/`, `mlartifacts/`, `artifacts/` are never committed.
- Payout backend `TABPFN_BACKEND`:
  - default `auto` = `api` if a token is set, else `fallback`
  - `local` (OSS TabPFN-3.5, CPU, `fit_mode="fit_with_cache"`) is opt-in
  - API predict costs ~10k credits/call: automated tests never call the API unless marked `@pytest.mark.api`
- Production trains on all rows (`HOLDOUT_DAYS_PROD=0`); `7` reproduces the original behaviour.
- Schedule: `0 5 * * 0`, `TZ=Asia/Jerusalem` (configurable).
- SLO: p99 ≤ 1.5 s (api), ≤ 400 ms (local/fallback). TabPFN deadline 1.2 s → fallback, `X-Degraded: true`.
- `/v1/predict` response **body** = the ranked brand dict, exactly as the original returns it, e.g. `{"fundera / nerdwallet": {"rank": 1.0, "expected_payout": 42.31}}`. Metadata goes only in the headers `X-Model-Version`, `X-Payout-Backend`, `X-Degraded`, `X-Cache-Hit`, `X-Request-ID`.
- Registry model name: `bl_ranker` (local), `main.bl.bl_ranker` (Databricks UC). Aliases: `champion`, `challenger`. Version tags: `payout_backend`, `data_end_date`, `gate`, `previous_champion`, `snapshot_sha256`, `git_sha`, `upstream_commit`.

## Review Focus

1. **Inputs the original crashes on:** a request without `register_date`, or with `cellphone` missing or non-numeric. The original raises, so the API must return **422** with a clear message, never 500. Test: `test_predict_missing_register_date_422`, `test_predict_bad_cellphone_422` (Task 7).
2. **Unseen category values** (new city, new page, int `sub1`): the request must still return a ranking, because CatBoost handles unseen categories. Test: `test_predict_unseen_categories_still_ranks` (Task 7).
3. **TabPFN slow, down or evicted:** the response must arrive within the deadline using fallback payouts with `X-Degraded: true`. An evicted API fit is refit once from the context. Tests: `test_deadline_returns_fallback`, `test_evicted_fit_refits_once` (Task 3).
4. **Concurrent requests from different users:** each request must get its own ranking, despite the original's global `user_data`. Test: `test_concurrent_users_get_own_ranking` (Task 5).
5. **Every brand under the 0.01 expected-payout cut:** the original returns `{}` (or `{"expected_payout": 0, "prob_lead": 0}` after the survey filter). The API must answer **200 with `{}`**. Test: `test_empty_ranking_returns_empty_dict` (Task 7).

---

## File Structure

```
pyproject.toml                       # package bl_ranking (src layout), ruff + pytest config, console script `bl`
requirements.in / requirements.txt   # top-level deps / pip-compile lock
src/bl_ranking/
  __init__.py
  original/__init__.py               # empty; the two scripts sit beside it, byte-identical
  config.py                          # Settings
  data/source.py                     # DataSource, DuckDBSource, SparkSource, Snapshot
  data/sinks.py                      # InferenceLog (SQLite WAL / Delta)
  payout/backends.py                 # PayoutBackend impls, BrandMeanRegressor, DeadlineRegressor, patch_module
  adapters.py                        # TrainAdapter, PredictAdapter
  training/metrics.py                # classification_metrics, payout_metrics
  training/tracking.py               # ExperimentTracker
  training/gate.py                   # GateResult, should_run_gate, evaluate_gate
  training/pipeline.py               # TrainingPipeline
  models/pyfunc.py                   # BLRankerModel, RankResult
  models/registry.py                 # ModelRegistryManager
  serving/app.py                     # create_app()
  serving/schemas.py                 # UserRequest
  serving/cache.py                   # ResponseCache
  serving/telemetry.py               # prometheus metrics + JSON logging
  cli.py                             # typer app `bl`
tests/
  conftest.py, fixtures/synthetic.py # synthetic dataset (same columns/categories, 21 days)
  test_*.py                          # per task
load_test/locustfile.py, load_test/payloads.py, load_test/report.py
scripts/bench_original_predictor.py
scheduler/crontab
docker/Dockerfile, docker-compose.yml, Makefile
databricks/databricks.yml
.github/workflows/ci.yml
README.md, docs/part2.md (→ PDF)
```

Pytest markers: `slow` (imports the original modules → NameDataset load, ~9 s and 2.2 GB), `data` (needs `data/bl_full_data.csv`), `api` (spends TabPFN credits). CI runs `-m "not data and not api"`.

---

### Task 1: Package skeleton, lockfile, settings, integrity test

**Files:**
- Create: `pyproject.toml`, `requirements.in`, `requirements.txt`, `src/bl_ranking/__init__.py`, `src/bl_ranking/original/__init__.py`, `src/bl_ranking/config.py`
- Move: `scripts/step1_*.py` keep as-is (they import via `sys.path`)
- Test: `tests/test_integrity.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Settings(BaseSettings)` with fields (env names in caps):
    - `env: Literal["local","databricks"]="local"`, `data_csv: Path=Path("data/bl_full_data.csv")`, `warehouse_dir: Path=Path("warehouse")`, `table_name: str="bl.bl_full_data"`
    - `mlflow_tracking_uri: str="http://localhost:5001"`, `model_name: str="bl_ranker"`
    - `tabpfn_backend: Literal["auto","api","local","fallback"]="auto"`, `tabpfn_token: SecretStr|None=None`, `tabpfn_deadline_s: float=1.2`
    - `holdout_days_prod: int=0`, `auto_promote: bool=True`, `gate_max_regression: float=0.05`
    - `cache_ttl_s: int=600`, `cache_max_items: int=10_000`
    - `api_url: str="http://localhost:8000"`, `admin_token: SecretStr|None=None`
  - `get_settings() -> Settings` (lru_cached; reads `.env`)

- [ ] **Step 1: Write failing tests**
```python
# tests/test_integrity.py
def test_original_scripts_match_upstream_sha():
    sums = dict(reversed(l.split()) for l in Path("reference/SHA256SUMS").read_text().splitlines())
    for name, sha in sums.items():
        assert hashlib.sha256(Path("src/bl_ranking/original", name).read_bytes()).hexdigest() == sha
# tests/test_config.py
def test_defaults(monkeypatch):  # env cleared
    s = Settings(_env_file=None)
    assert s.tabpfn_backend == "auto" and s.holdout_days_prod == 0 and s.tabpfn_deadline_s == 1.2
def test_token_is_secret(monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "abc"); assert "abc" not in repr(Settings(_env_file=None))
```
- [ ] **Step 2:** `pytest tests/test_integrity.py tests/test_config.py -v` → FAIL (module missing)
- [ ] **Step 3:** Write `pyproject.toml` (setuptools, `package-dir={""="src"}`, `[project.scripts] bl="bl_ranking.cli:app"`, ruff line-length 110 excluding `src/bl_ranking/original`, pytest markers above). Write `requirements.in` with the tech-stack packages (`torch` via the CPU index in Docker only). `pip-compile requirements.in -o requirements.txt`; `pip install -r requirements.txt -e .`. Implement `Settings`.
- [ ] **Step 4:** Tests PASS; `ruff check .` clean.
- [ ] **Step 5:** Commit `chore: package skeleton, locked deps, settings, upstream integrity test`.

### Task 2: Synthetic fixture + DuckDB data source with dtype parity

**Files:**
- Create: `tests/fixtures/synthetic.py`, `tests/conftest.py`, `src/bl_ranking/data/__init__.py`, `src/bl_ranking/data/source.py`
- Test: `tests/test_source.py`

**Interfaces:**
- Produces:
  - `make_synthetic_csv(path: Path, days: int = 21, sessions_per_day: int = 150, seed: int = 0) -> Path`. Same header as the real CSV (leading blank index column, `business_name` twice), the real category strings, `sub1` with leading zeros, ~50 % `register_date` empty, 4 brands + NaN client, `disposition_source` matching the brand for "Lead" rows.
  - Fixture `synthetic_csv` (session scope).
  - `@dataclass(frozen=True) Snapshot(path: Path, sha256: str, rows: int, min_session_dt: str, max_session_dt: str, table: str)`
  - `class DataSource(Protocol)`: `ingest_csv(csv_path: Path) -> int`, `snapshot(out_dir: Path, as_of: date | None = None) -> Snapshot`
  - `DuckDBSource(db_path: Path, table: str)`: ingest with `read_csv(..., all_varchar=true, header=true)`; snapshot = `COPY (SELECT * ... [WHERE session_dt <= as_of 23:59:59]) TO out_dir/'bl_full_data.csv' (HEADER, FORMAT csv)` + sha256
  - `SparkSource(table: str)`: same API via `spark.table(...).toPandas().to_csv(...)` with all-string schema (not tested locally)
  - `get_source(settings: Settings) -> DataSource`
  - `validate_snapshot(s: Snapshot, min_rows: int = 1000) -> None`. Raises `DataValidationError` with a clear message if:
    - any needed column is missing
    - `rows < min_rows`
    - `max_session_dt` is unparsable or earlier than `min_session_dt`
    - the share of rows with `register_date` is outside 5–95 %
    - any of the 7 survey columns is >50 % null among registered rows

- [ ] **Step 1: Write failing tests**
```python
NEEDED = [...]  # needed_columns list copied from train:66-70 as a test constant
def test_snapshot_reads_identically(tmp_path, synthetic_csv):
    src = DuckDBSource(tmp_path/"w.duckdb", "bl.bl_full_data"); assert src.ingest_csv(synthetic_csv) > 0
    snap = src.snapshot(tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(snap.path, low_memory=False)[NEEDED],
                                  pd.read_csv(synthetic_csv, low_memory=False)[NEEDED])
def test_snapshot_as_of_filters(tmp_path, synthetic_csv): ...  # max_session_dt <= as_of; rows < full
def test_snapshot_sha_stable(tmp_path, synthetic_csv): ...     # two snapshots → same sha256
def test_validate_rejects_missing_column(tmp_path): ...       # CSV without credit_score → DataValidationError("credit_score")
def test_validate_rejects_too_few_rows(tmp_path, synthetic_csv): ...  # min_rows=10**9 → DataValidationError
@pytest.mark.data
def test_real_data_parity(tmp_path): ...                        # same assert on data/bl_full_data.csv
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** `pytest tests/test_source.py -v` (incl. `-m data` locally) PASS.
- [ ] **Step 5:** Commit `feat: DuckDB data source with byte-faithful CSV snapshot, validation and parity tests`.

### Task 3: Payout backends, fallback, deadline

**Files:**
- Create: `src/bl_ranking/payout/__init__.py`, `src/bl_ranking/payout/backends.py`
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces:
  - `class BrandMeanRegressor`: `fit(X: pd.DataFrame, y) -> Self` stores the mean of `y` per `X["client_name"]` plus the global mean; `predict(X) -> np.ndarray` (unseen brand → global mean); accepts and ignores `**kwargs` in `__init__`.
  - `class PayoutBackend(Protocol)`: `name: str`; `regressor(**kwargs) -> Any` (drop-in for `TabPFNRegressor(...)`); `set_token(token: str) -> None`
  - `ApiBackend`, `LocalBackend(device="cpu")`, `FallbackBackend`; `resolve_backend(settings) -> PayoutBackend`
  - `patch_module(module: ModuleType, backend: PayoutBackend) -> None`: sets `module.TabPFNRegressor = backend.regressor` and `module.set_access_token = backend.set_token`. Callers pass `sys.modules[cls.__module__]`.
  - `class DeadlineRegressor(primary, fallback, deadline_s: float)`: `predict(X) -> np.ndarray`. Runs `primary.predict` on a shared `ThreadPoolExecutor`; on timeout or exception returns `fallback.predict(X)` and sets the context var `DEGRADED` (`contextvars.ContextVar[bool]`, default False) to True.
  - `load_api_model(record_path: Path, context: dict) -> Any`: `tabpfn_client.TabPFNRegressor.load_model(record_path)`. On `FittedModelNotFoundError` during the warm-up predict, refit once from `context["x"], context["y"]` (same constructor kwarg `ignore_pretraining_limits=True`).

- [ ] **Step 1: Write failing tests**
```python
def test_brand_mean_regressor():
    X = pd.DataFrame({"client_name": ["a","a","b"]}); m = BrandMeanRegressor().fit(X, [10, 20, 40])
    assert m.predict(pd.DataFrame({"client_name": ["a","b","zzz"]})).tolist() == [15, 40, pytest.approx(70/3)]
def test_deadline_returns_fallback():
    slow = Mock(predict=lambda X: time.sleep(0.5) or np.ones(len(X)))
    r = DeadlineRegressor(slow, BrandMeanRegressor().fit(X, y), deadline_s=0.05)
    out = r.predict(X); assert DEGRADED.get() is True and out.tolist() == fallback_expected
def test_primary_error_returns_fallback(): ...
def test_resolve_backend_auto_without_token_is_fallback(): ...
def test_patch_module_swaps_names(): ...  # types.ModuleType("m") with both attrs
def test_evicted_fit_refits_once(monkeypatch): ...  # fake load_model whose predict raises FittedModelNotFoundError once
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: payout backends (api/local/fallback), deadline fallback, persisted API fit`.

### Task 4: Training adapter + metrics

**Files:**
- Create: `src/bl_ranking/adapters.py` (TrainAdapter part), `src/bl_ranking/training/__init__.py`, `src/bl_ranking/training/metrics.py`
- Test: `tests/test_metrics.py`, `tests/test_train_adapter.py` (`slow`)

**Interfaces:**
- Consumes: `patch_module`, `FallbackBackend` (Task 3), `synthetic_csv` (Task 2)
- Produces:
  - `classification_metrics(y_true, y_pred, prefix: str) -> dict[str, float]`: keys `{prefix}_f1`, `{prefix}_precision`, `{prefix}_recall` for class 1 (`zero_division=0`)
  - `payout_metrics(y_true, y_pred) -> dict[str, float]`: `payout_mape`, `payout_mae`, plus `payout_mape_thr_{5,10,15,20}` / `payout_mae_thr_{t}` only where the slice `y_pred >= t` is non-empty
  - `class TrainAdapter(BLPayoutModelsFit)`:
    - `__init__(self, input_path: str, input_file: str, output_predictors_path: str, train_test: bool, holdout_days: int = 7)`
    - attributes after `fit_()`: `metrics_: dict[str, float]`, `model_tfm_`, `tfm_context_: dict`, `cb_model_`, `x_test_all_days_`, `y_test_all_days_`, `log_path_: Path`, `row_counts_: dict[str, int]`
    - overrides, each calling `super()`:
      - `split_by_time`: uses `days_for_test` when `train_test`, else `self.holdout_days`
      - `catbosot_model_sold_to_client`: keeps `cb_model_`
      - `tabpfn_regression_payout`: keeps model + context
      - `accuracy_classification_model`: adds overall + per-day `classification_metrics` (`sold_overall`, `sold_day_{d}`)
      - `accuracy_cont_payout_prediction`: calls super() inside `try/except ValueError` (an empty threshold slice under fallback raises after the first line has run), then reads `y_test_payout["pred_payout"]`, which the original assigns into the passed frame (train:302) → `payout_metrics`. One predict call, no cache.
      - `detect_gender_with_confidence`: per-instance dict cache keyed by `fname`
      - `setup_bl_logger`: preserves pre-existing root handlers after super() clears them
      - `bl_preprocessing`: records `row_counts_`; keeps the test frames
- [ ] **Step 1: Write failing tests**
```python
def test_payout_metrics_skips_empty_threshold():
    m = payout_metrics([10, 12], [6, 7]); assert "payout_mae_thr_20" not in m and m["payout_mae"] == pytest.approx(4.5)
@pytest.mark.slow
def test_train_test_on_synthetic_with_fallback(tmp_path, synthetic_csv):
    patch_module(sys.modules[TrainAdapter.__mro__[1].__module__], FallbackBackend())
    a = TrainAdapter(str(synthetic_csv.parent)+"/", synthetic_csv.name, str(tmp_path)+"/", train_test=True); a.fit_()
    assert {"sold_overall_f1", "payout_mae"} <= a.metrics_.keys() and a.log_path_.exists()
@pytest.mark.slow
def test_production_holdout_zero_trains_all_rows(tmp_path, synthetic_csv): ...
    # row_counts_["train"] == row_counts_["after_survey_filter"]; CB_bl_lead.cbm, payout_tfm_context.joblib, all_clients.csv written
@pytest.mark.slow
def test_holdout_seven_reproduces_original_split(...): ...  # train rows == original class run on the same CSV
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** `pytest -m "slow or not slow" tests/test_metrics.py tests/test_train_adapter.py -v` PASS.
- [ ] **Step 5:** Commit `feat: training adapter (holdout flag, metric capture, logger isolation, gender cache)`.

### Task 5: Predictor adapter, pyfunc model, registry manager

**Files:**
- Modify: `src/bl_ranking/adapters.py` (PredictAdapter)
- Create: `src/bl_ranking/models/__init__.py`, `models/pyfunc.py`, `models/registry.py`
- Test: `tests/test_predict_adapter.py` (`slow`), `tests/test_registry.py`

**Interfaces:**
- Consumes: TrainAdapter artifacts (Task 4), `DeadlineRegressor`, `load_api_model`, `resolve_backend` (Task 3)
- Produces:
  - `class PredictAdapter(BLPayoutModelsPredict)`:
    - `__init__(self, predictors_path: str, user_data: dict, loaded: LoadedModels)` (no per-request log file)
    - `load_models()` returns `loaded`
    - `import_preprocess()` runs `super()` under the module-level `threading.Lock` after setting `module.user_data = self.user_data`
    - `capture_warnings()` is a no-op
  - `@dataclass LoadedModels(model_tfm, tfm_columns: list[str], cb_model)`
  - `@dataclass RankResult(ranking: dict[str, dict[str, float]], degraded: bool)`
  - `class BLRankerModel(mlflow.pyfunc.PythonModel)`:
    - `load_context(context)`: lazy-imports the predictor module, builds the payout model per `model_config["payout_backend"]`, wraps it in `DeadlineRegressor`, runs a warm-up rank
    - `rank(user: dict) -> RankResult`: maps `{"expected_payout": 0, "prob_lead": 0}` → `{}`
    - `predict(context, model_input: list[dict[str, Any]], params=None) -> list[dict[str, Any]]`
  - `log_bl_ranker(predictors_dir: Path, payout_backend: str, api_record: Path | None) -> ModelInfo`: artifacts `cb_model`, `tfm_context`, `all_clients`, optional `tabpfn_api_model`; `code_paths=["src/bl_ranking"]`; synthetic input_example; pip requirements from `requirements.txt`
  - `class ModelRegistryManager(client: MlflowClient, name: str)`:
    - `register(run_id: str, tags: dict[str, str]) -> str`
    - `get(alias: str) -> ModelVersion | None`
    - `set_alias(alias: str, version: str) -> None`
    - `promote(version: str, *, force: bool = False, reason: str | None = None) -> str`: refuses a fallback version over a non-fallback champion unless `force`; tags `previous_champion`, `promoted_reason`
    - `rollback() -> str`: champion ← its `previous_champion` tag; raises if none
- [ ] **Step 1: Write failing tests**
```python
def test_promote_then_rollback(tmp_mlflow):   # sqlite tracking in tmp_path, 3 dummy pyfunc versions
    r = ModelRegistryManager(MlflowClient(), "bl_ranker_test")
    r.promote("1"); r.promote("2"); assert r.get("champion").version == "2"
    assert r.rollback() == "1" and r.get("champion").version == "1"
def test_fallback_cannot_replace_tabpfn_champion(tmp_mlflow): ...  # raises unless force=True
@pytest.mark.slow
def test_rank_matches_original_predict(trained_synthetic_dir):
    # original BLPayoutModelsPredict with module.user_data set + fallback-patched load_models
    assert BLRankerModel-from-dir.rank(EXAMPLE).ranking == original_result
@pytest.mark.slow
def test_concurrent_users_get_own_ranking(loaded_model):
    with ThreadPoolExecutor(8) as ex: results = list(ex.map(loaded_model.rank, users_a_b_alternating))
    assert all(r.ranking == expected[u] for r, u in zip(results, users_a_b_alternating))
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. The parity test is the **train-serve skew test**; name it that in its docstring.
- [ ] **Step 5:** Commit `feat: pyfunc bl_ranker with locked predictor adapter; registry manager with promote/rollback`.

### Task 6: Gate, training pipeline, CLI, run reliability

**Files:**
- Create: `src/bl_ranking/training/gate.py`, `training/tracking.py`, `training/pipeline.py`, `src/bl_ranking/cli.py`
- Test: `tests/test_gate.py`, `tests/test_pipeline.py` (`slow`), `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 2–5
- Produces:
  - `should_run_gate(snapshot_max: date, champion_data_end: date | None, holdout_days: int = 7) -> bool`: True only if a champion exists and `snapshot_max - holdout_days >= champion_data_end`
  - `evaluate_gate(challenger: dict, champion: dict, max_regression: float) -> GateResult`: fail if `champion_f1 - challenger_f1 > max_regression * champion_f1` or `challenger_mae - champion_mae > max_regression * champion_mae`
  - `@dataclass GateResult(decision: Literal["pass","fail","skipped_no_new_data","no_champion"], reason: str, challenger: dict, champion: dict | None)`
  - `score_on_holdout(cb_model, payout_model, x_test_all_days, y_test_all_days) -> dict` (same metric keys)
  - `ExperimentTracker(tracking_uri: str, prefix: str = "bl-ranking")`:
    - `run(mode: str, tags: dict) -> ContextManager[str]`: yields the run_id; experiment `{prefix}/{mode}`; `mlflow.enable_system_metrics_logging()` off
    - `log_snapshot(s: Snapshot)` (params `data_*`, sha tag, no rows)
    - `log_params/log_metrics/log_artifact`
  - `TrainingPipeline(settings, source, tracker, registry, backend)`: `run(mode: Literal["train_test","production"], as_of: date | None = None, trigger: str = "manual") -> PipelineResult(run_id, version: str | None, gate: GateResult | None, promoted: bool)`
    - every run starts with `validate_snapshot` (failure → exit 1, MLflow run FAILED with the reason)
    - production order: snapshot → (if a champion exists and `should_run_gate`) TrainAdapter train_test + champion `score_on_holdout` → TrainAdapter production → `model_tfm_.save_model()` when backend=api → log_bl_ranker → register (alias `challenger`, tags, **model card** = version description in markdown: data range + rows + snapshot sha, payout backend, metrics vs champion, gate decision, upstream commit) → gate → promote if `auto_promote` and gate ∈ {pass, skipped_no_new_data, no_champion} → best-effort `POST {api_url}/admin/reload`
  - CLI `bl`:
    - `load-data [--csv PATH]`: exits 1 with "CSV not found at …; download it from the Drive link in README" if missing
    - `train --mode {train_test,production} [--as-of YYYY-MM-DD] [--trigger schedule|manual]`
    - `promote [--version V] [--force --reason TEXT]`, `rollback`
    - `status`: champion version, model age, `last_success`
    - `serve`
  - Reliability: `train` takes the `fcntl` lock `warehouse/train.lock` (exit 2 if held); exit 1 on any exception (MLflow run marked FAILED); writes `warehouse/last_success.json` `{"mode","version","finished_at"}` on success
- [ ] **Step 1: Write failing tests**
```python
def test_gate_skipped_when_champion_saw_holdout():
    assert not should_run_gate(date(2026,2,11), date(2026,2,11))
def test_gate_runs_on_new_week():
    assert should_run_gate(date(2026,2,11), date(2026,2,4))
def test_gate_fails_on_f1_drop():
    r = evaluate_gate({"sold_overall_f1": .40, "payout_mae": 2.3}, {"sold_overall_f1": .47, "payout_mae": 2.3}, .05)
    assert r.decision == "fail"
@pytest.mark.slow
def test_production_run_registers_and_promotes(tmp_mlflow, synthetic_csv, tmp_path): ...
    # fallback backend; first run → no_champion → champion v1; second run with --as-of-later data → gate evaluated
def test_train_lock_exit_2(tmp_path): ...       # hold the lock, invoke via typer CliRunner → exit_code 2
def test_load_data_missing_csv_exit_1(): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. Run `bl load-data && bl train --mode train_test && bl train --mode production` against a local `mlflow server` and the real data; MLflow UI shows both experiments, params, per-day metrics, the researcher log and `bl_ranker` v1 @champion.
- [ ] **Step 5:** Commit `feat: training pipeline with leak-guarded promotion gate, CLI, run lock and last_success`.

### Task 7: Serving API

**Files:**
- Create: `src/bl_ranking/serving/__init__.py`, `serving/app.py`, `serving/schemas.py`, `serving/cache.py`, `serving/telemetry.py`, `src/bl_ranking/data/sinks.py`
- Test: `tests/test_api.py` (uses a `FakeRanker` for fast tests + one `slow` end-to-end), `tests/test_cache.py`, `tests/test_sinks.py`

**Interfaces:**
- Consumes: `BLRankerModel.rank`, `RankResult`, `DEGRADED`, `ModelRegistryManager.get`
- Produces:
  - `UserRequest(BaseModel)`: the 22 fields of the example dict in `bl_exp_payout_predictor.py`'s `__main__`. `campaign_id: int|str`; `sub1|sub2|sub3: int|str|None`; `cellphone: int` (validated numeric, required); `register_date: str` (required). Others `str|None`. `extra="ignore"`.
  - `ResponseCache(ttl_s: int, max_items: int)`: `key(payload: dict, model_version: str) -> str` (sha256 of canonical JSON + version), `get(key) -> dict | None`, `put(key, value)`, thread-safe (`cachetools.TTLCache` + lock)
  - `InferenceLog(db_path: Path)`: SQLite WAL; `start()`, `stop()`, `write(rec: InferenceRecord)` (non-blocking queue, batch insert every 1 s or 100 rows)
  - `InferenceRecord(ts, request_id, request_hash, model_version, payout_backend, latency_ms, degraded, cache_hit, n_brands, ranking_json)`; no PII columns
  - telemetry: `REQUEST_LATENCY` Histogram (buckets 0.05…3 s), `REQUESTS` Counter{status}, `FALLBACKS` Counter, `CACHE_HITS` Counter, `MODEL_INFO` Info{version, backend}; `configure_json_logging()`
  - `create_app(settings: Settings, loader: Callable[[], tuple[Ranker, str, str]] | None = None) -> FastAPI` (loader returns ranker, version, backend; default = load `models:/{name}@champion`). Endpoints:
    - `POST /v1/predict`: sync `def`; body = ranking dict; headers per Global Constraints; 422 on validation or original "cannot be a lead" errors
    - `GET /health`, `GET /ready` (503 until loaded; includes `model_version`, `model_age_hours`), `GET /version`
    - `POST /admin/reload` (header `X-Admin-Token` must match the setting if set; atomic swap), `GET /metrics`
- [ ] **Step 1: Write failing tests**
```python
def test_predict_returns_brand_dict_and_headers(client_fake):
    r = client_fake.post("/v1/predict", json=EXAMPLE)
    assert r.status_code == 200 and r.json() == FAKE_RANKING and r.headers["X-Model-Version"] == "7"
def test_predict_missing_register_date_422(client_fake): ...
def test_predict_bad_cellphone_422(client_fake): ...
def test_empty_ranking_returns_empty_dict(client_fake_empty): assert r.status_code == 200 and r.json() == {}
def test_cache_hit_second_call(client_fake): ...      # X-Cache-Hit false then true; ranker called once
def test_cache_key_changes_with_version(): ...
def test_reload_swaps_version(client_fake): ...       # loader returns v7 then v8 → /version shows 8
def test_metrics_exposed(client_fake): assert "bl_request_latency_seconds" in client.get("/metrics").text
def test_inference_log_no_pii(tmp_path): ...          # columns exclude fname/lname/cellphone; rows flushed
@pytest.mark.slow
def test_predict_unseen_categories_still_ranks(client_real_fallback): ...  # city="Atlantis", sub1=1121993
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS; then `bl serve` + `curl -s localhost:8000/v1/predict -d @load_test/example.json` returns a ranked dict, and `-i` shows the headers.
- [ ] **Step 5:** Commit `feat: FastAPI serving with brand-dict contract, cache, deadline fallback, metrics, hot reload, inference log`.

### Task 8: Load simulation and latency report

**Files:**
- Create: `load_test/payloads.py`, `load_test/locustfile.py`, `load_test/report.py`, `load_test/example.json` (synthetic person), `scripts/bench_original_predictor.py`
- Output: `docs/latency.md`

**Interfaces:**
- `build_payloads(csv: Path, n: int = 2000, repeat_rate: float = 0.05, seed: int = 0) -> list[dict]`: registered rows from the last 7 days → request dicts; names/phones replaced with synthetic values of the same length and prefix (so features are unchanged); kept in memory only
- locust user: `POST /v1/predict`, wait 0–50 ms; profiles `--users 20 --spawn-rate 5 --run-time 2m` (fallback/local) and `--users 2 --run-time 30s --iterations 40` (api, capped)
- `report.py <locust_stats.csv>... -> docs/latency.md`: table of p50/p95/p99/max/error rate per scenario, plus the step-1 TabPFN table and the original-predictor baseline
- `bench_original_predictor.py --n 20`: the original `BLPayoutModelsPredict.predict_` (global set, local/fallback patched) per request → p50/p99 (no credits)
- [ ] **Step 1:** `pytest tests/test_payloads.py`: `test_payloads_have_repeat_rate` (≈5 % ±2 % duplicate dicts), `test_payloads_no_real_names` (no payload fname appears in the CSV).
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Run the scenarios: original-baseline (bench script), ours+fallback, ours+local (if licence), ours+api (capped ≤50 requests). Generate `docs/latency.md`.
- [ ] **Step 4:** Verify `docs/latency.md` has the 4 rows with numbers and states the SLO pass/fail per row.
- [ ] **Step 5:** Commit `perf: load simulation and latency report`.

### Task 9: Docker Compose, scheduler, Makefile

**Files:**
- Create: `docker/Dockerfile`, `docker-compose.yml`, `scheduler/crontab`, `Makefile`, `.dockerignore`
**Interfaces:**
- One image (`python:3.12-slim`, CPU torch from `https://download.pytorch.org/whl/cpu`, supercronic pinned, `pip install -r requirements.txt -e .`), used by:
  - `api`: `bl serve`, 1 worker, port 8000
  - `scheduler`: `supercronic /app/scheduler/crontab`, `TZ=${SCHEDULER_TZ:-Asia/Jerusalem}`
  - `trainer`: profile `jobs`
- `mlflow` service: `ghcr.io/mlflow/mlflow:v3.17.1`, `mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --serve-artifacts --host 0.0.0.0 --port 5000`, published on host port **5001** (macOS AirPlay occupies 5000); `MLFLOW_TRACKING_URI` default becomes `http://localhost:5001`
- Volumes: `./data:/app/data:ro`, `warehouse`, `mlflow`, `hf-cache`. `env_file: .env`.
- `scheduler/crontab`: `0 5 * * 0 bl train --mode production --trigger schedule`
- Makefile targets: `up down logs load-data train-test train predict rollback status load-test test lint` and **`demo`**: up → wait ready → load-data → train-test → train (`--as-of 2026-02-04`, champion v1) → predict → train (full data, gate evaluated, v2) → predict → rollback → `/version` shows v1. It prints each step's outcome and stops on the first failure. It uses ≤10 API predictions and prints that up front.
- [ ] **Step 1:** `supercronic -test scheduler/crontab` → valid.
- [ ] **Step 2:** `make up && make load-data && make train && make predict` → ranked dict; `make rollback` (after a second `make train`) → `/version` changes without an api restart.
- [ ] **Step 3:** `docker stats` shows the api < 4 GB.
- [ ] **Step 3b: Early fresh-clone check.** `git clone` into an empty temp dir, copy only `data/` and `.env`, run `make demo` → completes. Fix findings now, not in Task 11.
- [ ] **Step 4:** Commit `build: docker compose stack, weekly scheduler, Makefile`.

### Task 10: Databricks bundle, Spark source wiring, CI

**Files:**
- Create: `databricks/databricks.yml`, `.github/workflows/ci.yml`
- Modify: `src/bl_ranking/config.py` (when `env=databricks`: tracking `databricks`, registry `databricks-uc`, model `main.bl.bl_ranker`)
**Interfaces:**
- Bundle: jobs `bl_load_data` (one-off) and `bl_weekly_train` (schedule `quartz_cron_expression: "0 0 5 ? * SUN"`, `timezone_id: Asia/Jerusalem`), each a `python_wheel_task` with entry point `bl` and the same CLI args. Secrets via the secret scope `bl` (`TABPFN_TOKEN`). The Model Serving endpoint `bl-ranker` serves `main.bl.bl_ranker@champion`.
- CI: on push/PR → setup-python 3.12 → `pip install -r requirements.txt -e .` (CPU torch index) → `ruff check .` → `pytest -m "not data and not api"`.
- [ ] **Step 1:** `test_config.py::test_databricks_env_switches_uris` FAIL → implement → PASS.
- [ ] **Step 2:** YAML lint passes; `databricks bundle validate` if the CLI is installed (otherwise noted in the README).
- [ ] **Step 3:** Push → CI green.
- [ ] **Step 4:** Commit `ci: GitHub Actions; deploy: Databricks asset bundle`.

### Task 11: README, Part 2 doc, fresh-clone acceptance

**Files:**
- Create/replace: `README.md`, `docs/part2.md` → `docs/part2.pdf`
- [ ] **Step 1:** README sections:
  - **Requirements map** (first section): one table row per brief requirement → where it's solved → how to see it (command/URL/file)
  - Quickstart (Docker Desktop ≥8 GB at the top, the Drive download step, `.env` with the optional `TABPFN_TOKEN` and its credit use, `make demo`)
  - Architecture diagram; local↔Databricks mapping
  - MLflow usage; promotion/rollback demo with `--as-of`
  - API contract; SLO + `docs/latency.md`
  - Reconciling the two briefs; findings about the original code
  - Limitations
  - Evidence: `docs/img/` MLflow screenshots (experiments compare, registry with the model card + aliases) and `docs/latency.md`
- [ ] **Step 2:** Part 2 (≤2 pages): implementation, dilemmas/decisions (api vs local with numbers, persisted fit, holdout fix, wrapping vs editing), production operation, issues (credits, eviction, fair-lending flag on inferred gender, data egress, 4-of-10 brands in the context, data gaps), improvements (feedback loop/revenue per session, shadow → A/B, drift, Redis, Postgres/MinIO), alternatives considered. Export to PDF; check the page count ≤ 2.
- [ ] **Step 3:** Fresh-clone acceptance: `git clone` into an empty temp dir, follow only the README, run the `make` flow end to end. Fix anything it reveals.
- [ ] **Step 4:** Commit `docs: README, Part 2 write-up`; push; publish the repo.
