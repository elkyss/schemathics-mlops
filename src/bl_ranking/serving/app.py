"""FastAPI service: POST /v1/predict returns the ranked brand dict for one post-funnel user.

The body is exactly what the researcher's predictor returns ({brand: {rank, expected_payout}}); serving metadata
travels in headers (X-Model-Version, X-Payout-Backend, X-Degraded, X-Cache-Hit, X-Request-ID).

Everything heavy happens once: the @champion model loads in a background thread at startup (/ready is 503 until
then) and again only on POST /admin/reload, which swaps the model atomically — rollback without a restart.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from bl_ranking.config import Settings, get_settings
from bl_ranking.data.sinks import InferenceLog, InferenceRecord
from bl_ranking.serving import telemetry
from bl_ranking.serving.cache import ResponseCache
from bl_ranking.serving.schemas import UserRequest

log = logging.getLogger("bl_ranking.serving")


class Ranker(Protocol):
    def rank(self, user: dict) -> Any: ...


Loader = Callable[[], tuple[Ranker, str, str]]


@dataclass(frozen=True)
class ServedModel:
    ranker: Ranker
    version: str
    backend: str
    loaded_at: float


def champion_loader(settings: Settings) -> Loader:
    """Load models:/<name>@champion and unwrap the BLRankerModel (its .rank skips pyfunc input conversion)."""

    def load() -> tuple[Ranker, str, str]:
        import mlflow

        from bl_ranking.models.registry import CHAMPION, ModelRegistryManager

        os.environ["TABPFN_DEADLINE_S"] = str(settings.tabpfn_deadline_s)
        if settings.tabpfn_token:
            os.environ["TABPFN_TOKEN"] = settings.tabpfn_token.get_secret_value()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_registry_uri(settings.mlflow_tracking_uri)
        mv = ModelRegistryManager(mlflow.MlflowClient(), settings.model_name).get(CHAMPION)
        if mv is None:
            raise RuntimeError(f"no @{CHAMPION} for {settings.model_name}; run `bl train --mode production`")
        model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{mv.version}")
        return model.unwrap_python_model(), str(mv.version), mv.tags.get("payout_backend", "unknown")

    return load


def create_app(settings: Settings, loader: Loader | None = None) -> FastAPI:
    loader = loader or champion_loader(settings)
    cache = ResponseCache(settings.cache_ttl_s, settings.cache_max_items)
    inference_log = InferenceLog(settings.warehouse_dir / "inference.db")
    state: dict[str, ServedModel | None] = {"model": None}
    reload_lock = threading.Lock()
    load_error: dict[str, str] = {}

    def _load() -> ServedModel:
        with reload_lock:
            ranker, version, backend = loader()
            served = ServedModel(ranker, version, backend, time.time())
            state["model"] = served  # atomic swap: in-flight requests keep the object they already hold
            telemetry.MODEL_INFO.info({"version": version, "backend": backend})
            load_error.clear()
            log.info("model loaded", extra={"fields": {"model_version": version, "payout_backend": backend}})
            return served

    def _initial_load() -> None:
        try:
            _load()
        except Exception as exc:  # stay up (health OK, ready 503) so the cause is visible, not a crash loop
            load_error["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        telemetry.configure_json_logging()
        inference_log.start()
        threading.Thread(target=_initial_load, name="model-load", daemon=True).start()
        yield
        inference_log.stop()

    app = FastAPI(title="BL brand ranking", version="1", lifespan=lifespan)

    def _served() -> ServedModel:
        served = state["model"]
        if served is None:
            raise HTTPException(503, detail=load_error.get("error", "model is loading"))
        return served

    @app.post("/v1/predict")
    def predict(req: UserRequest) -> JSONResponse:  # sync: runs in the threadpool, the model code is blocking
        start = time.perf_counter()
        served = _served()
        request_id = uuid.uuid4().hex
        payload = req.model_dump()
        key = cache.key(payload, served.version)
        ranking = cache.get(key)
        cache_hit, degraded = ranking is not None, False
        if not cache_hit:
            try:
                result = served.ranker.rank(payload)
            except Exception as exc:
                telemetry.REQUESTS.labels(status="error").inc()
                if "cannot be a lead" in str(exc):
                    raise HTTPException(422, detail=str(exc)) from exc
                log.exception("prediction failed", extra={"fields": {"request_id": request_id}})
                raise HTTPException(500, detail="prediction failed") from exc
            ranking, degraded = result.ranking, result.degraded
            if not degraded:  # don't pin a fallback answer after TabPFN recovers
                cache.put(key, ranking)
        latency = time.perf_counter() - start
        telemetry.REQUEST_LATENCY.observe(latency)
        telemetry.REQUESTS.labels(status="ok").inc()
        if degraded:
            telemetry.FALLBACKS.inc()
        if cache_hit:
            telemetry.CACHE_HITS.inc()
        fields = {"request_id": request_id, "model_version": served.version, "payout_backend": served.backend,
                  "latency_ms": round(latency * 1000, 2), "degraded": degraded, "cache_hit": cache_hit,
                  "n_brands": len(ranking)}
        log.info("predict", extra={"fields": fields})
        inference_log.write(InferenceRecord(ts=time.time(), request_hash=key, ranking_json=json.dumps(ranking),
                                            **fields))
        return JSONResponse(ranking, headers={
            "X-Model-Version": served.version, "X-Payout-Backend": served.backend,
            "X-Degraded": str(degraded).lower(), "X-Cache-Hit": str(cache_hit).lower(), "X-Request-ID": request_id})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        served = _served()
        return {"status": "ready", "model_version": served.version, "payout_backend": served.backend,
                "model_age_hours": round((time.time() - served.loaded_at) / 3600, 3)}

    @app.get("/version")
    def version() -> dict:
        served = _served()
        return {"model_name": settings.model_name, "alias": "champion", "model_version": served.version,
                "payout_backend": served.backend, "loaded_at": served.loaded_at}

    @app.post("/admin/reload")
    def reload(x_admin_token: str | None = Header(default=None)) -> dict:
        if settings.admin_token and x_admin_token != settings.admin_token.get_secret_value():
            raise HTTPException(401, detail="invalid admin token")
        try:
            served = _load()
        except Exception as exc:
            raise HTTPException(500, detail=f"reload failed, still serving the previous model: {exc}") from exc
        return {"model_version": served.version, "payout_backend": served.backend}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def build_app() -> FastAPI:
    """uvicorn factory: `uvicorn bl_ranking.serving.app:build_app --factory`."""
    return create_app(get_settings())
