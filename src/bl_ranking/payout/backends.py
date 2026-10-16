"""Where the payout model (TabPFN) runs, chosen by TABPFN_BACKEND.

The researcher's modules call ``set_access_token(os.environ["TABPFN_TOKEN"])`` and ``TabPFNRegressor(...)``
by their module-global names. ``patch_module`` rebinds those two names, so the original function bodies run
unchanged against whichever backend is configured:

- ``local``    (default) OSS ``tabpfn`` on CPU with the context cached once (no network; non-commercial weights).
- ``api``      tabpfn-client (hosted TabPFN, the researcher's choice). A fit lives on Prior Labs' servers under an
               id; ``PersistedApiRegressor`` reloads that id instead of re-uploading the context.
- ``fallback`` ``BrandMeanRegressor``: mean historical payout per brand. Used when there is no token and as the
               safety net behind ``DeadlineRegressor`` when TabPFN is slow or down.
"""
from __future__ import annotations

import contextvars
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from bl_ranking.config import Settings

log = logging.getLogger(__name__)

# Set to True by DeadlineRegressor when it had to answer with the fallback. Read in the same thread.
DEGRADED: contextvars.ContextVar[bool] = contextvars.ContextVar("bl_payout_degraded", default=False)

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="payout")


class BrandMeanRegressor:
    """Payout ≈ the brand's historical mean payout (global mean for unseen brands)."""

    def __init__(self, **_ignored: Any):
        self.brand_means_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y) -> BrandMeanRegressor:
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.brand_means_ = y.groupby(X["client_name"]).mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["client_name"].map(self.brand_means_).fillna(self.global_mean_).to_numpy(dtype=float)


class PayoutBackend(Protocol):
    name: str

    def regressor(self, **kwargs: Any) -> Any: ...

    def set_token(self, token: str) -> None: ...


def _client_regressor_cls():
    from tabpfn_client import TabPFNRegressor

    return TabPFNRegressor


class ApiBackend:
    name = "api"

    def regressor(self, **kwargs: Any) -> Any:
        return _client_regressor_cls()(**kwargs)

    def set_token(self, token: str) -> None:
        from tabpfn_client import set_access_token

        set_access_token(token)


class ChunkedRegressor:
    """Predict in fixed-size row chunks. TabPFN test rows attend only to the fitted context, so the predictions
    are the same as one big batch, but memory stays bounded (the full test week in one batch peaked at 11 GB)."""

    def __init__(self, model: Any, batch_rows: int):
        self.model, self.batch_rows = model, batch_rows

    def fit(self, X, y) -> ChunkedRegressor:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        parts = [np.asarray(self.model.predict(X.iloc[i:i + self.batch_rows]), dtype=float)
                 for i in range(0, len(X), self.batch_rows)]
        return np.concatenate(parts) if parts else np.empty(0)


class LocalBackend:
    name = "local"

    def __init__(self, device: str = "cpu", predict_batch_rows: int = 256):
        self.device, self.predict_batch_rows = device, predict_batch_rows

    def regressor(self, **kwargs: Any) -> ChunkedRegressor:
        from tabpfn import TabPFNRegressor  # weights download once with TABPFN_TOKEN after licence acceptance

        return ChunkedRegressor(TabPFNRegressor(device=self.device, fit_mode="fit_with_cache", **kwargs),
                                self.predict_batch_rows)

    def set_token(self, token: str) -> None:
        os.environ["TABPFN_TOKEN"] = token


class FallbackBackend:
    name = "fallback"

    def regressor(self, **kwargs: Any) -> BrandMeanRegressor:
        return BrandMeanRegressor(**kwargs)

    def set_token(self, token: str) -> None:
        return None


def resolve_backend(settings: Settings) -> PayoutBackend:
    """Default ``local``; ``api`` by config. Both need TABPFN_TOKEN (local: once, for the licensed weight
    download), so without a token the system still runs end to end on the fallback payout model."""
    token = settings.tabpfn_token.get_secret_value() if settings.tabpfn_token else ""
    if token:
        os.environ["TABPFN_TOKEN"] = token  # the original code reads it from the environment
    choice = "local" if settings.tabpfn_backend == "auto" else settings.tabpfn_backend
    if choice in ("api", "local") and not token:
        log.warning("TABPFN_TOKEN not set: payout backend %r unavailable, using the fallback (brand mean payout)",
                    choice)
        choice = "fallback"
    return {"api": ApiBackend, "local": LocalBackend, "fallback": FallbackBackend}[choice]()


def patch_module(module: ModuleType, backend: PayoutBackend) -> None:
    """Rebind the original module's TabPFN names. Pass ``sys.modules[cls.__module__]``, never an alias."""
    module.TabPFNRegressor = backend.regressor
    module.set_access_token = backend.set_token
    if backend.name == "fallback":
        # The original evaluates os.environ["TABPFN_TOKEN"] before calling set_access_token.
        os.environ.setdefault("TABPFN_TOKEN", "")


class PersistedApiRegressor:
    """A hosted TabPFN fit reloaded by id. If the server evicted it, refit once from the versioned context."""

    def __init__(self, record_path: Path, context: dict):
        self.record_path = Path(record_path)
        self.context = context
        self._lock = threading.Lock()
        self._model = _client_regressor_cls().load_model(self.record_path)

    def _refit(self, stale: Any) -> None:
        with self._lock:
            if self._model is not stale:  # another request already refit while we waited
                return
            model = _client_regressor_cls()(ignore_pretraining_limits=True)
            model.fit(self.context["x"], self.context["y"])
            self._model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        from tabpfn_client import FittedModelNotFoundError

        model = self._model
        try:
            return model.predict(X)
        except FittedModelNotFoundError:
            self._refit(stale=model)
            return self._model.predict(X)


def load_api_model(record_path: Path, context: dict) -> PersistedApiRegressor:
    return PersistedApiRegressor(record_path, context)


class DeadlineRegressor:
    """Answer within ``deadline_s``: the primary model if it is fast and healthy, else the fallback."""

    def __init__(self, primary: Any, fallback: Any, deadline_s: float, executor: ThreadPoolExecutor | None = None):
        self.primary, self.fallback, self.deadline_s = primary, fallback, deadline_s
        self.executor = executor or _EXECUTOR

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        future = self.executor.submit(self.primary.predict, X)
        try:
            return np.asarray(future.result(timeout=self.deadline_s), dtype=float)
        except Exception:  # timeout, network error, quota, eviction refit still running
            future.cancel()  # a call still queued never runs (no credits spent on an answer nobody reads)
            DEGRADED.set(True)
            return self.fallback.predict(X)
