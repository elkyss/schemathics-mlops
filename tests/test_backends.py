import os
import time
import types
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from tabpfn_client import FittedModelNotFoundError

from bl_ranking.config import Settings
from bl_ranking.payout import backends
from bl_ranking.payout.backends import (
    DEGRADED,
    ApiBackend,
    BrandMeanRegressor,
    DeadlineRegressor,
    FallbackBackend,
    PersistedApiRegressor,
    patch_module,
    resolve_backend,
)

X = pd.DataFrame({"client_name": ["a", "a", "b"], "f": [1, 2, 3]})
Y = [10.0, 20.0, 40.0]


def test_brand_mean_regressor():
    m = BrandMeanRegressor(ignore_pretraining_limits=True).fit(X, Y)
    out = m.predict(pd.DataFrame({"client_name": ["a", "b", "zzz"]}))
    assert out.tolist() == [15.0, 40.0, pytest.approx(70 / 3)]


def test_deadline_returns_fallback_on_timeout():
    DEGRADED.set(False)
    slow = Mock(predict=lambda x: time.sleep(0.5) or np.ones(len(x)))
    fb = BrandMeanRegressor().fit(X, Y)
    start = time.perf_counter()
    out = DeadlineRegressor(slow, fb, deadline_s=0.05).predict(X)
    assert time.perf_counter() - start < 0.3
    assert out.tolist() == fb.predict(X).tolist()
    assert DEGRADED.get() is True


def test_deadline_passes_through_fast_primary():
    DEGRADED.set(False)
    fast = Mock(predict=lambda x: np.full(len(x), 7.0))
    out = DeadlineRegressor(fast, BrandMeanRegressor().fit(X, Y), deadline_s=1).predict(X)
    assert out.tolist() == [7.0, 7.0, 7.0] and DEGRADED.get() is False


def test_primary_error_returns_fallback():
    DEGRADED.set(False)
    broken = Mock(predict=Mock(side_effect=ConnectionError("api down")))
    fb = BrandMeanRegressor().fit(X, Y)
    assert DeadlineRegressor(broken, fb, deadline_s=1).predict(X).tolist() == fb.predict(X).tolist()
    assert DEGRADED.get() is True


def test_default_backend_is_local(monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "tok")
    assert resolve_backend(Settings(_env_file=None)).name == "local"


@pytest.mark.parametrize("choice", ["local", "api", "auto"])
def test_no_token_falls_back(monkeypatch, choice):
    """Local needs the token once (licensed weight download); the API needs it per call."""
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.setenv("TABPFN_BACKEND", choice)
    assert resolve_backend(Settings(_env_file=None)).name == "fallback"


def test_api_backend_by_config(monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "tok")
    monkeypatch.setenv("TABPFN_BACKEND", "api")
    assert isinstance(resolve_backend(Settings(_env_file=None)), ApiBackend)


def test_patch_module_swaps_names():
    mod = types.ModuleType("fake_original")
    mod.TabPFNRegressor, mod.set_access_token = object(), object()
    backend = FallbackBackend()
    patch_module(mod, backend)
    assert isinstance(mod.TabPFNRegressor(ignore_pretraining_limits=True), BrandMeanRegressor)
    assert mod.set_access_token("anything") is None


def test_patch_module_fallback_sets_placeholder_token(monkeypatch):
    """The original evaluates os.environ['TABPFN_TOKEN'] before calling set_access_token."""
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    patch_module(types.ModuleType("m"), FallbackBackend())
    assert "TABPFN_TOKEN" in os.environ


def test_evicted_fit_refits_once(monkeypatch, tmp_path):
    calls = {"fit": 0}

    class FakeClientRegressor:
        def __init__(self, **kw):
            self.evicted = True

        @classmethod
        def load_model(cls, path):
            return cls()

        def fit(self, x, y):
            calls["fit"] += 1
            self.evicted = False
            return self

        def predict(self, x):
            if self.evicted:
                raise FittedModelNotFoundError("gone")
            return np.full(len(x), 3.0)

        def save_model(self, path):
            pass

    monkeypatch.setattr(backends, "_client_regressor_cls", lambda: FakeClientRegressor)
    reg = PersistedApiRegressor(tmp_path / "rec.json", {"x": X, "y": pd.Series(Y)})
    assert reg.predict(X).tolist() == [3.0, 3.0, 3.0]
    assert reg.predict(X).tolist() == [3.0, 3.0, 3.0]
    assert calls["fit"] == 1


def test_api_payout_model_warms_connection_without_credits(monkeypatch, tmp_path):
    """The first hosted predict pays ~1.3 s of connection/auth setup; a free usage call at load absorbs it."""
    import tabpfn_client

    from bl_ranking.models import pyfunc

    calls = []
    monkeypatch.setattr(tabpfn_client, "get_api_usage", lambda: calls.append("usage") or "ok")
    monkeypatch.setattr(tabpfn_client, "set_access_token", lambda token: None)
    primary = Mock(predict=Mock(side_effect=lambda x: np.ones(len(x))))
    monkeypatch.setattr(pyfunc, "load_api_model", lambda record, context: primary)
    (tmp_path / pyfunc.API_RECORD).write_text("{}")
    model = pyfunc._build_payout_model(tmp_path, {"x": X, "y": Y}, "api", deadline_s=1.4)
    assert calls == ["usage"] and isinstance(model, DeadlineRegressor)
    # one real 1-row prediction at load (one API call per model load) so the first user isn't served cold
    assert primary.predict.call_count == 1 and len(primary.predict.call_args[0][0]) == 1


def test_concurrent_eviction_refits_once(monkeypatch, tmp_path):
    """Final review: N in-flight requests hitting an evicted fit must trigger one refit, not N."""
    import threading

    calls = {"fit": 0}
    barrier = threading.Barrier(4)

    class FakeClientRegressor:
        def __init__(self, **kw):
            self.evicted = True

        @classmethod
        def load_model(cls, path):
            return cls()

        def fit(self, x, y):
            calls["fit"] += 1
            time.sleep(0.2)
            self.evicted = False
            return self

        def predict(self, x):
            if self.evicted:
                barrier.wait(timeout=5)  # all requests observe the eviction together
                raise FittedModelNotFoundError("gone")
            return np.full(len(x), 3.0)

    monkeypatch.setattr(backends, "_client_regressor_cls", lambda: FakeClientRegressor)
    reg = PersistedApiRegressor(tmp_path / "rec.json", {"x": X, "y": pd.Series(Y)})
    threads = [threading.Thread(target=reg.predict, args=(X,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert calls["fit"] == 1


def test_timed_out_queued_call_is_cancelled():
    """Final review: during an API hang, queued predicts must not run later and spend credits nobody reads."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release, started = threading.Event(), []

    def slow_predict(x):
        started.append(1)
        release.wait(timeout=5)
        return np.ones(len(x))

    fb = BrandMeanRegressor().fit(X, Y)
    reg = DeadlineRegressor(Mock(predict=slow_predict), fb, deadline_s=0.05, executor=ThreadPoolExecutor(1))
    reg.predict(X)  # occupies the only worker, times out
    reg.predict(X)  # queued behind it, times out → must be cancelled
    release.set()
    time.sleep(0.3)
    assert len(started) == 1


def test_local_backend_predicts_in_bounded_chunks(monkeypatch):
    """Local TabPFN on the whole test week in one batch peaked at 11 GB RSS (OOM in Docker's ~7.6 GB);
    test rows only attend to the context, so chunking keeps memory bounded with the same predictions."""
    import sys
    import types

    batches = []

    class FakeTabPFN:
        def __init__(self, **kw):
            self.kw = kw

        def fit(self, x, y):
            return self

        def predict(self, x):
            batches.append(len(x))
            return np.asarray(x["f"], dtype=float) * 2

    monkeypatch.setitem(sys.modules, "tabpfn", types.SimpleNamespace(TabPFNRegressor=FakeTabPFN))
    from bl_ranking.payout.backends import LocalBackend

    reg = LocalBackend(predict_batch_rows=4).regressor(ignore_pretraining_limits=True).fit(X, Y)
    q = pd.DataFrame({"client_name": ["a"] * 10, "f": range(10)})
    assert reg.predict(q).tolist() == [float(2 * i) for i in range(10)]
    assert batches == [4, 4, 2]
