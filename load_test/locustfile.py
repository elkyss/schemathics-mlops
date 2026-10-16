"""Locust traffic for /v1/predict. Configure with env vars:
PAYLOAD_CSV (default data/bl_full_data.csv), PAYLOADS (2000), REPEAT_RATE (0.05), WAIT ("0.0,0.05" seconds),
MAX_REQUESTS (stop after N requests — protects the TabPFN API quota), SUMMARY_JSON (degraded/cache-hit counts).
"""
import itertools
import json
import os
import threading
from pathlib import Path

from locust import HttpUser, between, events, task

from load_test.payloads import build_payloads

PAYLOADS = build_payloads(Path(os.environ.get("PAYLOAD_CSV", "data/bl_full_data.csv")),
                          n=int(os.environ.get("PAYLOADS", 2000)),
                          repeat_rate=float(os.environ.get("REPEAT_RATE", 0.05)))
_next = itertools.cycle(range(len(PAYLOADS)))
_lock = threading.Lock()
COUNTS = {"requests": 0, "degraded": 0, "cache_hit": 0}
MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS", 0))
WAIT = tuple(float(x) for x in os.environ.get("WAIT", "0.0,0.05").split(","))


class PredictUser(HttpUser):
    wait_time = between(*WAIT)

    def on_start(self):
        self.client.headers["content-type"] = "application/json"

    def _take(self):
        with _lock:
            if MAX_REQUESTS and COUNTS["requests"] >= MAX_REQUESTS:
                return None
            COUNTS["requests"] += 1
            return PAYLOADS[next(_next)]

    @task
    def predict(self):
        payload = self._take()
        if payload is None:
            self.environment.runner.quit()
            return
        with self.client.post("/v1/predict", json=payload, name="/v1/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            with _lock:
                COUNTS["degraded"] += r.headers.get("X-Degraded") == "true"
                COUNTS["cache_hit"] += r.headers.get("X-Cache-Hit") == "true"


@events.test_stop.add_listener
def _write_summary(environment, **_):
    if os.environ.get("SUMMARY_JSON"):
        Path(os.environ["SUMMARY_JSON"]).write_text(json.dumps(COUNTS))
