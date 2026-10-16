"""Prometheus metrics (/metrics) and one structured JSON log line per request (no personal data)."""
import json
import logging
import sys

from prometheus_client import Counter, Histogram, Info

REQUEST_LATENCY = Histogram("bl_request_latency_seconds", "End-to-end /v1/predict latency",
                            buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0))
REQUESTS = Counter("bl_requests_total", "Predict requests by outcome", ["status"])
FALLBACKS = Counter("bl_payout_fallbacks_total", "Requests answered with the fallback payout model")
CACHE_HITS = Counter("bl_cache_hits_total", "Requests answered from the response cache")
MODEL_INFO = Info("bl_model", "Model currently served")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
                   "msg": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("bl_ranking.serving")
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
