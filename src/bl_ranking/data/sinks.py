"""Inference log ("results" in the brief): one row per prediction, written off the request path.

Local: SQLite in WAL mode (one writer + concurrent readers across processes). Databricks: a Delta table fed the
same records. No personal data is stored — only a request hash, the ranking and serving metadata.
"""
from __future__ import annotations

import queue
import sqlite3
import threading
from dataclasses import astuple, dataclass, fields
from pathlib import Path


@dataclass
class InferenceRecord:
    ts: float
    request_id: str
    request_hash: str
    model_version: str
    payout_backend: str
    latency_ms: float
    degraded: bool
    cache_hit: bool
    n_brands: int
    ranking_json: str


_COLUMNS = [f.name for f in fields(InferenceRecord)]


class InferenceLog:
    def __init__(self, db_path: Path, batch_size: int = 100, flush_interval_s: float = 1.0):
        self.db_path = Path(db_path)
        self.batch_size, self.flush_interval_s = batch_size, flush_interval_s
        self._queue: queue.Queue[InferenceRecord | None] = queue.Queue(maxsize=10_000)
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def start(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="inference-log", daemon=True)
        self._thread.start()

    def write(self, rec: InferenceRecord) -> None:
        try:
            self._queue.put_nowait(rec)
        except queue.Full:  # never block a user request on logging
            self.dropped += 1

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10)

    def _run(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(f"CREATE TABLE IF NOT EXISTS inference_log ({', '.join(_COLUMNS)})")
        insert = f"INSERT INTO inference_log VALUES ({', '.join('?' * len(_COLUMNS))})"
        batch: list[tuple] = []
        stopping = False
        while not stopping:
            try:
                item = self._queue.get(timeout=self.flush_interval_s)
                if item is None:
                    stopping = True
                else:
                    batch.append(astuple(item))
            except queue.Empty:
                pass
            if batch and (stopping or len(batch) >= self.batch_size or self._queue.empty()):
                con.executemany(insert, batch)
                con.commit()
                batch.clear()
        con.close()
