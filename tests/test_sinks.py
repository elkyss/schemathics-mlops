import sqlite3
import time

from bl_ranking.data.sinks import InferenceLog, InferenceRecord


def test_inference_log_flushes_without_pii(tmp_path):
    log = InferenceLog(tmp_path / "inference.db")
    log.start()
    for i in range(3):
        log.write(InferenceRecord(ts=time.time(), request_id=f"r{i}", request_hash="h", model_version="7",
                                  payout_backend="api", latency_ms=12.5, degraded=False, cache_hit=False,
                                  n_brands=2, ranking_json="{}"))
    log.stop()
    con = sqlite3.connect(tmp_path / "inference.db")
    cols = {r[1] for r in con.execute("PRAGMA table_info(inference_log)")}
    assert not cols & {"fname", "lname", "cellphone", "address", "business_name"}
    assert con.execute("SELECT count(*) FROM inference_log").fetchone()[0] == 3
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
