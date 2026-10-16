from datetime import date

import pandas as pd
import pytest

from bl_ranking.data.source import DataValidationError, DuckDBSource, validate_snapshot
from tests.conftest import REAL_CSV

# needed_columns from bl_models_train.py:66-70 — the columns the original training code reads.
NEEDED = ['session_id', 'session_dt', 'conversion_dt', 'register_date',
          'campaign_id', 'page', 'auto_city', 'auto_country', 'auto_state', 'device_type', 'sub1',
          'sub2', 'sub3', 'business_type', 'credit_score', 'industry', 'loan_amount', 'loan_reason',
          'monthly_revenue', 'time_in_business',
          'fname', 'lname', 'cellphone', 'client_name', 'payout', 'disposition', 'disposition_source']


def _source(tmp_path, csv):
    src = DuckDBSource(tmp_path / "w.duckdb", "bl.bl_full_data")
    assert src.ingest_csv(csv) > 0
    return src


def _assert_reads_identically(snapshot_path, original_csv):
    pd.testing.assert_frame_equal(pd.read_csv(snapshot_path, low_memory=False)[NEEDED],
                                  pd.read_csv(original_csv, low_memory=False)[NEEDED])


def test_snapshot_reads_identically(tmp_path, synthetic_csv):
    snap = _source(tmp_path, synthetic_csv).snapshot(tmp_path / "snap")
    _assert_reads_identically(snap.path, synthetic_csv)
    assert snap.rows == len(pd.read_csv(synthetic_csv, low_memory=False))


def test_snapshot_keeps_leading_zeros(tmp_path, synthetic_csv):
    snap = _source(tmp_path, synthetic_csv).snapshot(tmp_path / "snap")
    # The raw text must keep them; how pandas then parses it is the original code's business.
    raw = pd.read_csv(snap.path, dtype=str, keep_default_na=False)["sub1"]
    assert raw.str.startswith("0112").all()


def test_snapshot_as_of_filters(tmp_path, synthetic_csv):
    src = _source(tmp_path, synthetic_csv)
    full = src.snapshot(tmp_path / "full")
    cut = src.snapshot(tmp_path / "cut", as_of=date(2026, 1, 10))
    assert cut.max_session_dt <= "2026-01-10 23:59:59"
    assert 0 < cut.rows < full.rows


def test_snapshot_sha_stable(tmp_path, synthetic_csv):
    src = _source(tmp_path, synthetic_csv)
    assert src.snapshot(tmp_path / "a").sha256 == src.snapshot(tmp_path / "b").sha256


def test_validate_accepts_synthetic(tmp_path, synthetic_csv):
    validate_snapshot(_source(tmp_path, synthetic_csv).snapshot(tmp_path / "snap"))


def test_validate_rejects_missing_column(tmp_path, synthetic_csv):
    broken = tmp_path / "broken.csv"
    pd.read_csv(synthetic_csv, low_memory=False).drop(columns=["credit_score"]).to_csv(broken, index=False)
    with pytest.raises(DataValidationError, match="credit_score"):
        validate_snapshot(_source(tmp_path, broken).snapshot(tmp_path / "snap"))


def test_validate_rejects_too_few_rows(tmp_path, synthetic_csv):
    with pytest.raises(DataValidationError, match="rows"):
        validate_snapshot(_source(tmp_path, synthetic_csv).snapshot(tmp_path / "snap"), min_rows=10**9)


@pytest.mark.data
def test_real_data_parity(tmp_path):
    if not REAL_CSV.exists():
        pytest.skip("real data not present")
    snap = _source(tmp_path, REAL_CSV).snapshot(tmp_path / "snap")
    _assert_reads_identically(snap.path, REAL_CSV)
    validate_snapshot(snap)
