"""Training data lives in a table (DuckDB locally, Delta/Unity Catalog on Databricks).

The researcher's code reads a CSV from a path (``pd.read_csv(input_path + input_file)``), so each run takes a
*snapshot* of the table as a CSV and hands that path to the unchanged constructor. Every column is stored as text
so the snapshot reproduces the source CSV's values exactly (leading zeros in ``sub1``, 18-digit ``campaign_id``,
``cellphone``) and ``pd.read_csv`` infers the same dtypes it did on the original file.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import duckdb
import pandas as pd

if TYPE_CHECKING:
    from bl_ranking.config import Settings

SNAPSHOT_FILE = "bl_full_data.csv"

# Columns the researcher's training code selects (bl_models_train.py:66-70).
NEEDED_COLUMNS = ['session_id', 'session_dt', 'conversion_dt', 'register_date',
                  'campaign_id', 'page', 'auto_city', 'auto_country', 'auto_state', 'device_type', 'sub1',
                  'sub2', 'sub3', 'business_type', 'credit_score', 'industry', 'loan_amount', 'loan_reason',
                  'monthly_revenue', 'time_in_business',
                  'fname', 'lname', 'cellphone', 'client_name', 'payout', 'disposition', 'disposition_source']
SURVEY_COLUMNS = ["business_type", "credit_score", "industry", "loan_amount", "loan_reason",
                  "monthly_revenue", "time_in_business"]


class DataValidationError(ValueError):
    """The snapshot is not fit to train on; the message says why."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    sha256: str
    rows: int
    min_session_dt: str
    max_session_dt: str
    table: str


class DataSource(Protocol):
    def ingest_csv(self, csv_path: Path) -> int: ...

    def snapshot(self, out_dir: Path, as_of: date | None = None) -> Snapshot: ...


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_of_bound(as_of: date | None) -> str | None:
    return None if as_of is None else f"{as_of.isoformat()} 23:59:59"


class DuckDBSource:
    """Local stand-in for a Databricks table: one DuckDB file, all columns VARCHAR."""

    def __init__(self, db_path: Path, table: str):
        self.db_path = Path(db_path)
        self.table = table
        self.schema, self.name = table.split(".", 1) if "." in table else ("main", table)
        # DuckDB names the catalog after the file; qualify fully so e.g. bl.duckdb + schema "bl" is unambiguous.
        self._qualified = f'"{self.db_path.stem}"."{self.schema}"."{self.name}"'

    def _connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.db_path), read_only=read_only)

    def ingest_csv(self, csv_path: Path) -> int:
        with self._connect() as con:
            con.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.db_path.stem}"."{self.schema}"')
            con.execute(
                f"CREATE OR REPLACE TABLE {self._qualified} AS "
                "SELECT * FROM read_csv(?, all_varchar=true, header=true, quote='\"', escape='\"')",
                [str(csv_path)],
            )
            return con.execute(f"SELECT count(*) FROM {self._qualified}").fetchone()[0]

    def snapshot(self, out_dir: Path, as_of: date | None = None) -> Snapshot:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / SNAPSHOT_FILE
        bound = _as_of_bound(as_of)
        where = "WHERE session_dt <= $bound" if bound else ""
        params = {"bound": bound} if bound else {}
        with self._connect(read_only=True) as con:
            # rowid keeps the source row order, which the original's (unstable) sort_values depends on.
            query = f"SELECT * FROM {self._qualified} {where} ORDER BY rowid"
            con.execute(f"COPY ({query}) TO '{path}' (HEADER, FORMAT csv)", params)
            rows, lo, hi = con.execute(
                f"SELECT count(*), min(session_dt), max(session_dt) FROM {self._qualified} {where}", params
            ).fetchone()
        return Snapshot(path=path, sha256=_sha256(path), rows=rows, min_session_dt=lo or "",
                        max_session_dt=hi or "", table=self.table)


class SparkSource:
    """Databricks: the same contract on a Unity Catalog table (runs only inside a Databricks job)."""

    def __init__(self, table: str):
        self.table = table

    @staticmethod
    def _spark():
        from pyspark.sql import SparkSession  # available on Databricks runtimes only

        return SparkSession.builder.getOrCreate()

    def ingest_csv(self, csv_path: Path) -> int:
        spark = self._spark()
        df = spark.read.csv(str(csv_path), header=True, inferSchema=False, multiLine=True, escape='"')
        df.write.mode("overwrite").saveAsTable(self.table)
        return df.count()

    def snapshot(self, out_dir: Path, as_of: date | None = None) -> Snapshot:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self._spark().table(self.table)
        bound = _as_of_bound(as_of)
        if bound:
            df = df.where(df.session_dt <= bound)
        pdf = df.toPandas()
        path = out_dir / SNAPSHOT_FILE
        pdf.to_csv(path, index=False)
        return Snapshot(path=path, sha256=_sha256(path), rows=len(pdf), min_session_dt=str(pdf.session_dt.min()),
                        max_session_dt=str(pdf.session_dt.max()), table=self.table)


def get_source(settings: Settings) -> DataSource:
    if settings.env == "databricks":
        return SparkSource(settings.table_name)
    return DuckDBSource(settings.warehouse_dir / "bl.duckdb", settings.table_name)


def validate_snapshot(s: Snapshot, min_rows: int = 1000) -> None:
    """Fail fast with a clear reason instead of silently training on a broken extract."""
    header = pd.read_csv(s.path, nrows=0).columns
    missing = [c for c in NEEDED_COLUMNS if c not in header]
    if missing:
        raise DataValidationError(f"snapshot is missing required columns: {missing}")
    if s.rows < min_rows:
        raise DataValidationError(f"snapshot has {s.rows} rows, fewer than the minimum {min_rows}")
    lo, hi = pd.to_datetime(s.min_session_dt, errors="coerce"), pd.to_datetime(s.max_session_dt, errors="coerce")
    if pd.isna(lo) or pd.isna(hi) or hi < lo:
        raise DataValidationError(f"invalid session_dt range: {s.min_session_dt!r} .. {s.max_session_dt!r}")
    df = pd.read_csv(s.path, usecols=["register_date", *SURVEY_COLUMNS], low_memory=False)
    registered_share = df["register_date"].notna().mean()
    if not 0.05 <= registered_share <= 0.95:
        raise DataValidationError(f"implausible share of registered rows: {registered_share:.1%}")
    null_rates = df[df["register_date"].notna()][SURVEY_COLUMNS].isna().mean()
    too_sparse = null_rates[null_rates > 0.5]
    if not too_sparse.empty:
        raise DataValidationError(f"survey columns >50% empty among registered rows: {too_sparse.round(2).to_dict()}")
