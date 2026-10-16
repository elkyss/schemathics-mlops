"""Realistic /v1/predict traffic built from the last 7 days of registered sessions, de-identified in memory.

Names and phone numbers are replaced (same length, same 3-digit area code — the model's features are name length,
inferred gender and phone prefix), so no real person's data is sent or stored. A share of requests are exact
repeats, like refreshes and double submits in production.
"""
from __future__ import annotations

import random
import string
from pathlib import Path

import pandas as pd

REQUEST_FIELDS = ["session_dt", "conversion_dt", "register_date", "campaign_id", "page", "auto_city",
                  "auto_country", "auto_state", "device_type", "sub1", "sub2", "sub3", "business_type",
                  "credit_score", "industry", "loan_amount", "loan_reason", "monthly_revenue", "time_in_business",
                  "fname", "lname", "cellphone"]
FIRST_NAMES = ["Al", "Ana", "Alex", "Maria", "Robert", "Jessica", "Jonathan", "Christina", "Alexandria"]


def _synthetic_name(rng: random.Random, length: int, original: str, pool: list[str] | None = None) -> str:
    candidates = [n for n in (pool or []) if len(n) == length and n != original]
    if candidates:
        return rng.choice(candidates)
    while True:
        name = (rng.choice(string.ascii_uppercase) + "".join(rng.choices(string.ascii_lowercase, k=length - 1))
                if length > 0 else "")
        if name != original:
            return name


def _to_request(row: pd.Series, rng: random.Random) -> dict:
    d = {k: (None if pd.isna(row[k]) else row[k]) for k in REQUEST_FIELDS}
    fname, lname = str(d["fname"] or ""), str(d["lname"] or "")
    d["fname"] = _synthetic_name(rng, len(fname), fname, FIRST_NAMES)
    d["lname"] = _synthetic_name(rng, len(lname), lname)
    phone = str(int(float(d["cellphone"])))
    while True:
        new_phone = int(phone[:3] + "".join(rng.choices(string.digits, k=max(len(phone) - 3, 7))))
        if new_phone != int(phone):
            break
    d["cellphone"] = new_phone
    if d["campaign_id"] is not None:
        d["campaign_id"] = int(float(d["campaign_id"]))
    return d


def build_payloads(csv: Path, n: int = 2000, repeat_rate: float = 0.05, seed: int = 0) -> list[dict]:
    raw = pd.read_csv(csv, dtype=str, keep_default_na=True, usecols=REQUEST_FIELDS + ["session_id"])
    raw = raw[raw["register_date"].notna() & raw["cellphone"].notna()].drop_duplicates("session_id")
    ts = pd.to_datetime(raw["session_dt"], errors="coerce")
    recent = raw[ts >= ts.max() - pd.Timedelta(days=7)]
    rng = random.Random(seed)
    rows = recent.sample(n=min(n, len(recent)), random_state=seed, replace=len(recent) < n)
    out: list[dict] = []
    for _, row in rows.iterrows():
        if out and rng.random() < repeat_rate:
            out.append(dict(rng.choice(out)))
        else:
            out.append(_to_request(row, rng))
        if len(out) == n:
            break
    return out
