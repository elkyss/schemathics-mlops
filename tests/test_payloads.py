import pandas as pd

from load_test.payloads import build_payloads
from tests.fixtures.synthetic import HEADER


def test_payloads_have_repeat_rate(synthetic_csv):
    p = build_payloads(synthetic_csv, n=400, repeat_rate=0.05, seed=1)
    assert len(p) == 400
    distinct = {tuple(sorted(d.items())) for d in p}
    assert 0.03 <= 1 - len(distinct) / len(p) <= 0.07


def test_payloads_are_deidentified(synthetic_csv):
    raw = pd.read_csv(synthetic_csv, low_memory=False)
    raw = raw[raw["register_date"].notna()].drop_duplicates("session_id").set_index("session_dt")
    for d in build_payloads(synthetic_csv, n=50, repeat_rate=0.0, seed=2):
        src = raw.loc[d["session_dt"]]
        src = src.iloc[0] if isinstance(src, pd.DataFrame) else src
        assert d["lname"] != src["lname"] and d["cellphone"] != int(src["cellphone"])
        assert str(d["cellphone"])[:3] == str(int(src["cellphone"]))[:3]  # area code kept: it's a model feature
        assert len(d["fname"]) == len(str(src["fname"])) and len(d["lname"]) == len(str(src["lname"]))


def test_payloads_have_request_fields_only(synthetic_csv):
    d = build_payloads(synthetic_csv, n=1, seed=3)[0]
    assert "client_name" not in d and "payout" not in d and "address" not in d
    assert {"register_date", "cellphone", "credit_score"} <= d.keys() and set(d) <= set(HEADER)
