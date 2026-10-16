import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_original_scripts_match_upstream_sha():
    """The researcher's scripts must stay byte-identical to DYK-Schemathics/Candidates@dev."""
    sums = dict(reversed(line.split()) for line in (ROOT / "reference" / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == {"bl_models_train.py", "bl_exp_payout_predictor.py"}
    for name, sha in sums.items():
        assert hashlib.sha256((ROOT / "src" / "bl_ranking" / "original" / name).read_bytes()).hexdigest() == sha
