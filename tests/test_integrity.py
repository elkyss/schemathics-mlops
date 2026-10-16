import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_original_scripts_match_upstream_sha():
    """The researcher's scripts must stay byte-identical to DYK-Schemathics/Candidates@dev."""
    sums = dict(reversed(line.split()) for line in (ROOT / "reference" / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == {"bl_models_train.py", "bl_exp_payout_predictor.py"}
    for name, sha in sums.items():
        assert hashlib.sha256((ROOT / "src" / "bl_ranking" / "original" / name).read_bytes()).hexdigest() == sha


def test_all_source_files_are_tracked_by_git():
    """A broad .gitignore rule once hid src/bl_ranking/data/ — the fresh clone could not start the API."""
    import shutil
    import subprocess

    import pytest

    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split())
    dirs = ("src", "tests", "load_test", "scripts")
    on_disk = {str(p.relative_to(ROOT)) for d in dirs for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts}
    assert on_disk - tracked == set()
