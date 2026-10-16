from pathlib import Path

import pytest

from tests.fixtures.synthetic import make_synthetic_csv

ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = ROOT / "data" / "bl_full_data.csv"


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    return make_synthetic_csv(tmp_path_factory.mktemp("synthetic") / "bl_full_data.csv")
