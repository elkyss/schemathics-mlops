import fcntl

from typer.testing import CliRunner

from bl_ranking.cli import app

runner = CliRunner()


def test_load_data_missing_csv_exit_1(tmp_path, monkeypatch):
    monkeypatch.setenv("WAREHOUSE_DIR", str(tmp_path))
    result = runner.invoke(app, ["load-data", "--csv", str(tmp_path / "nope.csv")])
    assert result.exit_code == 1 and "CSV not found" in result.output and "README" in result.output


def test_train_lock_exit_2(tmp_path, monkeypatch):
    monkeypatch.setenv("WAREHOUSE_DIR", str(tmp_path))
    lock = (tmp_path / "train.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = runner.invoke(app, ["train", "--mode", "train_test"])
        assert result.exit_code == 2 and "already running" in result.output
    finally:
        lock.close()


def test_help_is_fast_and_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("load-data", "train", "promote", "rollback", "status", "serve"):
        assert cmd in result.output
