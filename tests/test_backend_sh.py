"""scripts/backend.sh decides the payout backend for every Docker entry point (make demo/up/train/train-test)."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def run(tmp_path, env_file: str, docker_mem_bytes: int | None, shell_env: dict | None = None):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    docker = bindir / "docker"
    if docker_mem_bytes is None:
        docker.write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    else:
        docker.write_text(f"#!/bin/sh\necho {docker_mem_bytes}\n")
    docker.chmod(0o755)
    (tmp_path / ".env").write_text(env_file)
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path), **(shell_env or {})}
    script = f'cd "{tmp_path}" && source "{ROOT}/scripts/backend.sh" && echo "RESULT=$TABPFN_BACKEND"'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)


def result(proc):
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1].removeprefix("RESULT=")


@pytest.mark.parametrize("env_file,mem,expected", [
    ("TABPFN_TOKEN=abc\n", 20 * GIB, "local"),                       # default local, big Docker
    ("TABPFN_TOKEN=abc\n", 19.4 * GIB, "local"),                     # "20 GB" in Docker Desktop reports ~19.x
    ("TABPFN_TOKEN=abc\n", 7.6 * GIB, "api"),                        # small Docker + token → api
    ("", 7.6 * GIB, "fallback"),                                     # small Docker, no token → fallback
    ("TABPFN_TOKEN=\n", 32 * GIB, "fallback"),                       # empty token is no token
    ("TABPFN_TOKEN=abc\nTABPFN_BACKEND=auto\n", 7.6 * GIB, "api"),   # auto == local → memory check applies
    ('TABPFN_TOKEN=abc\nTABPFN_BACKEND="local" # cpu\r\n', 7.6 * GIB, "api"),  # quotes, comment, CRLF
    ("TABPFN_TOKEN=abc\nTABPFN_BACKEND=api\n", 7.6 * GIB, "api"),
    ("TABPFN_TOKEN=abc\nTABPFN_BACKEND=fallback\n", 64 * GIB, "fallback"),
])
def test_effective_backend(tmp_path, env_file, mem, expected):
    assert result(run(tmp_path, env_file, int(mem))) == expected


def test_shell_env_overrides_dotenv(tmp_path):
    proc = run(tmp_path, "TABPFN_TOKEN=abc\nTABPFN_BACKEND=api\n", 64 * GIB, {"TABPFN_BACKEND": "local"})
    assert result(proc) == "local"


def test_invalid_backend_is_an_error(tmp_path):
    proc = run(tmp_path, "TABPFN_BACKEND=gpu\n", 64 * GIB)
    assert proc.returncode != 0 and "gpu" in proc.stderr


def test_docker_not_running_is_a_clear_error(tmp_path):
    proc = run(tmp_path, "TABPFN_TOKEN=abc\n", None)
    assert proc.returncode != 0 and "Docker" in proc.stderr and "operand" not in proc.stderr


def test_reason_is_printed(tmp_path):
    proc = run(tmp_path, "TABPFN_TOKEN=abc\n", int(7.6 * GIB))
    assert "local TabPFN needs" in proc.stderr and "api" in proc.stderr


def test_too_little_docker_memory_warns(tmp_path):
    """Below ~8 GB the stack itself (API + trainer + MLflow) gets OOM-killed with a bare exit 137."""
    proc = run(tmp_path, "TABPFN_TOKEN=abc\nTABPFN_BACKEND=api\n", 4 * GIB)
    assert result(proc) == "api" and "at least 8 GB" in proc.stderr


def test_enough_docker_memory_no_warning(tmp_path):
    proc = run(tmp_path, "TABPFN_TOKEN=abc\nTABPFN_BACKEND=api\n", int(7.6 * GIB))
    assert "at least 8 GB" not in proc.stderr
