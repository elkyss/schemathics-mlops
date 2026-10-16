#!/usr/bin/env bash
# Run a command with TABPFN_BACKEND decided by scripts/backend.sh (token, .env, Docker memory).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/backend.sh
exec "$@"
