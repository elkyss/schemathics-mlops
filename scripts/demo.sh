#!/usr/bin/env bash
# End-to-end demo of the stack. Stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")/.."
API=http://localhost:8000
step() { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✔ %s\033[0m\n' "$*"; }

[[ -f data/bl_full_data.csv ]] || { echo "data/bl_full_data.csv missing — download it from the Drive link in README"; exit 1; }
# Local TabPFN (the default) holds ~5.5 GB per process; with the API and a trainer running side by side it needs
# ~20 GB of Docker memory. On smaller Docker hosts the demo uses the hosted API (token) or the fallback instead.
has_token=false; grep -qs '^TABPFN_TOKEN=.\+' .env && has_token=true
backend=${TABPFN_BACKEND:-$(grep -s '^TABPFN_BACKEND=' .env | cut -d= -f2 || true)}; backend=${backend:-local}
docker_gb=$(( $(docker info --format '{{.MemTotal}}') / 1024 / 1024 / 1024 ))
if [[ "$backend" == local && $docker_gb -lt 20 ]]; then
  backend=$([[ $has_token == true ]] && echo api || echo fallback)
  echo "Docker has ${docker_gb} GB memory; local TabPFN needs ~20 GB for this demo → using '$backend' instead."
  echo "(Raise Docker Desktop memory to 20 GB+ to demo the default local backend.)"
fi
export TABPFN_BACKEND=$backend
case "$backend" in
  api)      echo "Payout model = hosted TabPFN: this demo spends about 8 API predictions (~80k of 20M monthly credits)." ;;
  local)    echo "Payout model = local TabPFN on CPU (default): weights download once with your token; no API credits." ;;
  *)        echo "Payout model = fallback (brand mean payout): no TABPFN_TOKEN in .env. Everything else is identical." ;;
esac

step "1/8 start MLflow, API, scheduler";             make -s up;          ok "MLflow http://localhost:5001 · API $API"
step "2/8 load data into the training table";        make -s load-data;   ok "table loaded"
step "3/8 researcher train/test evaluation";         make -s train-test;  ok "metrics + researcher log in MLflow"
step "4/8 production model on data up to 2026-02-04"; make -s train AS_OF=2026-02-04; ok "v1 = @champion"
step "5/8 predict";                                  make -s predict
step "6/8 weekly retrain on all data (gate vs v1)";  make -s train;       ok "see gate decision above"
step "7/8 predict with the new champion";            make -s predict
step "8/8 rollback (no API restart)";                make -s rollback
curl -s "$API/version"; echo
ok "demo complete — open http://localhost:5001 (experiments bl-ranking/*, model bl_ranker)"
