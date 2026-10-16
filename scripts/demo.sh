#!/usr/bin/env bash
# End-to-end demo of the stack. Stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")/.."
API=http://localhost:8000
step() { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✔ %s\033[0m\n' "$*"; }

[[ -f data/bl_full_data.csv ]] || { echo "data/bl_full_data.csv missing — download it from the Drive link in README"; exit 1; }
if grep -qs '^TABPFN_TOKEN=.\+' .env; then
  echo "TABPFN_TOKEN found: payout model = hosted TabPFN. This demo spends ~6 API predictions (~60k of the"
  echo "free tier's 20M monthly credits)."
else
  echo "No TABPFN_TOKEN in .env: payout model = fallback (brand mean payout). Everything else is identical."
fi

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
