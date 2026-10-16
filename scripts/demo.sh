#!/usr/bin/env bash
# End-to-end demo of the stack. Stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")/.."
API=http://localhost:8000
step() { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✔ %s\033[0m\n' "$*"; }

[[ -f data/bl_full_data.csv ]] || { echo "data/bl_full_data.csv missing — download it from the Drive link in README"; exit 1; }
source scripts/backend.sh   # decides TABPFN_BACKEND (token, .env, Docker memory) and says why
case "$TABPFN_BACKEND" in
  api)   echo "Payout model = hosted TabPFN: this demo spends about 8 API predictions (~80k of 20M monthly credits)." ;;
  local) echo "Payout model = local TabPFN on CPU: weights download once with your token; no API credits." ;;
  *)     echo "Payout model = fallback (mean payout per brand). Everything else is identical." ;;
esac

step "1/8 start MLflow, API, scheduler";             make -s up;          ok "MLflow http://localhost:5001 · API $API"
step "2/8 load data into the training table";        make -s load-data;   ok "table loaded"
step "3/8 researcher train/test evaluation";         make -s train-test;  ok "metrics + researcher log in MLflow"
step "4/8 production model on data up to 2026-02-04"; make -s train AS_OF=2026-02-04; ok "v1 = @champion"
step "5/8 predict";                                  make -s predict
served_version() { curl -s "$API/version" | grep -o '"model_version":"[^"]*"'; }
before=$(served_version)
step "6/8 weekly retrain on all data (gate vs the champion)"; make -s train
if [[ "$(served_version)" == "$before" ]]; then
  ok "the gate refused the new version (see reason above): it waits at @challenger"
  echo "  → approving it manually (the human-approval path) to continue the demo"
  make -s promote || { echo "  promotion refused (see above) — skipping the rollback demo"; skip_rollback=1; }
else
  ok "the gate passed: the new version is @champion"
fi
step "7/8 predict with the new champion";            make -s predict
if [[ -z ${skip_rollback:-} ]]; then
  step "8/8 rollback (no API restart)";              make -s rollback
fi
curl -s "$API/version"; echo
ok "demo complete — open http://localhost:5001 (experiments bl-ranking/*, model bl_ranker)"
