#!/usr/bin/env bash
# Source me: decides and exports TABPFN_BACKEND for every Docker entry point (make demo/up/train/train-test).
#
# Same rule as bl_ranking.payout.backends.resolve_backend, plus the Docker memory check:
#   requested = shell TABPFN_BACKEND > .env TABPFN_BACKEND > local   ("auto" means local)
#   local/api without a TABPFN_TOKEN            -> fallback (brand mean payout)
#   local with < ~18 GiB Docker memory          -> api if a token is set, else fallback
# Local TabPFN holds ~5.5 GB per process (context cache); the API and a trainer side by side need ~20 GB
# Docker memory (Docker Desktop's "20 GB" reports ~19.x GiB, hence the 18 GiB threshold).

_bl_dotenv() {  # value of KEY in ./.env with quotes, inline comment, CR and whitespace removed
  [[ -f .env ]] || return 0
  grep -E "^[[:space:]]*$1=" .env | tail -1 | cut -d= -f2- | tr -d '\r' | sed -E 's/[[:space:]]+#.*$//' \
    | sed -E "s/^[[:space:]]*['\"]?//; s/['\"]?[[:space:]]*$//" || true
}

_bl_resolve_backend() {
  local requested token mem_bytes mem_gib min_bytes=$((18 * 1024 * 1024 * 1024)) effective reason=""
  requested=${TABPFN_BACKEND:-$(_bl_dotenv TABPFN_BACKEND)}
  requested=$(printf '%s' "$requested" | tr -d '\r"'"'"'[:space:]')
  requested=${requested:-local}
  [[ $requested == auto ]] && requested=local
  case "$requested" in local|api|fallback) ;; *)
    echo "TABPFN_BACKEND='$requested' is not one of: local, api, fallback" >&2; return 1 ;;
  esac
  if mem_bytes=$(docker info --format '{{.MemTotal}}' 2>/dev/null) && [[ $mem_bytes =~ ^[0-9]+$ ]] \
      && (( mem_bytes < 7 * 1024 * 1024 * 1024 )); then  # Docker Desktop's "8 GB" reports ~7.6 GiB
    echo "warning: Docker has ~$(( mem_bytes / 1024 / 1024 / 1024 )) GB memory; this stack needs at least 8 GB" \
         "(Docker Desktop → Settings → Resources), or training/serving will be killed (exit 137)." >&2
  fi
  token=${TABPFN_TOKEN:-$(_bl_dotenv TABPFN_TOKEN)}
  effective=$requested
  if [[ $effective != fallback && -z $token ]]; then
    effective=fallback; reason="no TABPFN_TOKEN"
  fi
  if [[ $effective == local ]]; then
    if ! mem_bytes=$(docker info --format '{{.MemTotal}}' 2>/dev/null) || [[ ! $mem_bytes =~ ^[0-9]+$ ]]; then
      echo "Docker isn't running (or isn't reachable): start Docker Desktop and retry." >&2; return 1
    fi
    if (( mem_bytes < min_bytes )); then
      mem_gib=$(( (mem_bytes + 512 * 1024 * 1024) / (1024 * 1024 * 1024) ))
      effective=api
      reason="Docker has ~${mem_gib} GB; local TabPFN needs ~20 GB here (raise it in Docker Desktop → Resources)"
    fi
  fi
  export TABPFN_BACKEND=$effective
  if [[ $effective != "$requested" ]]; then
    echo "payout backend: $effective instead of $requested — $reason" >&2
  else
    echo "payout backend: $effective" >&2
  fi
}

_bl_resolve_backend
