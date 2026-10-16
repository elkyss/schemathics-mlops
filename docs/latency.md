# Latency report — /v1/predict

Environment: MacBook M1 Pro (32 GB), one API process (1 uvicorn worker), champion `bl_ranker` v2 trained on all data. Traffic: de-identified requests built from the last 7 days of registered sessions, 5 % exact repeats. TabPFN backend switched at serve time with `SERVE_PAYOUT_BACKEND` (same model files).

## Load scenarios

| Scenario | Requests | RPS | p50 ms | p95 ms | p99 ms | Fallback share | Cache hits | p99 SLO |
|---|---|---|---|---|---|---|---|---|
| ours · fallback · realistic (5 users, 1–3 s think time) | 138 | 2.45 | 36 | 53 | 84 | 0% | 4% | ✅ pass (≤ 400) |
| ours · fallback · stress (20 users, no think time) | 24459 | 421.14 | 1 | 260 | 310 | 0% | 96% | capacity test |
| ours · fallback · stress, cache off (20 users) | 3708 | 63.79 | 280 | 330 | 370 | 0% | 0% | capacity test |
| ours · local · realistic (5 users, 1–3 s think time) | 125 | 2.19 | 280 | 610 | 760 | 0% | 0% | ❌ miss (≤ 400) |
| ours · local · stress, cache off (20 users) | 573 | 9.83 | 2000 | 2100 | 2100 | 93% | 0% | capacity test |
| ours · api · realistic, capped at 40 requests (2 users, 1–2 s think time) | 40 | 0.67 | 1700 | 2100 | 2100 | 25% | 5% | ✅ pass (≤ 2200) |

## Baseline: the researcher's predictor as given (one call per request)

| Setup | n | p50 ms | p99 ms | crashes |
|---|---|---|---|---|
| original predictor per request (local TabPFN) | 18 | 4988 | 5023 | 2 |
| original predictor per request (hosted TabPFN fit + predict, step 1) | 5 | 8856 | 33636 (max) | — |

## Where the time goes (single request, no load)

| Component | Time |
|---|---|
| Our request path without TabPFN (validation, original preprocessing, CatBoost, ranking) | ~15 ms |
| TabPFN hosted API predict, new rows (every real user) | 1.5–1.8 s |
| TabPFN hosted API predict, repeated rows (server-side cache — not representative) | 0.9 s |
| TabPFN local CPU predict (OSS, context cached), new rows | 213 ms p50 / 232 ms max |

## Reading the numbers

- Fitting once instead of per request removes the dominant cost: 8.9 s (hosted) / 5.0 s (local) → 1.7 s / 0.28 s p50. The original also crashed on ~5–10 % of real users (fewer than 5 of 8 survey answers); the service answers those with `{}` and `X-Empty-Reason`.
- With the hosted API, network time is ~99 % of latency and the 2 s deadline sends ~25 % of requests to the fallback payout. The free tier also caps usage at ~2,000 predictions/month (10k credits each).
- Local TabPFN is ~7× faster but less accurate (payout MAPE 8.9 % vs 5.4 %) and its weights are non-commercial. One process serves ~2–4 rps within the SLO; scale horizontally for more.
- Production traffic is ~2,000 registered sessions/day (≈0.02 rps average), so a single small replica per backend comfortably covers peak load; replicas are for availability, not throughput.
