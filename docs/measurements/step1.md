# Step 1 measurements (2026-10-16, MacBook M1 Pro 32 GB, Python 3.12.13)

## Original training code, unchanged (train_test mode)
| Stage | Seconds |
|---|---|
| import_module_incl_NameDataset | 8.74 |
| import_preprocess | 1.65 |
| impute_survey_columns | 0.15 |
| credit_score_to_numeric | 0.1 |
| credit_loan_amount_to_numeric | 0.11 |
| monthly_revenue_to_numeric | 0.1 |
| time_in_business_to_num | 0.01 |
| time_features | 0.02 |
| additional_features | 5.49 |
| split_by_time | 0.04 |
| catbosot_model_sold_to_client | 29.24 |
| tabpfn_regression_payout | 10.7 |
| accuracy_classification_model | 0.13 |
| accuracy_cont_payout_prediction | 9.6 |
| fit_total | 57.42 |

Peak RSS: **3.15 GB** (NameDataset ≈ 2.2 GB of that, loaded at import).
Baseline metrics (researcher log): sold_to_client F1(class 1) 0.47 overall; payout MAPE 5.4 %, MAE $2.34.
Test-week data gap: day 4 has 23 rows, day 5 has 3 rows → per-day metrics on those days are noise.

## TabPFN API cost
Free tier: 20M credits/month. **predict ≈ 10,000 credits per call** (independent of 10 vs 4,000 rows); **fit is free**.
→ ~2,000 API predictions/month on the free tier; 1M predictions ≈ 10B credits.

## TabPFN serving latency (real production context, 1,000 rows × 25 cols; query = 10 rows)
| Option | p50 | max | n |
|---|---|---|---|
| A. original: fit + predict per request | 8856 ms | 33636 ms | 5 |
| B. fit once, predict only | 941 ms | 1386 ms | 5 |
| C. save_model → load_model → predict | 859 ms | 863 ms | 2 |

Single API fit: 7.07 s. load_model: 1 ms from a 762-byte JSON record.
Local OSS TabPFN: pending licence acceptance on ux.priorlabs.ai.

## Local OSS TabPFN-3.5 (licence accepted; no credits)
| Backend | predict p50 | max | fit |
|---|---|---|---|
| API (fit once) | 941 ms | 1386 ms | 7.1 s (free) |
| **Local CPU, fit_with_cache** | **218 ms** | **288 ms** | 31.9 s (incl. first weight download) |
| Local MPS (Apple GPU) | 543 ms | 548 ms | 6.4 s |

## Payout accuracy, same original train_test code, only the TabPFN backend swapped
| Backend | MAPE | MAE |
|---|---|---|
| API (TabPFN-3.5 hosted) | **5.4 %** | **$2.34** |
| Local CPU (TabPFN-3.5 OSS) | 8.9 % | $4.00 |

Likely cause: local treats the high-cardinality string columns `city` (513 distinct in the 1,000-row context)
and `cellphone_prefix` (267) as text; the hosted API embeds text natively.
Also: the 1,000-row context contains only **4 of the 10 brands** → payouts for the other 6 are extrapolated.
