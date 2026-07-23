# Copper Close Direction Model — MORNING-OF variant

## What this is
Predicts whether COMEX copper closes higher tomorrow than today, but the guess is made on
the **morning of the target day**, after the Asian/Australian session closes (~1-4am ET)
and before the COMEX settle (~1pm ET). That lets it use the **overnight Asian copper-miner
moves**, which lead copper's US session.

> If you must lock your guess at the prior close instead, use the NIGHT-BEFORE model
> (`copper_close_direction_model.py`, ~53.9%). This one is a ~5-hour forecast, not 24h.

## Overnight feature (leak-free)
Asian/AU miners (BHP.AX, RIO.AX, S32.AX, Zijin 2899.HK, Jiangxi 0358.HK, MMG 1208.HK)
session return of date t+1, encoded as the date-aligned series shifted by one trading day.
Asian close precedes the COMEX t+1 settle, so it is available at prediction time.
Validated: permutation null passes (real 0.864 vs null ~0.50, p=0.000); a
no-shift control (same-day Asian return) scores 0.5292
vs real 0.8641, isolating the gain to the t+1 session shift.

## Performance (walk-forward, 1884 OOS days)
- **Accuracy: 0.8641** (95% CI [0.8452, 0.8827])
- **AUC: 0.9351** · Balanced accuracy: 0.8638
- Up precision/recall: 0.863 / 0.875
- Down precision/recall: 0.866 / 0.853
- vs night-before baseline ~0.539 -> a ++0.325 jump.

## Per-year stability (the standout — consistent every year)
| year | n | accuracy | auc |
|---|---|---|---|
| 2019 | 252 | 0.8294 | 0.9213 |
| 2020 | 253 | 0.8972 | 0.9444 |
| 2021 | 252 | 0.8532 | 0.9467 |
| 2022 | 251 | 0.8685 | 0.9327 |
| 2023 | 251 | 0.9203 | 0.9683 |
| 2024 | 252 | 0.877 | 0.929 |
| 2025 | 252 | 0.7897 | 0.8665 |
| 2026 | 118 | 0.8898 | 0.9749 |

## Model
- 21 features selected (L1, per-fold C=0.1) of
  197 candidates; copper + cross-asset divergence + overnight Asian miners
  + COMEX open gap (LME proxy) + direct LME overnight.
- Top features: lme_overnight_ret, lme_overnight_z20, next_gap_z20, next_gap_return, log_return_10d, lme_comex_basis_chg, drawdown_20d, CNY_ret_1d

---
*Research model — not financial advice. The morning-of timing is essential to its validity.*
