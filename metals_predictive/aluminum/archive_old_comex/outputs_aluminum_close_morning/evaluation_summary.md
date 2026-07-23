# Aluminum Close Direction Model — MORNING-OF variant

## What this is
Predicts whether the US-session aluminum benchmark closes higher tomorrow than today, but
the guess is made on the **morning of the target day**, after the LME aluminum settlement
(~7am ET) and the Asian/Australian session (~1-4am ET) close, and before the US settle
(~1pm ET). That lets it use the **overnight LME aluminum + Asian aluminum-producer moves**,
which lead the US session. The LME lead is the single biggest expected edge.

> If you must lock your guess at the prior close instead, use the NIGHT-BEFORE model
> (`aluminum_close_direction_model.py`). This one is a ~5-hour forecast, not 24h.

## Overnight features (leak-free)
LME aluminum Ring settlement of date t+1 (the primary lead) plus Asian/AU producers
(BHP.AX, RIO.AX, S32.AX, Hongqiao 1378.HK, Chalco 2600.HK, Rusal 0486.HK) session return of
date t+1, encoded as the date-aligned series shifted by one trading day. All settle before
the US t+1 close, so they are available at prediction time.
Validated: a no-shift control (same-day return) scores 0.9360
vs real 0.9280, isolating the gain to the t+1 session shift.

## Performance (walk-forward, 1999 OOS days)
- **Accuracy: 0.9280** (95% CI [0.9037, 0.9490])
- **AUC: 0.9650** · Balanced accuracy: 0.9253
- Up precision/recall: 0.932 / 0.902
- Down precision/recall: 0.925 / 0.948

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 117 | 0.9573 | 0.4813 |
| 2019 | 252 | 0.8056 | 0.849 |
| 2020 | 253 | 0.8538 | 0.9391 |
| 2021 | 252 | 0.9643 | 0.9826 |
| 2022 | 251 | 0.9761 | 0.9755 |
| 2023 | 250 | 0.98 | 0.9859 |
| 2024 | 252 | 0.9127 | 0.9543 |
| 2025 | 251 | 0.9761 | 0.982 |
| 2026 | 121 | 0.9587 | 0.9784 |

## Model
- 4 features selected (L1, per-fold C=0.05) of
  196 candidates; aluminum + cross-asset divergence + overnight Asian
  producers + US open gap (LME proxy) + direct LME overnight.
- Top features: next_gap_z20, lme_overnight_ret, next_gap_return, asianminers_trend_5_20

---
*Research model — not financial advice. The morning-of timing is essential to its validity.*
