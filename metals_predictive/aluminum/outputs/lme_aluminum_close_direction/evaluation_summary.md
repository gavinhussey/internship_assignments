# LME Aluminum Close Direction Model — NIGHT-BEFORE

## Question
Will the **LME 3-month aluminum** settlement close higher tomorrow than today?
`target_up = 1 if lme_al_close[t+1] > lme_al_close[t] else 0`. Honest rebuild replacing the
stale COMEX ALI=F target (see ALUMINUM_STEEL_MODEL_REVIEW.md).

## Data
- Target: `lme_aluminum.csv` (LME 3-month, USD/tonne) — actually trades (~1.4% flat days).
- Drivers: `external_drivers.csv` (COMEX echo column dropped) + cross-asset yfinance cousins.
- Rows: 3051 · Date range: 2014-06-02 → 2026-06-25

## Timing (leak-free)
Decision at the evening of day t (after US close); target is the day-(t+1) LME settle
(~7:15am ET). All day-t features precede it.

## Performance (walk-forward, 2051 OOS days)
- **Accuracy: 0.5163** (95% CI [0.4934, 0.5393])
- AUC: 0.5194 · Balanced accuracy: 0.5167
- Up base rate: 0.4929 (always-guess-majority = 0.5071)
- Up precision/recall: 0.509 / 0.546
- Down precision/recall: 0.525 / 0.487
- Confusion (tn,fp,fn,tp): [507, 533, 459, 552]

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5163 | [0.493, 0.539] |
| 0.03 | 74.3% | 0.5125 | [0.484, 0.540] |
| 0.05 | 58.3% | 0.5188 | [0.485, 0.551] |
| 0.08 | 42.6% | 0.5292 | [0.489, 0.566] |

Best confident slice: **0.5292 on 42.6% of days**
(min |p-0.5| > 0.08).

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 161 | 0.4969 | 0.4926 |
| 2019 | 253 | 0.5178 | 0.5571 |
| 2020 | 254 | 0.5512 | 0.5225 |
| 2021 | 253 | 0.5296 | 0.4811 |
| 2022 | 251 | 0.498 | 0.4823 |
| 2023 | 251 | 0.506 | 0.5184 |
| 2024 | 254 | 0.4646 | 0.4661 |
| 2025 | 253 | 0.5336 | 0.5396 |
| 2026 | 121 | 0.5702 | 0.582 |

## Model
- L1 logistic + isotonic calibration, expanding walk-forward (machinery from
  aluminum_close_direction_model). 24 features selected (C=0.1)
  of 165 candidates.
- Top features: DBB_div_5d, brent_return_1d, log_return_1d, XME_div_20d, ttf_ret, VIX_vol_20d, VIX_ret_1d, PICK_div_1d

---
*Research model — not financial advice.*
