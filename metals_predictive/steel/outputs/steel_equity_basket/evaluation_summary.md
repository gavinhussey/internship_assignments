# Steel Equity Basket Direction Model — NIGHT-BEFORE

## Question
Will the equal-weight NUE+STLD+MT steel equity basket close higher tomorrow?
`target_up = 1 if basket_return[t+1] > 0 else 0`. Liquid-target rebuild replacing
the illiquid HRC=F futures target (see ALUMINUM_STEEL_MODEL_REVIEW.md §10).

## Why a basket instead of HRC=F
HRC=F (Yahoo): 35.6% flat days, 47% no-range, median vol ~11 contracts. Any model
runs below the 64.1% "always-flat" baseline. NUE+STLD+MT: genuinely liquid, balanced
~51% up-rate, free on yfinance, ~0.75 corr with HRC direction.

## Target construction
Equal-weight daily return: `basket_ret[t] = mean(NUE_ret[t], STLD_ret[t], MT_ret[t])`.
Basket price index: `(1 + basket_ret).cumprod() * 100` (starts 2014). Target is sign
of the next-day basket return — identical to close[t+1] > close[t] on the index.

## Timing (leak-free)
Night-before: decision at day-t US close (~4pm ET). All features use day-t data.
Target is day-(t+1) basket close — all three stocks trade 9:30am–4pm ET next day.

## Data
- Basket: NUE (Nucor), STLD (Steel Dynamics), MT (ArcelorMittal) equal-weight
- Rows: 3146 · Date range: 2014-06-02 → 2026-06-26

## Performance (walk-forward, 2146 OOS days)
- **Accuracy: 0.5042** (95% CI [0.4837, 0.5253])
- AUC: 0.5128 · Balanced accuracy: 0.5044
- Up base rate: 0.4893 (always-guess-majority = 0.5107)
- Up precision/recall: 0.494 / 0.513
- Down precision/recall: 0.515 / 0.495

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5042 | [0.484, 0.525] |
| 0.03 | 64.8% | 0.5223 | [0.495, 0.551] |
| 0.05 | 48.0% | 0.5204 | [0.489, 0.554] |
| 0.08 | 33.0% | 0.5106 | [0.472, 0.552] |

Best confident slice: **0.5223 on 64.8% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 196 | 0.5153 | 0.535 |
| 2019 | 261 | 0.4943 | 0.5033 |
| 2020 | 262 | 0.4618 | 0.4517 |
| 2021 | 261 | 0.4943 | 0.4724 |
| 2022 | 260 | 0.5231 | 0.5362 |
| 2023 | 260 | 0.4615 | 0.4674 |
| 2024 | 262 | 0.542 | 0.4888 |
| 2025 | 258 | 0.5426 | 0.5531 |
| 2026 | 126 | 0.5079 | 0.5328 |

## Model
- 76 features selected (L1, C=0.5) of
  144 candidates.
- Top features: MT_ratio_mom20, RS_div_5d, momentum_10d, WOR_div_1d, PKX_div_z20, FMG_AX_div_5d, STLD_div_20d, WOR_div_z20, VIX_mom_20d, PKX_div_1d

---
*Research model — not financial advice.*
