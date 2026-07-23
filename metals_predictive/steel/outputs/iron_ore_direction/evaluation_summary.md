# Iron Ore (SGX 62% Fe, TIO=F) Direction Model — NIGHT-BEFORE

## Question
Will SGX iron ore close higher tomorrow than today? The liquid ferrous-benchmark rebuild
(steel's LME analog) replacing the thin HRC=F target. See ALUMINUM_STEEL_MODEL_REVIEW.md.

## Data
- Target: TIO=F (SGX 62% Fe iron ore, yfinance) — liquid, ~2.1% flat days.
- Drivers: cross-asset ferrous cousins (steelmakers, iron-ore miners, ETFs) + macro.
- Rows: 3034 · Date range: 2014-06-02 → 2026-06-25

## Timing (leak-free)
Night-before: decision at day-t US close; target is the day-(t+1) SGX settle. All day-t
features precede it.

## Performance (walk-forward, 2034 OOS days)
- **Accuracy: 0.5202** (95% CI [0.4891, 0.5517])
- AUC: 0.5408 · Balanced accuracy: 0.5198
- Up base rate: 0.5167 (always-guess-majority = 0.5167)
- Up precision/recall: 0.536 / 0.530
- Down precision/recall: 0.504 / 0.510

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5202 | [0.489, 0.552] |
| 0.03 | 90.7% | 0.5257 | [0.495, 0.558] |
| 0.05 | 81.0% | 0.5331 | [0.498, 0.567] |
| 0.08 | 66.1% | 0.5331 | [0.495, 0.572] |

Best confident slice: **0.5331 on 81.0% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 153 | 0.5033 | 0.5488 |
| 2019 | 252 | 0.5079 | 0.4679 |
| 2020 | 253 | 0.5296 | 0.5853 |
| 2021 | 252 | 0.5516 | 0.5426 |
| 2022 | 251 | 0.5737 | 0.5756 |
| 2023 | 250 | 0.54 | 0.5539 |
| 2024 | 252 | 0.4762 | 0.5095 |
| 2025 | 251 | 0.5498 | 0.5791 |
| 2026 | 120 | 0.3583 | 0.3814 |

## Model
- 33 features selected (L1, C=0.1) of 167
  candidates. Top: VALE_div_5d, CNY_vol_20d, shfe_hrc_mom5, sgx_io_mom5, DBB_div_1d, dce_io_div, TX_div_20d, VALE_div_1d

---
*Research model — not financial advice.*
