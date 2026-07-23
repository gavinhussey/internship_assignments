# Iron Ore (SGX TIO=F) Direction Model — MORNING-OF

## What this is
Predicts whether SGX 62% Fe iron ore closes higher tomorrow, guessed after the China ferrous
complex (SHFE rebar/HRC, DCE iron ore and coking coal) closes at ~3pm Beijing / 2am ET on
day t+1 — before the TIO=F final settlement at ~5-6am ET. See iron_ore_direction_model.py
for the night-before baseline (~51.9%).

## No-shift control (the honesty check)
Real model acc 0.5501 / AUC 0.5827.
No-shift control acc 0.5202 / AUC 0.5408.
**Overnight shift lift: +0.0300 acc / +0.0419 AUC.**
Only a **positive, meaningful lift** makes the morning timing worthwhile.
If lift ≤ 0, the China ferrous "overnight" data is contemporaneous with TIO=F — do not deploy.

## Performance (walk-forward, 2034 OOS days)
- **Accuracy: 0.5501** (95% CI [0.5206, 0.5777])
- AUC: 0.5827 · Balanced accuracy: 0.5467
- Up base rate: 0.5167 (always-guess-majority = 0.5167)
- Up precision/recall: 0.555 / 0.651
- Down precision/recall: 0.542 / 0.443

## Deployable confidence gate
| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5501 | [0.521, 0.578] |
| 0.03 | 84.8% | 0.5615 | [0.530, 0.590] |
| 0.05 | 79.0% | 0.5710 | [0.537, 0.601] |
| 0.08 | 65.4% | 0.5913 | [0.554, 0.626] |

Best confident slice: **0.5913 on 65.4% of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 153 | 0.5752 | 0.5727 |
| 2019 | 252 | 0.5119 | 0.4784 |
| 2020 | 253 | 0.5455 | 0.6036 |
| 2021 | 252 | 0.5238 | 0.5782 |
| 2022 | 251 | 0.5936 | 0.6657 |
| 2023 | 250 | 0.552 | 0.6089 |
| 2024 | 252 | 0.5913 | 0.5861 |
| 2025 | 251 | 0.5936 | 0.5988 |
| 2026 | 120 | 0.3917 | 0.3852 |

## Model
- 18 features selected (L1, C=0.05) of 162
  candidates; iron ore price block + cross-asset divergence + China ferrous overnight leads.
- Top features: shfe_rb_ret_z20, shfe_hrc_mom5, CNY_vol_20d, dce_io_ret, SID_div_20d, shfe_hrc_ret_z20, VALE_div_5d, shfe_rb_mom5

## China ferrous features (Bloomberg required)
Populated by bloombergSteel.py. If overnight shift lift ≤ 0, the features are not leading
TIO=F and this model should be retracted in favour of iron_ore_direction_model.py.

---
*Research model — not financial advice. Valid only if overnight shift lift is positive.*
