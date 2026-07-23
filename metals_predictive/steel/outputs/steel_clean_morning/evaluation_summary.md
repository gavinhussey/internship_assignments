# Steel (CME HRC) MORNING-OF — de-leaked rebuild

## What changed
Removed the `next_gap_return` open-gap leak that inflated the original 73.4% (it was a
tautology on the thin HRC=F contract). Overnight leads are now only genuinely-leading
sessions: Asian/EU steelmakers + SGX iron ore (TIO=F), all closing before the 1pm ET settle.

## No-shift control (the honesty check)
Real model acc 0.6072 / AUC 0.5593.
No-shift control acc 0.6290 / AUC 0.5652.
**Overnight shift lift: -0.0217 acc / -0.0059 AUC.**
Only a positive, meaningful lift makes the morning timing worthwhile.

## Performance (walk-forward, 2024 OOS days)
- **Accuracy: 0.6072** (95% CI [0.5766, 0.6369])
- AUC: 0.5593 · Balanced accuracy: 0.5255
- Up base rate: 0.3592 (always-guess-majority = 0.6408)
- Up precision/recall: 0.417 / 0.235
- Down precision/recall: 0.656 / 0.816

> CAVEAT: HRC=F is still thin — 35.6% of close-to-close days are exactly flat (bucketed into
> DOWN), which inflates the majority baseline to ~0.64. A genuine
> edge must clear THAT, not 0.50.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 143 | 0.5874 | 0.4788 |
| 2019 | 252 | 0.75 | 0.4467 |
| 2020 | 252 | 0.6071 | 0.5574 |
| 2021 | 252 | 0.5159 | 0.5176 |
| 2022 | 251 | 0.5697 | 0.5154 |
| 2023 | 250 | 0.596 | 0.6067 |
| 2024 | 252 | 0.7143 | 0.4777 |
| 2025 | 251 | 0.5179 | 0.4783 |
| 2026 | 121 | 0.5868 | 0.5399 |

## Model
- 27 features selected (L1, C=0.1) of 172
  candidates; HRC price block + cross-asset divergence + leak-free overnight steelmakers + iron ore.
- Top features: close_location, drawdown_20d, dow_sin, TNX_ret_1d, volatility_5d, FMG_AX_div_z20, CMC_ratio_mom20, 600019_SS_overnight_ret

---
*Research model — not financial advice.*
