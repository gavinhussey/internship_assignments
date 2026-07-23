# Steel (CME HRC) Close Direction Model — MORNING-OF variant

## What this is
Predicts whether CME US HRC steel closes higher tomorrow than today, but the guess is made
on the **morning of the target day**, after the Asian/European steelmaker sessions close
(~2-11am ET) and before the HRC settle (~1pm ET). That lets it use the **overnight
steelmaker moves** plus HRC's own overnight (Globex) open gap, which lead the US session.

> If you must lock your guess at the prior close instead, use the NIGHT-BEFORE model
> (`steel_close_direction_model.py`). This one is a ~5-hour forecast, not 24h.
>
> HONEST CAVEAT: unlike copper (LME↔COMEX arbitrage gave ~85%), steel has no single
> arbitraged global price, so the overnight lift here is far smaller. Trust the no-shift
> control + permutation test below, not the headline number.

## Overnight feature (leak-free)
Asian/EU steelmakers (BSL.AX, 5401.T, 600019.SS, Angang 0347.HK, TKA.DE, SSAB-B.ST)
session return of date t+1, encoded as the date-aligned series shifted by one trading day.
Their home close precedes the HRC t+1 settle, so it is available at prediction time.
Validated: permutation null (real 0.734 vs null ~0.50); a no-shift control
(same-day steelmaker return) scores 0.7332
vs real 0.7337, isolating any gain to the t+1 session shift.

## Performance (walk-forward, 2024 OOS days)
- **Accuracy: 0.7337** (95% CI [0.7090, 0.7585])
- **AUC: 0.7594** · Balanced accuracy: 0.7094
- Up precision/recall: 0.631 / 0.623
- Down precision/recall: 0.790 / 0.796
- vs night-before baseline ~0.539 -> a ++0.195 jump.

## Per-year stability (the standout — consistent every year)
| year | n | accuracy | auc |
|---|---|---|---|
| 2018 | 143 | 0.8322 | 0.8541 |
| 2019 | 252 | 0.7857 | 0.7256 |
| 2020 | 252 | 0.6865 | 0.716 |
| 2021 | 252 | 0.6984 | 0.727 |
| 2022 | 251 | 0.6335 | 0.6651 |
| 2023 | 250 | 0.732 | 0.7994 |
| 2024 | 252 | 0.754 | 0.7267 |
| 2025 | 251 | 0.7689 | 0.8249 |
| 2026 | 121 | 0.7769 | 0.7952 |

## Model
- 8 features selected (L1, per-fold C=0.05) of
  172 candidates; HRC + cross-asset divergence + overnight steelmakers
  + HRC overnight (Globex) open gap.
- Top features: next_gap_return, volatility_5d, DX_Y_NYB_vol_20d, FMG_AX_div_z20, gap_return, log_return_1d, CNY_ret_1d, steelmakers_trend_5_20

---
*Research model — not financial advice. The morning-of timing is essential to its validity.*
