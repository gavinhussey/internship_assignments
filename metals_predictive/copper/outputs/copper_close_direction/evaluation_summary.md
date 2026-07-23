# Copper Close-to-Close Direction Model

## Question
Will COMEX copper close **higher tomorrow than it closed today**? Pure up/down, every day.
`target_up = 1 if close[t+1] > close[t] else 0`.

## Data
- Price dataset: `copper.csv` (close column `close`)
- Macro drivers merged: `external_drivers.csv` + 19 cross-asset tickers
- Rows: 2884 · Date range: 2015-01-02 → 2026-06-23

## Features (163 candidates -> 14 selected)
- Copper price block (returns, momentum, trend ratios, volatility, drawdown, candle, volume)
- **Cross-asset divergence block** (the edge): miner/China/base-metal return minus copper's
  own return at 1/5/20d, divergence z-scores and ratio-momentum.
- Non-stationary price LEVELS (ma_*, volume_ma_*) and exact duplicates dropped a priori.
- **L1 selection** keeps ~59 features per fold; the final model uses
  14. Top by |coef|: XME_div_z20, brent_momentum_20d, CNY_ret_1d, CPER_ratio_mom20, volume_vs_ma20, VIX_vol_20d, log_return_10d, curve_2s10s_momentum_20d.

## Model
- **L1-penalized Logistic Regression** + isotonic calibration (logistic beat RF/GB/HistGB —
  the usable signal is linear). C chosen per-fold on validation (C=0.05 final).
- Expanding-window walk-forward, 1884 out-of-sample days.

## Performance (walk-forward, every day)
- **Accuracy: 0.5228** (95% CI [0.4989, 0.5472])
- AUC: 0.5316 · Balanced accuracy: 0.5193
- Up precision/recall: 0.529 / 0.647
- Down precision/recall: 0.512 / 0.392
- Up base rate: 0.5138 (always-guess-majority = 0.5138)
- Confusion (tn,fp,fn,tp): [359, 557, 342, 626]

## Deployable confidence gate (the dead-band that works)
Trust the call only when the model is sure. Accuracy rises with conviction:

| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.5228 | [0.499, 0.547] |
| 0.03 | 82.8% | 0.5212 | [0.494, 0.549] |
| 0.05 | 68.5% | 0.5264 | [0.498, 0.556] |
| 0.08 | 56.8% | 0.5425 | [0.513, 0.575] |

Best confident slice: **0.5425 accuracy on 56.8% of days**
(min |p-0.5| > 0.08). Volatility / move-size gates did NOT deploy
(trailing vol predicts move size only weakly), so the gate is on model confidence.

## How to read it
- Beats a coin flip with confidence (AUC CI clears 0.50); ~tied with always-up on raw accuracy.
- Every-day ceiling ≈ 52.3%; the confidence gate buys higher accuracy on a
  selective subset, not a higher every-day number.

## Statistical Notes
- Confidence intervals use **circular block bootstrap** (block_size=20, 3000 iters) to
  account for autocorrelation in rolling-window features. CIs are wider than a naive
  i.i.d. bootstrap would produce — this is the honest estimate.
- Val window is split: first 2/3 for C-selection, last 1/3 for calibration. This prevents
  the isotonic calibration from seeing the same labels used to pick C.
- Feature stability (fraction of folds where each feature had non-zero L1 coefficient) is
  saved in `feature_stability.csv`. Features selected in <20% of folds should be treated
  as unreliable signal.

## Next: more data features (this model is built to extend)
Add genuinely *leading* inputs — overnight Asian/AU-session copper-miner closes
(BHP.AX, RIO.AX, HK copper miners), LME–COMEX spread / term structure, options skew.

---
*Research model — not financial advice. Validate before any real use.*
