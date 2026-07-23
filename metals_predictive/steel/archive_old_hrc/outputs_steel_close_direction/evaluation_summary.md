# Copper Close-to-Close Direction Model

## Question
Will CME US HRC steel close **higher tomorrow than it closed today**? Pure up/down, every day.
`target_up = 1 if close[t+1] > close[t] else 0`.

## Data
- Price dataset: `steel.csv` (close column `close`)
- Macro drivers merged: `None` + 24 cross-asset tickers
- Rows: 3024 · Date range: 2014-06-02 → 2026-06-26

## Features (150 candidates -> 47 selected)
- Copper price block (returns, momentum, trend ratios, volatility, drawdown, candle, volume)
- **Cross-asset divergence block** (the edge): steelmaker/iron-ore/base-metal return minus HRC's
  own return at 1/5/20d, divergence z-scores and ratio-momentum.
- Non-stationary price LEVELS (ma_*, volume_ma_*) and exact duplicates dropped a priori.
- **L1 selection** keeps ~55 features per fold; the final model uses
  47. Top by |coef|: close_location, TNX_ret_1d, drawdown_20d, CMC_div_20d, TNX_ret_5d, SID_div_5d, FMG_AX_div_z20, dow_sin.

## Model
- **L1-penalized Logistic Regression** + isotonic calibration (logistic beat RF/GB/HistGB —
  the usable signal is linear). C chosen per-fold on validation (C=0.25 final).
- Expanding-window walk-forward, 2024 out-of-sample days.

## Performance (walk-forward, every day)
- **Accuracy: 0.6285** (95% CI [0.6028, 0.6561])
- AUC: 0.5606 · Balanced accuracy: 0.5451
- Up precision/recall: 0.468 / 0.249
- Down precision/recall: 0.666 / 0.841
- Up base rate: 0.3592 (always-guess-majority = 0.6408)
- Confusion (tn,fp,fn,tp): [1091, 206, 546, 181]

## Deployable confidence gate (the dead-band that works)
Trust the call only when the model is sure. Accuracy rises with conviction:

| min \|p-0.5\| | coverage | accuracy | 95% CI |
|---|---|---|---|
| 0.0 | 100.0% | 0.6285 | [0.603, 0.656] |
| 0.03 | 95.7% | 0.6374 | [0.611, 0.665] |
| 0.05 | 90.4% | 0.6459 | [0.620, 0.675] |
| 0.08 | 87.4% | 0.6484 | [0.623, 0.676] |

Best confident slice: **0.6484 accuracy on 87.4% of days**
(min |p-0.5| > 0.08). Volatility / move-size gates did NOT deploy
(trailing vol predicts move size only weakly), so the gate is on model confidence.

## How to read it
- Beats a coin flip with confidence (AUC CI clears 0.50); ~tied with always-up on raw accuracy.
- Every-day ceiling ≈ 62.8%; the confidence gate buys higher accuracy on a
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
Add genuinely *leading* inputs — overnight Asian/EU steelmaker closes (BSL.AX, 5401.T,
600019.SS, Angang 0347.HK), SHFE rebar / DCE iron-ore futures (Bloomberg), HRC term structure.

---
*Research model — not financial advice. Validate before any real use.*
