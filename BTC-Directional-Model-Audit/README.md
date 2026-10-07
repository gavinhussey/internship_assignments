# BTC Directional Model — Audit and Search for an Edge

## Executive summary

A previous intern built a supervised-learning model that reportedly predicts
next-day Bitcoin direction at **55.94% accuracy** (vs. a 50.31% majority-class baseline). I
audited that result, found it was inflated by feature-selection leakage, rebuilt the model
leakage-free, and then tried substantially harder to find a real edge: a much larger
candidate feature pool, proper walk-forward validation with significance testing, longer
(7-day and 30-day) horizons, and a completely different modeling paradigm (regime-switching
/ trend-following instead of classification).

**None of it produced a statistically defensible edge.** Every model in this folder that
beats "always predict up" does so by a point estimate of 1-4 percentage points, and none of
those point estimates survive McNemar's test + a fold-clustered bootstrap confidence
interval (i.e. none can be distinguished from noise). At 7-day and 30-day horizons, most
models are numerically *worse* than always predicting up. The conclusion is not "the model
needs more tuning" — it's that BTC price, on-chain, and macro data at daily granularity does
not contain an easily-discoverable directional edge, and the honest next step is new data
(funding rates, futures basis, order flow) or a different question entirely (trading-strategy
evaluation with real costs, confidence-thresholded predictions), not another variation on
the same inputs.

This folder is self-contained: every notebook reads from and writes to `data/` inside this
folder only. No files outside this folder are required to run it.

## How to read this folder

| # | Notebook | Question it answers |
|---|----------|----------------------|
| 1 | `notebooks/01_original_model_replication.ipynb` | Does the original 14-feature model hold up under proper validation (walk-forward + significance testing), using its own original data? |
| 2 | `notebooks/02_original_model_replication_multiday.ipynb` | Same model/data/methodology, but predicting 7-day and 30-day direction instead of next-day. |
| 3 | `notebooks/03_broader_feature_search_and_walkforward.ipynb` | What if we pull fresh live data and search a much larger (100+) candidate feature pool instead of the original's 14 hand-picked ones? |
| 4 | `notebooks/04_extended_feature_selection.ipynb` | Same broad-search idea, but over a longer 2014-2026 history with feature selection done fresh inside every walk-forward fold (no single "pick once" selection step). |
| 5 | `notebooks/05_regime_switching_and_alternatives.ipynb` | After five classifiers all failed, try a genuinely different paradigm: hidden-Markov regime-switching and fixed moving-average trend rules. |

`docs/development-log.md` is the running log I kept during the earlier stage of this work
(the audit of the leakage problem itself, and the first leakage-safe rebuild, "Mk2"). It's
included for context on how the leakage was found and reasoned about; the notebooks above
are the more rigorous, final versions of that work.

## Step 1 — The original claim and why it doesn't hold up

The original notebook (`Supervised-Learning-BTC-Price-Direction/Pricing Models/ML-MODELS.ipynb`,
not included in this folder — see `docs/development-log.md` for the full audit) built its
full transformed dataset — including the final holdout period — before running correlation
analysis and Boruta-style feature selection, and only split into train/test *after* that
selection was final. That means the test period influenced which features were chosen. It
also evaluated on a single 80/20 split with no significance testing at all, and its 14
features were hand-picked by watching performance on the only historical window available —
a second, subtler form of the same problem that walk-forward validation alone can't undo.

`notebooks/01_original_model_replication.ipynb` rebuilds the original's exact 14
hand-picked features from its own source data, but replaces the evaluation with:
- an 8-fold expanding-window walk-forward split with a purge step (drop any training row
  whose label reaches into the test fold),
- McNemar's test and a fold-clustered bootstrap confidence interval against an "always
  predict up" baseline,
- all four of the original's model families (Logistic Regression, Random Forest, LSTM,
  CNN-LSTM).

**Results:**

| Model | Accuracy | Baseline (always-up) | Edge | McNemar p | 95% CI excludes 0? |
|---|---|---|---|---|---|
| Logistic Regression | 52.32% | 51.54% | +0.77pp | 0.75 | No |
| Random Forest | 53.16% | 51.54% | +1.62pp | 0.44 | No |
| LSTM | 50.00% | 51.54% | -1.54pp | 0.47 | No |
| CNN-LSTM | 50.84% | 51.54% | -0.70pp | 0.74 | No |

Nothing clears the bar. The best point estimate (Random Forest, +1.62pp) has a p-value of
0.44 — nowhere near significant — and its bootstrap CI comfortably includes zero.

## Step 2 — Does it hold up at longer horizons?

`notebooks/02_original_model_replication_multiday.ipynb` reruns the identical setup, only
retargeting to 7-day and 30-day direction (purge window scaled to the horizon). Longer
horizons raise the bar, because "always predict up" itself becomes a much stronger baseline
once you're averaging over more of BTC's overall uptrend during this period (54.9% at 7
days, 57.3% at 30 days).

**7-day horizon:**

| Model | Accuracy | Baseline | Edge | McNemar p |
|---|---|---|---|---|
| Logistic Regression | 50.00% | 54.86% | -4.86pp | 0.022 (worse, significant) |
| Random Forest | 51.36% | 54.86% | -3.50pp | 0.056 |
| LSTM | 48.14% | 54.86% | -6.71pp | <0.001 (worse, significant) |
| CNN-LSTM | 48.50% | 54.86% | -6.36pp | <0.001 (worse, significant) |

**30-day horizon:**

| Model | Accuracy | Baseline | Edge | McNemar p |
|---|---|---|---|---|
| Logistic Regression | 45.21% | 57.29% | -12.07pp | <0.001 (worse, significant) |
| Random Forest | 56.57% | 57.29% | -0.71pp | 0.68 |
| LSTM | 49.79% | 57.29% | -7.50pp | <0.001 (worse, significant) |
| CNN-LSTM | 46.79% | 57.29% | -10.50pp | <0.001 (worse, significant) |

At both horizons, every model is at or below baseline, and several are *significantly worse*
than just staying long. Stretching the prediction horizon does not help; it hurts.

## Step 3 — What if the feature set was just too small?

`notebooks/03_broader_feature_search_and_walkforward.ipynb` pulls fresh live data (Coinbase
OHLCV, a broad CryptoQuant on-chain catalog, yfinance cross-asset market data, FRED
macro series) and builds 111 candidate features instead of the original's 14 — BTC-native
technicals (returns, momentum, realized volatility, moving-average gaps) plus transforms of
every pulled on-chain/market/macro series. Feature selection is done on the training window
only, with a sign-stability filter across the training period.

**Results (leakage-safe, single 80/20 split, 9 features selected):** 51.98% accuracy vs.
50.07% baseline — a +1.92pp edge.

**Full expanding-window walk-forward (2018-2026, 9 folds, 2,922 pooled predictions):**
52.94% pooled accuracy vs. 50.86% pooled baseline — a +2.09pp edge, but the sign flips
negative in 3 of the 9 individual folds. This notebook also replicates the original 14-feature
spec on a like-for-like walk-forward:

| Feature set | Pooled accuracy | Pooled baseline | Edge |
|---|---|---|---|
| Original 14 features (with on-chain) | 53.94% | 51.24% | +2.70pp |
| Same, on-chain features dropped (market + macro only) | 51.17% | 50.86% | +0.31pp |

Dropping on-chain features collapses almost all of the edge — on-chain data (miner
behavior, coin age, hashrate) is carrying most of whatever signal exists, and even with it
included, the edge is a few points, not the ~5.6pp originally reported, and (per notebook 01)
not statistically distinguishable from zero once tested properly.

## Step 4 — What if selection itself was the problem?

`notebooks/04_extended_feature_selection.ipynb` goes further: a 112-candidate pool over a
longer 2014-2026 history, with feature selection re-run fresh *inside every walk-forward
fold* rather than picked once and reused (the mk3/mk4 approach still selected once on a
training window and kept those features fixed across folds). A small stable core of ~6
features does recur across most folds (BTC 2-day return, 30-day return percentile, 14-day
close-location, Dow/Gold z-scores, 7-day momentum acceleration) — so the selection isn't
just picking noise.

**Results:** every model sits numerically *below* the always-up baseline:

| Model | Accuracy | Baseline | Edge | McNemar p |
|---|---|---|---|---|
| Logistic Regression | 51.03% | 52.35% | -1.32pp | 0.12 |
| Random Forest | 51.38% | 52.35% | -0.97pp | 0.23 |
| LSTM | 50.38% | 52.35% | -1.97pp | 0.066 |
| CNN-LSTM | 51.38% | 52.35% | -0.97pp | 0.35 |

More history, a larger candidate pool, and leakage-safe per-fold selection all at once
still does not beat the simplest possible baseline. This is the clearest evidence that "more
of the same kind of data" isn't the fix.

## Step 5 — A different paradigm entirely

`notebooks/05_regime_switching_and_alternatives.ipynb` stops trying a sixth classifier and
tests a structurally different idea: model BTC returns as switching between a small number
of latent (hidden) regimes (Hamilton-style Markov regime-switching, `statsmodels`), predict
tomorrow's direction from the regime's sign, and separately test a family of fixed (unfitted)
moving-average trend-following rules across 14 window lengths (5-300 days).

**Regime-switching, best variant (2-state, mean-sign rule):** 52.38% accuracy vs. 52.65%
baseline — edge -0.26pp, not significant. All other regime variants (3-state, up-frequency
rule, excess/de-meaned returns, an exogenous on-chain regressor, 30-day horizon) also failed
to beat baseline; the excess-return and exogenous-regressor variants were *significantly
worse* (bootstrap CI entirely below zero). The root cause, diagnosed directly in the
notebook: in 6 of 8 walk-forward folds, every regime the model finds has a *positive* mean
return — there is essentially no genuine bearish regime in this sample, so "predict the
regime's sign" degenerates into "predict up" almost always, which is exactly the baseline.

**Moving-average trend-following, all 14 window lengths (5-300 days):** every single window
length has a negative edge vs. baseline; the five shortest windows (5-30 days) are
significantly worse.

Ten independent modeling attempts — five classifiers across three feature sets, plus five
regime/trend formulations — all reach the same null result.

## Bottom line

1. **Don't allocate against this model or feature set as it stands.** The apparent edge in
   the original notebook was mostly a byproduct of leakage; what survives proper validation
   is a few points of point-estimate accuracy that isn't statistically distinguishable from
   the "always predict up" baseline, and that disappears or reverses at longer horizons.
2. **More of the same data is not the answer.** A 111-candidate pool, a 12-year history,
   per-fold leakage-safe selection, and a genuinely different model class (regime-switching)
   were all tried, and none of them moved the needle.
3. **What would actually be needed for a real edge:**
   - **Genuinely new data classes**, not new slices of the same price/on-chain/macro
     history: derivatives/options data (implied vol, put-call skew, options positioning),
     funding rates and futures basis, and exchange order-flow/order-book data. (BTC ETF flow
     data was pulled — `farside_btc_etf_flows.csv` — but only covers ~2.5 years since launch,
     too short a history to evaluate on its own yet.)
   - **Untested model classes**, e.g. gradient boosting (XGBoost/LightGBM), which sits
     between the plain Random Forest tried here and the LSTM/CNN-LSTM sequence models.
   - **A different framing of the problem**: confidence-thresholded predictions (only act on
     the highest-conviction days instead of forcing a call every day), and evaluating as an
     actual trading strategy (Sharpe ratio, drawdown, realistic transaction costs) rather
     than plain next-day accuracy.
   - Any future "win" should be treated as a hypothesis to confirm on genuinely fresh,
     held-out data the selection process never touched — not a result to act on from the
     table it was found in.

## Running the notebooks

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook notebooks/
```

Every notebook here defaults to the cached CSVs already committed under `data/` and runs
fully offline — no API keys or network access required. Notebook 03
(`03_broader_feature_search_and_walkforward.ipynb`) can optionally refresh that cached data
from live sources (Coinbase, CryptoQuant, yfinance, FRED); to enable that, copy
`.env.example` to `.env` in this folder's root and fill in a CryptoQuant and FRED API key.
Without a `.env` file, that notebook still runs end-to-end against its cached data.

## Folder contents

```
BTC-Directional-Model-Audit/
├── README.md                     <- this file
├── requirements.txt
├── .env.example                  <- optional API keys for notebook 03's live-refresh path
├── docs/
│   └── development-log.md        <- earlier working log: leakage audit + first leakage-safe rebuild ("Mk2")
├── notebooks/
│   ├── 01_original_model_replication.ipynb
│   ├── 02_original_model_replication_multiday.ipynb
│   ├── 03_broader_feature_search_and_walkforward.ipynb
│   ├── 04_extended_feature_selection.ipynb
│   └── 05_regime_switching_and_alternatives.ipynb
└── data/
    ├── original/                 <- the original model's own source CSVs (BTC price, market, on-chain, economic)
    └── mk4/                      <- cached pulls + every experiment's output tables, feeding notebooks 01-05
        └── extended/             <- longer-history (2014-2026) candidate feature set used by notebooks 04-05
```
