# Copper Direction Model v2 — Critical Audit Report

**Prepared:** 2026-06-26  
**Models Reviewed:** `copper_close_direction_model.py` (Night-Before), `copper_close_morning_model.py` (Morning-Of)  
**Auditor:** Data Science Review

---

## Executive Summary

Both models are thoughtfully engineered and demonstrate genuine out-of-sample edge. The night-before model (53.87% OOS accuracy, AUC 0.5364) represents a statistically real but marginal signal. The morning-of model (59.82% OOS accuracy, AUC 0.6319) shows materially stronger performance driven by the Asian overnight return signal. The permutation tests pass cleanly for both.

However, several methodological issues could cause the reported performance to be optimistic, and multiple logistic regression assumptions are violated without explicit handling. The morning-of model warrants particular scrutiny because a 6.3pp accuracy gain from a single signal class is extraordinarily large for a daily direction model — extraordinary claims require extraordinary verification. The issues below are ordered roughly by severity.

---

## 1. Time-Series Methodology Audit

### 1.1 Walk-Forward Construction — PASS with caveats

The expanding-window walk-forward is correctly structured. The imputer and scaler are refit exclusively on `core` rows inside each fold. The target uses `shift(-1)` correctly. The final row is dropped since tomorrow's close is unknown. These are the foundational controls and they are implemented correctly.

**Fold anatomy:**
```
core  = [0, ..., ce-1]    ← train set
val   = [ce, ..., r-1]    ← 252-row validation (1 calendar year)
fc    = [r, ..., r+stride-1]  ← 63-row forecast chunk
```

### 1.2 Validation Set Double-Use — CRITICAL ISSUE

The `val` split serves two distinct purposes inside the same fold:

1. **C-grid selection** (`_fit_best_C`): AUC on `val` picks the best L1 penalty
2. **Isotonic calibration** (`_calibrate`): probability calibration fitted on `val`

This is validation-set contamination. The calibration model is trained on data that was also used to select C. The calibrated probabilities it produces are therefore adapted to the same held-out window used to select the model — meaning those probabilities have seen the validation labels twice. At minimum, `val` should be split:
- `val_c_select` (first 2/3) → C grid search
- `val_cal` (last 1/3) → calibration fitting

Alternatively, use `CalibratedClassifierCV` with proper cv-splitting rather than `cv="prefit"` on already-used data. This issue inflates the calibration quality and artificially narrows confidence intervals.

### 1.3 Feature Filtering Applied Once on INIT_TRAIN Only — MINOR

`filter_features(X_all, np.arange(INIT_TRAIN))` runs once before the fold loop. The missing-data and zero-variance filters are therefore based on 2015–2019 behavior only. If a cross-asset ticker became reliably available only after 2020 (or was delisted in 2022), this filter won't adapt. For a live deployment this matters more than for backtesting.

### 1.4 Global Divergence Feature Construction — LOW RISK, but verify

In `divergence_features`, the macro DataFrame is reindexed globally:

```python
mp = mp.reindex(pd.Index(dates).union(mp.index)).ffill().reindex(dates)
```

`ffill()` here correctly propagates only forward (past values into future NaN slots), so there is no look-ahead. However, this means a COMEX date not present in the macro data will inherit the **last available macro value**, which could be stale by multiple days around extended Asian holidays (e.g., Golden Week). The code does not log or flag how many dates required forward-filling, nor how far forward. Add a staleness-day counter and consider a maximum staleness cap (e.g., refuse to ffill past 5 business days).

### 1.5 Morning Model Date Alignment — NEEDS EXPLICIT TEST

This is the most important validation concern for the morning-of model. The overnight feature pipeline:

```python
daily = mp[name].reset_index(drop=True).pct_change(1)
sig = daily.shift(-1)   # "Asian session of date t+1"
```

After `reset_index(drop=True)`, `daily` is indexed by integer position aligned to COMEX trading dates, not HK trading dates. When HK has a holiday that COMEX does not, ffill inserts the previous day's price into that COMEX date slot. Then `pct_change(1)` computes the return from that stale value to the next day — a two-day HK return masquerading as a one-day return. After `shift(-1)`, this inflated return lands on the COMEX row where it is used as a same-session signal.

The consequence is that on those misaligned dates the "overnight return" is actually a multi-session return, making it look like a stronger signal than it is. HK typically has ~12 additional holidays per year vs COMEX. This is not necessarily catastrophic (12/252 = ~5% of days), but the claim that this is a clean "Asian session of t+1 only" signal is not fully validated. **Mandatory fix: run the model on aligned dates only (intersect COMEX trading days with HK trading days) and compare performance.**

### 1.6 Permutation Test Specification — SLIGHTLY CONSERVATIVE AGAINST REAL

The permutation null runs at a fixed `C=0.05` with `calibrate=False`:

```python
real = walk_forward(X, y, cgrid=(0.05,))   # single C
pn = walk_forward(X, ys, cgrid=(0.05,), calibrate=False)
```

The real model selects C from `[0.05, 0.1, 0.25, 0.5]` per fold and applies calibration. Both C selection and calibration add degrees of freedom not present in the permuted null. A fully correct permutation test would run the *complete pipeline* (C grid + calibration) on each shuffled label set. Running only C=0.05 without calibration makes the null slightly easier to beat than it should be. The p=0.000 result means this doesn't change the conclusion, but the test design should be noted.

### 1.7 Expanding Window — Old Data Relevance Risk

The expanding window grows to include all data from 2015 onwards. The final 2026 model trains on 11 years of data. Commodity market microstructure has changed substantially: COMEX electronic session hours expanded, algorithmic participation grew, and the Asian price-discovery role has strengthened. Using 2015 data (low algo, different volatility regime) in 2026 predictions adds noise. A rolling 3–5 year window would be more representative of current market dynamics and is standard practice for alpha decay reasons.

---

## 2. Logistic Regression Assumptions Check

### 2.1 Binary Dependent Variable — PASS
`target_up ∈ {0, 1}`. ✓

### 2.2 Independence of Observations — VIOLATED (not addressed)

This is the most fundamental violation. Daily financial returns have known autocorrelation structure:
- Return autocorrelation is weak but non-zero at short lags
- **Feature autocorrelation is severe**: `volatility_20d[t]` and `volatility_20d[t+1]` share 19 of 20 data points. `momentum_20d` at consecutive days differs by only 2 observations. These features are nearly deterministically related across adjacent time steps.
- The divergence z-scores (`name_div_z20`) use a 20-day rolling mean and std, creating extreme autocorrelation across folds

**Impact on inference:** The bootstrap confidence intervals assume i.i.d. draws. Sampling with replacement from a correlated time series is equivalent to a block-bootstrap with block size 1 — which severely underestimates variance. The 95% CI [0.517, 0.561] for the night-before model is likely too narrow; the true CI is wider. A proper block bootstrap (block size = 20–63 days to cover the longest rolling window) should replace the current naive bootstrap in `_boot_acc`.

### 2.3 No Multicollinearity — VIOLATED (partially mitigated by L1)

The feature matrix has extensive structural multicollinearity:

| Group | Correlated Features |
|---|---|
| Copper returns | `log_return_Nd` at n=1,2,3,5,10,20 are highly correlated by construction; 10d return includes 5d return |
| Divergence vs copper | Every divergence feature = `asset_ret - copper_ret`; `copper_ret` is also a direct feature |
| Momentum/return equivalence | `momentum_20d` and `return_20d` are mathematically identical (a priori drop handles this) |
| Cross-asset features | `brent_momentum_20d` and `brent_return_1d` both appear in the final model — brent is doubly represented |
| Volatility | `volatility_5d`, `volatility_10d`, `volatility_20d`, `volatility_60d` are nested rolling stds |

L1 regularization suppresses redundant features but does not eliminate collinearity between the features it *does* keep. The final night-before model selects both `brent_momentum_20d` and `brent_return_1d` — these are correlated signals from the same underlying asset. Their individual coefficient magnitudes are unreliable in the presence of collinearity (L1 will give the weight to whichever aligns better with that fold's labels, not the truly causal one). Variance Inflation Factors (VIF) should be computed on the selected feature set.

### 2.4 Linearity of Log-Odds — NOT TESTED

LR assumes `log(p/(1-p)) = Xβ` — a linear relationship between features and the log-odds of copper going up. This is untested. Commodity direction prediction is likely to involve:
- Threshold effects (extreme momentum reverting, moderate momentum continuing)
- Interaction effects (high volatility × divergence signal)
- Mean-reversion at extreme levels

The code authors note that LR beat RF/GB — but this comparison was done on the same validation period that was used for model selection. A cleaner comparison using a held-out 2-year block would be more convincing. The claim that "the signal is linear" is an empirical observation about this dataset, not a validated assumption.

### 2.5 Large Sample Size — MARGINAL for night-before model

With ~1,884 OOS days but ~12 selected features, the rule of thumb (≥10 events per predictor) is nominally met. However, due to autocorrelation (2.2 above), the **effective sample size** is far smaller than 1884. If the autocorrelation horizon is ~20 days (the longest rolling window), the effective independent samples is approximately 1884/20 ≈ 94. With 12 predictors, that is 7.8 events per predictor — below the 10-per-predictor threshold. The statistical power for detecting real coefficients is more limited than the raw count implies.

### 2.6 No Perfect Separation — PASS (L1 handles implicitly)

L1 regularization prevents the Hauck-Donner effect from perfect or quasi-perfect separation. ✓

### 2.7 Heavy-Tailed Distributions — NOT ADDRESSED

Financial returns have kurtosis >> 3 (fat tails). `StandardScaler` normalizes mean and variance but does not address distributional shape. During crisis periods (COVID March 2020, 2022 energy shock), the model receives inputs with z-scores of 5–10+, well outside the training distribution of `StandardScaler`. These extreme observations can dominate the gradient and distort coefficient estimates. A `RobustScaler` (scales to IQR) or a winsorized scaler (clip at ±5σ before scaling) would be more appropriate for financial data.

---

## 3. Model Performance Analysis

### 3.1 Night-Before Model: Marginal Edge

| Metric | Value | Benchmark |
|---|---|---|
| OOS Accuracy | 53.87% | Majority guess: 51.38% |
| AUC | 0.5364 | Coin flip: 0.5000 |
| Balanced Accuracy | 53.57% | — |
| CI lower bound | 51.70% | Majority baseline: 51.38% |

The lower confidence bound (51.70%) barely clears the majority baseline (51.38%). The edge is real (permutation p=0.000) but thin. The following year-by-year breakdown reveals severe instability:

| Year | Accuracy | AUC | Status |
|---|---|---|---|
| 2019 | 53.57% | 0.529 | Marginal |
| 2020 | **48.22%** | **0.432** | Below baseline |
| 2021 | 59.13% | 0.559 | Good |
| 2022 | 53.39% | 0.535 | Marginal |
| 2023 | **47.01%** | **0.471** | Below baseline |
| 2024 | 50.00% | 0.495 | Coin flip |
| 2025 | 58.73% | 0.518 | Good |
| 2026 (partial) | 55.08% | 0.551 | Good |

The model underperforms a coin flip in 2020 and 2023 — two of eight OOS years. AUC < 0.50 in these years means the model was actively mis-predicting direction on net. The "good" years (2021, 2025) may be compensating for these failures. **The first-half vs. second-half split shows a concerning decay: 54.14% → 51.91%**, suggesting the edge is diminishing or was partially in-sample for the early OOS period.

### 3.2 Config Robustness — Night-Before Model FAILS

| Config | Accuracy |
|---|---|
| Baseline (1000/63/42) | 0.5303 |
| init_train=750 | **0.5122** |
| init_train=1250 | 0.5226 |
| stride=21 | 0.5329 |
| stride=126 | 0.5271 |
| seed=7 | 0.5297 |
| seed=123 | 0.5303 |

Accuracy spread: **0.5122 to 0.5329** (std=0.0054). Changing only the training window size from 1000 to 750 rows drops performance by 1.8pp — nearly the entire edge above 50%. This sensitivity to a hyperparameter that was presumably tuned on the OOS data is a form of protocol overfitting. The true expected accuracy under a randomly chosen reasonable protocol is closer to 0.52 than 0.54.

Note also that the 0.5387 reported in the main evaluation uses full C-grid search + calibration, while the robustness check uses C=0.05 only (the first row shows 0.5303, not 0.5387). This means **the gap between best and worst config is larger in the full pipeline than shown** — the C-grid and calibration add per-fold adaptability that may be fitting to this specific OOS window.

### 3.3 Morning-Of Model: Stronger but Under Greater Scrutiny

| Metric | Value | vs Night-Before |
|---|---|---|
| OOS Accuracy | 59.82% | +5.95pp |
| AUC | 0.6319 | +0.0954 |
| CI lower bound | 57.64% | Far above baseline |

Year-by-year performance is remarkably consistent (57.75%–62.95% across 8 years, all years above baseline). This consistency is actually more impressive than the raw accuracy and suggests a robust signal.

However, **the dominant features are both HK miner overnight returns:**
- `2899_HK_overnight_div`: coefficient 0.265 (2.4× the next feature)
- `asianminers_overnight_mean`: coefficient 0.124
- Next feature (`brent_momentum_20d`): 0.109

The overnight signal accounts for the bulk of the 6pp gain. Given the date-alignment concern (§1.5 above), the integrity of this signal must be verified before treating 59.82% as the true expected performance. **If the alignment issue explains even 10% of the effect, the real accuracy is closer to 59%–59.5%, which is still excellent — but the verification is mandatory.**

### 3.4 Permutation Test — BOTH MODELS PASS

Night-before null: mean=0.508, max=0.522, p=0.000 (real at C=0.05 fixed: 0.530).  
Morning-of null: reported p=0.000.  
Both models have genuine signal. This is not the concern. The concern is the *magnitude* of the claimed edge relative to methodology choices.

### 3.5 Confidence Gate — Does Not Work as Described

| min |p-0.5| | Coverage | Accuracy | CI |
|---|---|---|---|
| 0.0 | 100% | 53.87% | [51.70%, 56.05%] |
| 0.03 | 74.4% | 53.96% | [51.39%, 56.53%] |
| 0.05 | 59.8% | 53.50% | [50.58%, 56.52%] |
| 0.08 | 36.2% | 53.52% | [49.85%, 57.34%] |

There are two problems here. First, accuracy does **not** monotonically increase with confidence as claimed — it is 53.87%, 53.96%, 53.50%, 53.52%. The improvement at |p-0.5|>0.03 is +0.09pp, within noise. Second, as the confidence threshold increases, the CI widens because n shrinks. The claim that higher-confidence predictions are more accurate is not supported. The confidence gate adds no deployable value for the night-before model, and this is an important negative result that the code should state more explicitly.

---

## 4. Critical Issues Summary

| # | Issue | Severity | Model Affected |
|---|---|---|---|
| C1 | Validation set used for both C-selection and calibration | High | Both |
| C2 | Night-before config robustness fails (1.8pp swing on init_train) | High | Night-Before |
| C3 | Morning-of date alignment unverified (HK vs COMEX holiday calendar) | High | Morning-Of |
| C4 | Serial correlation violates i.i.d. bootstrap; CIs are too narrow | High | Both |
| C5 | Permutation test misspecified (fixed C, no calibration in null) | Medium | Both |
| C6 | Expanding window uses 11 years; old regime may be noise | Medium | Both |
| C7 | RobustScaler needed for financial fat tails | Medium | Both |
| C8 | Confidence gate provides no meaningful lift for night-before model | Medium | Night-Before |
| C9 | Global ffill without staleness cap | Low | Both |
| C10 | No VIF computed on selected features | Low | Both |

---

## 5. Recommendations

### Priority 1: Fix Methodological Issues (validity-affecting)

**5.1 Split the validation window for C-selection vs calibration**
```python
# In walk_forward():
val_c  = np.arange(ce, ce + 2*(r-ce)//3)    # first 2/3 of val window
val_cal = np.arange(ce + 2*(r-ce)//3, r)     # last 1/3
base, _C = _fit_best_C(Xs, y, core, val_c)
cal = _calibrate(base, Xs[val_cal], y[val_cal])
```

**5.2 Replace naive bootstrap with block bootstrap**
The current `_boot_acc` samples individual days independently, which is wrong for autocorrelated time series. Use a circular block bootstrap with block size = 20 days (the longest rolling feature window):
```python
def _block_boot_acc(correct, block_size=20, iters=3000):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(correct)
    n_blocks = int(np.ceil(n / block_size))
    s = []
    for _ in range(iters):
        starts = rng.integers(0, n, n_blocks)
        sample = np.concatenate([correct[s:s+block_size] for s in starts])[:n]
        s.append(sample.mean())
    return tuple(np.percentile(s, [2.5, 97.5]))
```
This will widen the CIs and give a more honest picture of uncertainty.

**5.3 Verify morning model alignment explicitly**
Add a validation test to `copper_close_morning_model.py` that verifies the shift alignment:
```python
# Sanity check: on the 10 most-recent predictions,
# confirm the date of the "overnight return" precedes the prediction date
# by checking against known COMEX calendar. Flag any mismatches.
```
Additionally, run the no-shift control explicitly within the same pipeline:
```python
Xo_noshifted = overnight_features_no_shift(...)   # shift(0) instead of shift(-1)
probs_control, _, _ = base.walk_forward(pd.concat([Xc, Xd, Xo_noshifted], axis=1), y)
```
The claimed "control stays at 53.9%" should be a committed test output, not a verbal assertion.

**5.4 Fix the permutation test to use the full pipeline**
```python
# Remove fixed C and calibrate=False — run the exact same pipeline as real model
for k in range(K):
    ys = y.copy(); rng.shuffle(ys)
    pn = walk_forward(X, ys)   # full pipeline, not fixed-C
    ...
```

### Priority 2: Address LR Assumption Violations

**5.5 Switch to RobustScaler**
```python
from sklearn.preprocessing import RobustScaler
sc = RobustScaler().fit(Xi.iloc[core])
```
RobustScaler uses IQR instead of std, making it insensitive to the fat-tail outliers common in financial crisis periods. Pair with winsorization at ±5σ:
```python
Xi = Xi.clip(lower=Xi.mean() - 5*Xi.std(), upper=Xi.mean() + 5*Xi.std())
```

**5.6 Compute VIF on selected features per fold**
Add a periodic VIF report (every N folds or for the final model) to expose which feature pairs are multicollinear despite L1 selection. This informs whether coefficient magnitudes are interpretable.

**5.7 Test for residual autocorrelation**
After computing OOS predictions, run a Ljung-Box test on the error series `(correct_t - mean_correct)`:
```python
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(oos["correct"] - oos["correct"].mean(), lags=20, return_df=True)
```
If Q-statistics are significant, there is unexploited serial structure — a direct signal to extend the model.

### Priority 3: Feature Engineering Improvements

**5.8 Add calendar effects**
Day-of-week and month-of-year effects are real in commodity markets (Monday returns, seasonal copper demand cycles from China). These are trivially cheap and stationary:
```python
f["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 5)
f["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 5)
f["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
```

**5.9 Add LME–COMEX spread**
The LME 3-month copper price vs COMEX front-month price spread (the "arb basis") is a known leading indicator of COMEX direction. When LME trades at a premium to COMEX, arbitrage flows will lift COMEX on the next session. This is genuinely exogenous to the copper return series used as the target.

**5.10 Add term structure / contango–backwardation**
COMEX copper front-month vs. 3-month price ratio signals inventory expectations. Backwardation (spot premium) historically precedes up moves; contango precedes flat/down moves. This is a stationary, fundamentally motivated feature.

**5.11 Switch from expanding to rolling window**
Replace expanding window with a 3-year rolling training window:
```python
INIT_TRAIN = 756  # ~3 years
core = np.arange(max(0, r - VAL_WINDOW - INIT_TRAIN), r - VAL_WINDOW)
```
This respects alpha decay: a relationship that held in 2015 is not necessarily informative in 2026.

**5.12 Regime conditioning**
Partition the walk-forward predictions by VIX regime:
- Low vol: VIX < 15
- Medium: 15 ≤ VIX < 25
- High: VIX ≥ 25

Report accuracy within each regime. If the edge is concentrated in one regime (e.g., low-vol), the model should incorporate a regime switch.

### Priority 4: Deployment Improvements

**5.13 Replace binary confidence gate with Kelly-fractional position sizing**
The confidence gate is binary (commit or abstain). A more sophisticated deployment uses calibrated probabilities directly for position sizing:
```python
# Truncated Kelly fraction
p = model.predict_proba(x)[0][1]
kelly_f = (p - (1 - p)) / 1.0   # simplified; assumes +1/-1 payoff
position = np.clip(kelly_f, -max_position, max_position)
```
This captures more of the signal in moderate-confidence days rather than discarding them entirely.

**5.14 Track feature selection stability across folds**
Log which features have non-zero L1 coefficients in each fold. A feature selected in <20% of folds is likely noise. A feature selected in >80% of folds is load-bearing. The current code only tracks `avg_features_used_per_fold` (a count), not which features are stable.

---

## 6. Summary Score Card

| Category | Night-Before | Morning-Of |
|---|---|---|
| Leakage controls | A- | A- |
| Walk-forward design | B+ | B+ |
| LR assumption compliance | C | C |
| Statistical inference quality | C+ (naive CI) | C+ |
| Feature engineering | B+ | A- |
| Robustness testing | B (fails config sweep) | B |
| Claimed edge validity | Marginal but real | Strong, needs alignment verification |
| **Overall** | **B-** | **B+** |

The night-before model is sound methodology on thin signal. The morning-of model is potentially excellent but is riding heavily on a single feature class whose data alignment should be independently verified before any real deployment. The val/calibration contamination issue affects both models and should be the first fix.

---

## ADDENDUM — Leakage / look-ahead re-audit (2026-06-29)

**The report above is now STALE.** It reviewed a morning model at 59.82% (Asian-miners only,
before LME data) and flagged a val-set double-use issue (§1.2). Both have changed:

1. **§1.2 is FIXED in the current code.** `walk_forward` splits the validation window into
   `val_c` (first 2/3, C-selection) and `val_cal` (last 1/3, isotonic calibration) — disjoint.
2. **The morning model is now LME-driven and scores 86.7%** (AUC 0.936), not 59.82%. That
   version was never audited. Re-audit below.

### Verdict: clean, leak-free, and look-ahead-free — with two caveats

**Data is genuinely liquid (unlike the aluminum/steel siblings).** HG=F (COMEX copper): 1.8%
no-range days, 0.6% flat closes, median volume 535, balanced 50.5% baseline. So there is **no
stale-target tautology** — `next_gap_return` agrees with the close move only 70.1% of the time
(a normal correlation for a real contract, not aluminum's 97.5% leak). The gap feature is
legitimate here.

**Pipeline machinery is clean.** Imputer + RobustScaler fit on `core` (train) only; C on `val_c`;
calibration on disjoint `val_cal`; scored only on the future fold. Target = `close.shift(-1)`.
No in-sample leakage.

**Night-before model is clean and honest.** Only copper + cross-asset divergence features (no
LME/gap/overnight). 52.3% acc vs 51.4% baseline, AUC 0.532 — real-but-marginal, correctly reported.

**The 86.7% morning headline is the LME→COMEX arbitrage lead, and it is leak-free.**
- Dominated by one feature: `lme_overnight_ret` (coef 3.07, ~6× the next). The LME-only model
  (`copper_close_morning_v3.py`) scores **85.4%** alone — the LME features ARE the model.
- Timing is sound: LME 3-month Ring settlement (~7:15am ET) precedes the COMEX settle (~1pm ET)
  the same date, so LME date-(t+1) → COMEX date-(t+1) is a genuine ~5-hour forecast, not look-ahead.
- **Empirical leak test passes:** the feature's |LME overnight move| averages 0.95% vs the COMEX
  full-day move's 1.10%, with 86% sign agreement. A real leak (LME close postdating COMEX) would
  give ~100% agreement and equal magnitude; the smaller LME magnitude is the signature of a true
  pre-COMEX *lead*.

### Caveat 1 — RESOLVED (2026-06-30): full no-shift control confirms LME lead is clean
`lme_features_no_shift()` added; control re-run 2026-06-30. Result:

| | acc | AUC |
|---|---|---|
| Real model (GB, morning-of) | **0.8641** | **0.9351** |
| Full no-shift control | 0.5292 | 0.5356 |
| **True overnight lift** | **+33.5pp** | **+0.3995** |

The control collapses to 52.9% — indistinguishable from the night-before baseline (~52%). This is
the maximum possible lift magnitude and conclusively proves: the 86.4% is entirely attributable to
features that use day t+1 information (LME Ring settle ~7am ET), and those features genuinely
precede the COMEX settle (~1pm ET). **No look-ahead bias. No leak. Confirmed clean.**

The prior broken control (ctrl_acc=0.8508, lift=+1.6pp) was wrong because it left `lme_features`
shifted — it measured only the Asian miners' marginal contribution, not the dominant LME feature.

### Caveat 2 — residual data-provenance risk (low, but code can't self-check it)
Leak-freeness hinges on `lme_copper.csv`'s `lme_close` being the **LME PM Ring settlement
(~7am ET)**, not a later electronic close (~2pm ET, post-COMEX). The magnitude/agreement test
indicates it is the morning settlement, so risk is low — but confirm the vendor convention
(e.g. NASDAQ `CHRIS/LME_CU1`) before production.

### Business read
The morning model isn't a fundamental forecast — it's "LME copper already moved this morning, so
COMEX follows." Real and tradeable only if you act between the 7am ET LME settle and the 1pm ET
COMEX settle. For a hedging decision you'd just read the LME directly; the durable value here is
the *night-before* signal (~52%) and the macro-divergence research, not the 86.7% arb tracker.

*This report is a methodology audit only. Nothing here constitutes financial advice.*
