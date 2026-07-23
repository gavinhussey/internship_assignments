# Copper Close Direction Model — Documentation

**Version:** 2.0  
**Last Updated:** 2026-06-29  
**Author:** Gavin's Scrap  

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [How It Works — Plain English](#2-how-it-works--plain-english)
3. [Two Models: Night-Before vs. Morning-Of](#3-two-models-night-before-vs-morning-of)
4. [Data Sources](#4-data-sources)
5. [Features](#5-features)
6. [Model Architecture](#6-model-architecture)
7. [Results](#7-results)
8. [Honest Interpretation](#8-honest-interpretation)
9. [Known Weaknesses](#9-known-weaknesses)
10. [How to Run](#10-how-to-run)
11. [Output Files](#11-output-files)

---

## 1. Purpose

This model predicts whether COMEX copper (HG front month) will **close higher or lower** on a given trading day. It is built as a pricing proxy for a scrap copper business: knowing the likely direction of copper before the COMEX settle (~1 PM CT) helps inform buy/sell decisions and daily pricing.

**This is not a trading model.** It does not predict magnitude, only direction (UP or DOWN). It is intended as one input into a broader pricing process.

---

## 2. How It Works — Plain English

Copper trades on two major global exchanges:

- **LME (London Metal Exchange):** Ring official settlement finalized at ~1:15 PM London = **~7:15 AM CT**
- **COMEX (Chicago/New York):** Pit session settles at ~1:00 PM CT — roughly **6 hours later**

Because LME and COMEX both price the same underlying commodity, the direction LME copper settles in the morning is a strong predictor of which way COMEX will close that afternoon. When LME copper is up 2% by 7:15 AM CT, COMEX copper closes up that day about **84–91% of the time** in normal market conditions.

The model captures this relationship statistically using 12 years of historical data and confirms it holds out-of-sample across every year from 2019–2026 (with one notable exception in 2025 — see [Known Weaknesses](#9-known-weaknesses)).

**The model's edge is fundamentally this:** LME settles 6 hours before COMEX. Same commodity. Same direction almost every day.

---

## 3. Two Models: Night-Before vs. Morning-Of

| | Night-Before Model | **Morning-Of Model (this doc)** |
|---|---|---|
| **Script** | `copper_close_direction_model.py` | `copper_close_morning_model.py` |
| **When to predict** | Prior day's COMEX close (~1 PM CT, day t) | Morning of target day, by 8:30 AM CT |
| **Key data available** | Everything through prior close | + LME morning settlement, Asian miners |
| **Accuracy** | 52.3% | **84.7%** |
| **AUC** | 0.532 | **0.916** |
| **Interpretation** | Marginally better than coin flip | Strongly useful |

The night-before model has almost no edge because copper direction is largely unpredictable a day in advance. The morning-of model is dramatically better because by 8:30 AM CT you already know what LME did — which is essentially the same information as COMEX.

---

## 4. Data Sources

| Source | File | Description | Update Frequency |
|---|---|---|---|
| COMEX copper OHLCV | `data/raw/copper.csv` | Daily open/high/low/close/volume, 2014–present | Daily (manual) |
| LME copper settlement | `data/external/lme_copper.csv` | LMCADS03 Comdty — LME 3-month official settlement, USD/tonne | Daily via `bloombergCopper.py` |
| LME aluminum settlement | `data/external/lme_aluminum.csv` | LMAHDS03 Comdty — LME 3-month official settlement, USD/tonne | Daily via `bloombergCopper.py` |
| LME zinc settlement | `data/external/lme_zinc.csv` | LMZSDS03 Comdty — LME 3-month official settlement, USD/tonne | Daily via `bloombergCopper.py` |
| SHFE copper front month | `data/external/shfe_copper.csv` | CU1 Comdty — Shanghai Futures Exchange copper, CNY/tonne | Daily via `bloombergCopper.py` |
| Asian copper miners | Downloaded via yfinance | BHP.AX, RIO.AX, S32.AX, SFR.AX, 2899.HK, 0358.HK, 1208.HK, 3993.HK | Cached locally |
| Macro cross-assets | Downloaded via yfinance | ~180 macro features: USD, rates, oil, equity indices, metals ETFs | Cached locally |

**Bloomberg is required** to update the LME/SHFE data. Run `python bloombergCopper.py` with Terminal open to refresh all four datasets.

---

## 5. Features

### 5.1 Feature Groups

The model starts with **214 candidate features** across four groups. L1 regularization selects the most predictive at each walk-forward fold, settling on **21 features** in the final model.

| Group | Count | Description |
|---|---|---|
| LME / metals | 8 | Same-day settlement returns for copper, aluminum, zinc, SHFE; basis features |
| COMEX gap | 3 | COMEX pit open price features (available at 8:10 AM CT) |
| Asian miners | 23 | Individual overnight returns + multi-day momentum for 8 Asian copper miners |
| Copper (COMEX) technical | ~70 | Price momentum, volatility, volume, candlestick features |
| Cross-asset macro | ~110 | Divergence features vs. USD, rates, oil, equity indices, VIX, metals ETFs |

### 5.2 Selected Features (Final Model, Ranked by Coefficient)

| Feature | Coefficient | Interpretation |
|---|---|---|
| `lme_overnight_ret` | **+3.048** | LME copper Ring settlement return (day t+1). Dominant signal. |
| `lme_overnight_z20` | +0.572 | Same return, scaled to 20-day volatility. Regime context. |
| `next_gap_z20` | +0.338 | COMEX pit open vs. prior close, in sigma units (8:10 AM CT) |
| `next_gap_return` | +0.205 | COMEX pit open return (8:10 AM CT, confirms LME direction) |
| `log_return_10d` | -0.159 | COMEX 10-day momentum — mean reversion tendency |
| `lme_comex_basis_chg` | +0.141 | Daily change in LME-COMEX basis (arb flow signal) |
| `drawdown_20d` | -0.081 | Drawdown from 20-day high (mean reversion in extended trends) |
| `CNY_ret_1d` | +0.063 | Chinese yuan 1-day return (China demand signal) |
| `fed_funds_volatility_20d` | +0.042 | Fed policy uncertainty |
| `asianminers_trend_5_20` | +0.038 | Short/medium-term trend in Asian miner aggregate |
| `curve_2s10s_momentum_20d` | +0.036 | Yield curve slope momentum |
| `volume_pct_change_5d` | -0.033 | Volume trend (declining volume = weaker conviction) |
| `wti_momentum_20d` | -0.030 | Oil momentum (demand/risk signal) |
| `KWEB_div_5d` | -0.028 | China tech divergence from copper (risk-off signal) |
| `CPER_div_20d` | +0.028 | Copper ETF divergence |
| `VIX_mom_20d` | +0.025 | Volatility regime |
| `high_low_range` | -0.024 | Intraday range (mean reversion after volatile days) |
| `volume_pct_change_1d` | +0.024 | Single-day volume spike |
| `lme_comex_basis` | -0.017 | LME-COMEX basis level (persistent arb signal) |
| `SFR_AX_overnight_div` | -0.002 | Sandfire Resources (AU copper miner) divergence |
| `ust_2y_momentum_20d` | -0.001 | 2-year Treasury momentum |

**Key observation:** `lme_overnight_ret` has a coefficient of 3.05 — roughly 5× larger than the second feature. The LME settlement return alone is the model. Everything else adds marginal context.

---

## 6. Model Architecture

### 6.1 Algorithm

**L1-penalized Logistic Regression** (`sklearn.linear_model.LogisticRegression`, penalty='l1', solver='liblinear').

- L1 regularization performs automatic feature selection — it drives irrelevant coefficients to exactly zero, so the model self-prunes from 214 candidates to ~21 active features each fold.
- No tree-based or deep learning methods are used. Logistic regression is appropriate here because the relationship is approximately linear in log-odds and interpretability matters.
- `RobustScaler` (IQR-based) is used instead of `StandardScaler` to handle fat-tailed financial return distributions.

### 6.2 Walk-Forward Validation

To respect the time-series nature of the data, the model uses **expanding-window walk-forward validation** — the only correct approach for sequential financial data.

```
|<---- INIT_TRAIN (1000 days) --->|<-- VAL_WINDOW (252 days) -->|<-- STRIDE -->|... forecast
                                  |<-- val_c (168d) ->|<- val_cal(84d) ->|
```

| Parameter | Value | Rationale |
|---|---|---|
| `INIT_TRAIN` | 1,000 trading days (~4 years) | Minimum history for stable macro features |
| `VAL_WINDOW` | 252 trading days (1 year) | Held-out for hyperparameter tuning + calibration |
| `STRIDE` | 63 trading days (1 quarter) | Re-trains quarterly as new data arrives |
| `val_c` | First 2/3 of val window | C (regularization strength) selected here |
| `val_cal` | Last 1/3 of val window | Isotonic calibration fitted here |

This produces **1,884 out-of-sample predictions** (roughly 2018–2026), none of which were ever used during training.

### 6.3 Regularization

C is selected each fold from the grid `[0.05, 0.1, 0.25, 0.5]` based on AUC on `val_c`. Smaller C = more regularization = fewer features selected. The final model converges on **C = 0.1**.

### 6.4 Probability Calibration

Raw logistic regression probabilities are calibrated using **isotonic regression** (`CalibratedClassifierCV`, method='isotonic') fitted on `val_cal`. This ensures predicted probabilities (e.g., "68% confidence UP") reflect true empirical frequencies rather than raw model scores.

### 6.5 Confidence Intervals

Bootstrap confidence intervals on accuracy use **circular block bootstrap** (block size = 20 days, 3,000 iterations). Standard i.i.d. bootstrap is invalid for autocorrelated time series — the block bootstrap preserves the autocorrelation structure of daily returns.

---

## 7. Results

### 7.1 Summary Statistics

| Metric | Value |
|---|---|
| **Accuracy** | **84.66%** |
| 95% CI (block bootstrap) | [82.1%, 87.2%] |
| **AUC (ROC)** | **0.916** |
| Balanced accuracy | 84.69% |
| Majority baseline (always predict UP) | 51.38% |
| **Lift over baseline** | **+33.3 percentage points** |
| OOS trading days | 1,884 |
| Sample period | ~2018–2026 |
| Features selected (final model) | 21 of 214 candidates |
| Chosen regularization C | 0.1 |

### 7.2 Up vs. Down Breakdown

| | Predicted UP | Predicted DOWN |
|---|---|---|
| **Precision** | 86.1% | 83.2% |
| **Recall** | 83.7% | 85.7% |

The model is slightly better at catching DOWN days (recall 85.7%) than UP days (recall 83.7%), which is useful since missing a down move in scrap pricing tends to be costlier.

### 7.3 Per-Year Stability

| Year | Trading Days | Accuracy | AUC | Notes |
|---|---|---|---|---|
| 2019 | 252 | 82.5% | 0.933 | |
| 2020 | 253 | 88.5% | 0.940 | COVID volatility — LME-COMEX held |
| 2021 | 252 | 86.9% | 0.945 | |
| 2022 | 251 | 90.0% | 0.930 | Highest accuracy year |
| 2023 | 251 | 91.2% | 0.958 | Highest AUC year |
| 2024 | 252 | 87.3% | 0.916 | |
| **2025** | **252** | **63.1%** | **0.701** | **Tariff dislocation — see §9** |
| 2026 | 118 | 90.7% | 0.949 | Partial year (through June) |
| **Overall** | **1,884** | **84.66%** | **0.916** | |

### 7.4 Confidence Gating

The model outputs a calibrated probability. Higher-confidence predictions are more accurate. You can filter to only act on higher-confidence days:

| Min. Confidence | Coverage | Days Committed | Accuracy |
|---|---|---|---|
| Any (≥ 0%) | 100% | 1,884 | 84.66% |
| ≥ 3% from 50/50 | 97.7% | 1,840 | 85.11% |
| ≥ 5% from 50/50 | 95.5% | 1,800 | 85.89% |
| ≥ 8% from 50/50 | 94.2% | 1,775 | 86.20% |

Near-50/50 predictions (confidence < 5%) account for only ~4.5% of days and can be safely treated as "no signal" without meaningful coverage loss.

### 7.5 Comparison to Night-Before Model

| | Night-Before | Morning-Of |
|---|---|---|
| **When** | Prior close (~1 PM CT day t) | By 8:30 AM CT day t+1 |
| **Accuracy** | 52.3% | **84.7%** |
| **AUC** | 0.532 | **0.916** |
| **Lift over 51.4% baseline** | +0.9 pp | **+33.3 pp** |

The night-before model has essentially no useful predictive power. The morning-of model's edge comes entirely from having access to the LME settlement, which is simply not available the night before.

---

## 8. Honest Interpretation

### What the model is really doing

The 84.7% accuracy is real and the methodology is sound. But it is important to understand *why* it works — because it is not traditional machine learning prediction.

**LME copper and COMEX copper are the same underlying commodity priced on different exchanges, 6 hours apart.** When LME copper settles up 1.5% in London at 7:15 AM CT, COMEX copper almost always closes up that afternoon. This is enforced by arbitrage: traders can and do buy/sell across exchanges when they diverge, keeping prices tightly linked.

The model is essentially:

> *"LME settled up this morning → COMEX will close up today (84% of the time)"*

The logistic regression wrapper adds modest value on top of this core signal by incorporating 20 additional features (COMEX gap open, basis changes, macro context), but the `lme_overnight_ret` coefficient alone accounts for the vast majority of the predictive power.

**This is not a weakness** — for the purpose of scrap copper pricing, it is exactly the right signal. The LME is the global copper benchmark. If LME is up, copper is up. That is real, actionable, and commercially useful information.

### What the accuracy numbers mean in practice

- **On a typical day:** LME is clearly up or down 1–3%, confidence is high, accuracy is ~87–90%.
- **On flat days:** LME is near unchanged (±0.2%), the model has low confidence, accuracy approaches 50/50.
- **During dislocations:** LME-COMEX basis blows out (as in 2025), accuracy degrades significantly.

---

## 9. Known Weaknesses

### 9.1 LME-COMEX Dislocation — The 2025 Problem

In 2025, COMEX copper traded at a historically unprecedented premium over LME — at times exceeding $1,500/tonne — driven by market participants front-running potential US import tariffs on copper. This caused the LME-COMEX price relationship to partially decouple.

**Result:** Accuracy fell from 87–91% (2019–2024) to **63.1% in 2025**. The model's core assumption — that LME and COMEX move together — was violated.

**Implication:** The model's reliability is directly tied to whether LME-COMEX basis is behaving normally. During tariff uncertainty or other structural dislocations, treat model output with increased skepticism.

**Mitigation:** Monitor `lme_comex_basis` in the model's output. When basis is unusually elevated or volatile, downweight the model's signal.

### 9.2 The Asian Miners Add No Measurable Lift

A no-shift control test — where the Asian miner features are replaced with same-day (no forward shift) values — produced **84.98% accuracy vs. 84.66% real model accuracy**. The difference is -0.32% (statistically negligible). 

This means the 8 Asian copper miners currently in the feature set (BHP.AX, RIO.AX, S32.AX, SFR.AX, 2899.HK, 0358.HK, 1208.HK, 3993.HK) contribute essentially nothing once the LME signal is present. They could be removed without harming performance. They are retained because they may matter during LME data outages.

### 9.3 The Model Cannot Predict Magnitude

The model outputs a directional probability only. It does not forecast whether copper will be up 0.1% or 3.0%. For large pricing decisions, you may want to supplement the directional signal with a magnitude estimate.

### 9.4 Data Dependency on Bloomberg

LME and SHFE data require Bloomberg Terminal access. If `bloombergCopper.py` has not been run recently, features will be stale. The model will still run (using only COMEX-derived features) but accuracy will degrade significantly.

---

## 10. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
pip install xbbg   # for Bloomberg data fetching
```

### Step 1 — Update Bloomberg data (daily, requires Terminal running)

```bash
python bloombergCopper.py
```

This refreshes four CSVs:
- `data/external/lme_copper.csv` — LME copper Ring settlement
- `data/external/lme_aluminum.csv` — LME aluminum Ring settlement
- `data/external/lme_zinc.csv` — LME zinc Ring settlement
- `data/external/shfe_copper.csv` — SHFE copper front month

### Step 2 — Run the morning model (LR only, ~2 minutes)

```bash
python copper_close_morning_model.py --lr-only
```

Add `--lr-only` to skip the HistGradientBoosting comparison and get results in ~2 minutes.
Without the flag, the full LR + GB comparison runs (~20 minutes).

### Step 3 — Check results

Key outputs in `outputs/copper_close_morning/`:

| File | Contents |
|---|---|
| `metrics_summary.json` | Accuracy, AUC, CI, precision/recall |
| `per_year_stability.csv` | Year-by-year breakdown |
| `walk_forward_predictions.csv` | Every OOS prediction with confidence |
| `selected_features.csv` | Active features and their coefficients |
| `confidence_gate.csv` | Accuracy by confidence threshold |
| `confidence_gate_chart.png` | Chart of confidence vs. accuracy |
| `evaluation_summary.md` | Auto-generated summary narrative |

### For the night-before model

```bash
python copper_close_direction_model.py
```

---

## 11. Output Files

### `walk_forward_predictions.csv`

Row-by-row OOS predictions. Columns:

| Column | Description |
|---|---|
| `date` | Trading date (the target day — COMEX closes this day) |
| `close` | COMEX closing price on this date (actual, for reference) |
| `actual_up` | 1 if copper closed higher than prior day, 0 otherwise |
| `predicted_p_up` | Model's calibrated probability that copper closes UP (0–1) |
| `predicted_dir` | "UP" or "DOWN" based on predicted_p_up ≥ 0.5 |
| `confidence` | Distance from 50/50: \|predicted_p_up - 0.5\| (0 = no signal, 0.5 = max) |
| `correct` | 1 if predicted_dir matched actual_up, 0 otherwise |

### Key Signals for Daily Use

For a given morning's prediction:

- **`predicted_p_up > 0.60`:** Strong UP signal. LME is meaningfully higher.
- **`predicted_p_up < 0.40`:** Strong DOWN signal. LME is meaningfully lower.
- **`0.45 < predicted_p_up < 0.55`:** Weak signal. LME near unchanged. No strong directional bet.

---

*This is a research model built as a pricing proxy. It is not financial advice. Always use judgment alongside model output, particularly during macro dislocations or unusual basis conditions.*
