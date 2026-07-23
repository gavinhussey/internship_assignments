# Critical Review — Aluminum & Steel Direction Models

*Reviewed 2026-06-29. Scope: `aluminum_direction_model_v1/` and `steel_direction_model_v1/`,
both ported from `copper_direction_model_v2/`. Each ships a **night-before** model (guess at
prior close) and a **morning-of** model (guess after overnight sessions, before the US settle).*

---

## TL;DR verdict

| Model | Headline | Real predictive power? | Status |
|---|---|---|---|
| **Aluminum morning-of** | 92.8% acc / 0.965 AUC | **No.** Forward-looking artifact. | ❌ Corrupted (data + feature leak) |
| **Aluminum night-before** | 55.2% acc | **No.** Below the 55.9% always-down baseline. | ⚠️ Honest but no edge |
| **Steel morning-of** | 73.4% acc / 0.76 AUC | **Mostly no.** ~all of it is HRC's own stale open-gap. | ❌ Inflated by staleness |
| **Steel night-before** | 62.9% acc | **No.** Below the 64.1% baseline. | ⚠️ Honest but no edge |

**Bottom line:** the impressive numbers (especially aluminum's 92.8%) are **not forecasting
skill** — they are a measurement artifact of predicting a *stale, illiquid settlement series*
from a feature that mechanically encodes the answer. The pipeline machinery (walk-forward,
calibration, leakage controls) is actually sound; the corruption lives entirely in **the target
data and one feature**. The two night-before models are honest and correctly report **no edge**.

---

## 1. The smoking gun: the headline accuracy is forward-looking

The morning models' top feature is `next_gap_return = open[t+1] / close[t] − 1` (and its
z-scored sibling `next_gap_z20`). The logic in the code is "the US pit opens after the LME
settles, so the open gap is a leak-free proxy for the overnight move." Timing-wise that is
true. **The problem is what the target series actually is.**

Both price files are stale, barely-traded contracts where the OHLC collapse to a single number
on most days (COMEX `ALI=F` for aluminum, Yahoo `HRC=F` for steel). Measured directly from the
raw CSVs:

| Metric | Aluminum (`ALI=F`) | Steel (`HRC=F`) |
|---|---|---|
| No intraday range (`high==low`) | **84.6%** of days | 47.0% of days |
| Flat close (`close==prior close`) | 16.0% | **35.6%** |
| Zero-volume days | 62.9% | 33.9% |
| Median daily volume | **0 contracts** | 11 contracts |
| **`sign(next_open_gap)` == `sign(close move)`** | **97.5%** | 76.4% |

That last row is the whole story. **When `high==low` 84.6% of the time, the contract does not
trade between the open and the close — so `open[t+1] ≈ close[t+1]`.** Predicting
"`close[t+1] > close[t]`" from `open[t+1]/close[t]` is then a near-tautology: you are not
forecasting tomorrow, you are *observing tomorrow's answer at the open and copying it down*.
The model scores 92.8% because the feature literally agrees with the label 97.5% of the time.

This is forward-looking in substance even though it is not in clock-time: on a real liquid
contract there would be a large, uncertain open→close move and the gap would **not** determine
the close. Here there is no open→close move to forecast.

> This matches the prior finding in memory (`aluminum-ali-stale-target-leakage`): *"COMEX ALI=F
> is a stale LME echo (85% no-range); the morning model's 92.8% is a leakage artifact."*

---

## 2. The model's own control already proves there's no real edge

The morning pipeline includes a **no-shift control**: rebuild the overnight features using the
*same-day* session return instead of the next-day one. If the overnight signal were real, the
control should collapse toward the ~55% baseline. The saved results
(`overnight_shift_validation.csv`):

| | Real model | No-shift control | "Lift" from the shift |
|---|---|---|---|
| **Aluminum** | acc 0.9280 / AUC 0.9650 | acc **0.9360** / AUC 0.9679 | **−0.008 acc / −0.003 AUC** |
| **Steel** | acc 0.7337 / AUC 0.7594 | acc 0.7332 / AUC 0.7546 | +0.0005 acc / +0.005 AUC |

The aluminum control **beats** the real model, and the steel lift is statistical zero. The
overnight Asian/LME shift — the entire premise of the morning model — **adds nothing**. Both
models are riding the stale open-gap, full stop. The selected-feature coefficients confirm it:

- **Aluminum:** `next_gap_z20` (2.30), `lme_overnight_ret` (1.36), `next_gap_return` (0.86) —
  the gap features dominate; `asianminers_trend_5_20` is −0.007 (noise). And `lme_overnight_ret`
  is *also* circular: since `ALI=F` mechanically tracks LME, the LME overnight return ≈ the
  target's own move.
- **Steel:** `next_gap_return` (0.86) is ~10× the next feature (`volatility_5d` 0.07) and
  ~100–1000× the rest. The cross-asset steelmaker basket contributes essentially nothing.

So even the *structure the authors intended to test* (cross-asset divergence, overnight foreign
producers) **failed its own validity check.** To their credit, the code runs that check and the
READMEs say so plainly — but the headline tables still lead with the inflated number.

---

## 3. The night-before models are honest — and have no edge

These never see the open gap (they use only data through `close[t]`), so they are uncorrupted.
They also don't work:

- **Aluminum night-before:** 55.2% acc vs **55.9%** always-down baseline → below baseline.
  AUC 0.596 nudges over 0.50, so there's a whisper of ranking signal, but raw accuracy loses
  to a constant.
- **Steel night-before:** 62.9% acc vs **64.1%** baseline, AUC 0.56 → no edge, barely above a
  coin flip on ranking. The thin target (35.6% flat days bucketed into "DOWN") inflates the
  baseline so high the model can't clear it.

The confidence-gate trick ("only act when |p−0.5| is large") buys a few points on a subset
(aluminum 0.61 on 51% of days), but that is selection on a model with no underlying edge —
it is not a deployable signal.

---

## 4. Is the pipeline itself corrupted? No — and that matters

I checked the machinery for the usual leakage traps and it is **clean**:

- **Target:** `target_up = 1[ close.shift(-1) > close ]`, last row dropped. Standard, correct.
- **Walk-forward:** expanding window; `SimpleImputer` and `RobustScaler` are fit on the **train
  core only** (`np.arange(ce)`), then applied forward. No scaler-fit-on-full-data leak.
- **Hyperparameter / calibration hygiene:** C is selected on `val_c`, isotonic calibration is
  fit on a **disjoint** `val_cal` slice, predictions scored only on the future fold `fc`. This
  is better discipline than most research code.
- **Non-stationary levels** (`ma_*`, raw `volume_ma_*`) are pruned a priori.
- CIs use a circular block bootstrap (block=20) to respect autocorrelation — appropriately
  conservative.

**This is the important nuance:** the validation framework is good enough that it *caught its own
problem* (the no-shift control). The failure is not sloppy ML — it's that **a sound pipeline was
pointed at a degraded target**, and a feature (`next_gap_return`) that is legitimate on a liquid
market becomes a leak on an illiquid one.

---

## 5. Strengths

1. **Methodologically faithful clone of the validated copper pipeline.** Walk-forward,
   per-fold L1 selection, isotonic calibration on a held-out slice, block-bootstrap CIs,
   permutation null, per-year stability — all present and correctly wired.
2. **Built-in honesty checks that work.** The no-shift control and permutation test are exactly
   the right diagnostics, and they *did* expose the artifact. Most "92% accuracy" models ship
   without ever running this.
3. **The READMEs are candid.** Both explicitly flag the thin/stale target and that the morning
   edge is the self-gap, not cross-asset prediction. The documentation is more trustworthy than
   the metrics tables.
4. **Extensible feature scaffolding.** Divergence block, overnight block, LME/China block, and
   a Bloomberg hook (`bloombergSteel.py`) are stubbed and ready for a real data feed.
5. **Reproducible & deterministic** (fixed `RANDOM_STATE`, cached fetches).

## 6. Weaknesses

1. **Corrupted target series.** `ALI=F` (84.6% no-range, 0 median volume) and `HRC=F` (35.6%
   flat, 11-lot volume) are not tradeable price discovery — they are stale index settlements.
   Any binary up/down target built on them is degraded at the source.
2. **Forward-looking headline number.** The 92.8% / 73.4% are open-gap tautologies, not
   forecasts (Sections 1–2).
3. **The intended edge does not exist at this horizon.** Cross-asset divergence and overnight
   foreign producers failed the no-shift control. Daily direction on these metals is ~a coin
   flip, exactly as the copper work found.
4. **Baseline-losing night-before models** presented alongside the inflated morning models can
   mislead a casual reader into thinking the family "works."
5. **Inflated majority baseline from flat days.** Flat closes bucket into "DOWN," pushing the
   always-down baseline to 56–64% and making honest accuracy look worse while also making the
   stale model look better — both directions of distortion come from the same defect.
6. **`lme_overnight_ret` is not independent of the target** for aluminum (ALI mechanically
   echoes LME), so it cannot be trusted as a "cross-market" confirmation.

---

## 7. How to fix this

### Fix #1 — Replace the target with a real, liquid mark (highest priority)
Everything else is downstream of this. Options, best first:

- **Aluminum:** predict **LME aluminum 3-month** directly (you already have
  `data/external/lme_aluminum.csv`). It is the global benchmark and actually trades. Caveat: it
  is a London close, so the "US open gap leads the close" trick no longer applies — which is
  *correct*, because that trick was the leak. You'd forecast LME close[t+1] from genuinely prior
  information. Alternatively, source true OHLCV for COMEX ALI from a real feed (CME/Barchart),
  not Yahoo's stale front month.
- **Steel:** get a liquid daily HRC settlement (CME settlement feed / Barchart continuous
  back-adjusted front month) instead of the 11-lot Yahoo print. Or move to the **CME busheling
  / shred ferrous** contracts that actually match the yard's exposure.

### Fix #2 — Drop or separately model flat days
A flat close is "no information," not "down." Either (a) exclude `close[t+1]==close[t]` rows from
training/scoring, or (b) make it a 3-class problem (up/flat/down) so the baseline isn't gamed by
bucketing flats into down.

### Fix #3 — Re-run the no-shift control as a *gate*, not a footnote
Make the pipeline **refuse to report** a morning number whose no-shift lift isn't positive and
significant. Right now it computes the control but still prints 92.8% up top. Invert that: the
control result should be the headline.

### Fix #4 — Validate `next_gap_return` only on a contract with real intraday range
Add an automated guard: if `mean(high==low) > ~20%` or median volume is trivially small, **block
the gap features and flag the dataset as unsuitable.** This single check would have stopped the
aluminum model from ever reporting 92.8%.

### Fix #5 — Match the horizon to the asset and the business
Daily up/down on monthly-index-settled scrap is the wrong question. For the yard, a **weekly**
direction model on a liquid mark (LME aluminum / a real HRC + the China ferrous complex:
SHFE rebar, DCE iron ore & coking coal, SGX iron ore) is far more honest and more decision-
relevant than forcing a daily binary on a stale series. The `bloombergSteel.py` China block is
the right lead to wire in — but validate it with the no-shift control *before* trusting it.

### Fix #6 — Add genuinely aluminum/steel-specific drivers
Per `DATA_NEEDED.md`: aluminum smelting power cost (TTF / Henry Hub / EU power), LME cancelled
warrants & inventory, Midwest premium; for steel, iron ore & coking coal term structure. These
are real fundamentals copper doesn't have and are the only plausible source of a daily edge.

---

## 8. Recommendation

**Do not deploy or cite either morning model's headline number.** Treat the current aluminum and
steel models as **infrastructure demos** that prove the pipeline ports cleanly — not as working
forecasters. The honest, usable conclusions today are:

1. Day-ahead direction on these specific contracts is ~a coin flip (night-before models, AUC
   0.56–0.60). Copper's finding repeats.
2. The 92.8% / 73.4% morning numbers are open-gap artifacts of a stale target and should be
   retracted from any summary that doesn't sit next to the no-shift control.
3. The path to a *real* model is **fix the target first** (Fix #1), then re-test the China /
   LME leads with the existing controls. Until the target is a liquid mark, no feature work will
   produce a trustworthy number.

*Strengths are real (the framework is sound and self-auditing); the weaknesses are all
downstream of one root cause — the target data. Fix the data and this becomes a legitimate
research model. Leave it and the numbers are fiction.*

---

## 9. Rebuild result — LME aluminum as the target (2026-06-29)

Fix #1 implemented. New scripts (target = LME 3-month settlement, `lme_aluminum.csv`, which
actually trades — 1.4% flat vs 84.6% no-range for ALI):

- `aluminum_direction_model_v1/lme_aluminum_direction_model.py` — night-before
- `aluminum_direction_model_v1/lme_aluminum_morning_model.py` — morning-of

Both reuse the validated walk-forward / L1 / calibration machinery. The morning model
**deliberately excludes** the two leaks: the LME-on-LME `lme_overnight_ret` (which *is* the
target) and `next_gap_return` (no US open precedes the LME settle, and the series trades anyway).
Overnight leads are only foreign sessions that close *before* the ~7:15am ET LME Ring settle:
ASX/HK aluminum producers + SHFE, `shift(-1)`-aligned.

**Walk-forward results (2,051 OOS days, 2018–2026):**

| Model | Accuracy | Baseline | AUC | No-shift control | Shift lift |
|---|---|---|---|---|---|
| Night-before | 51.97% | 50.71% | 0.516 | — | — |
| **Morning-of** | **55.49%** | 50.71% | **0.570** | 51.49% / AUC 0.513 | **+4.0pp acc / +0.057 AUC** |

This is the **inverse of the corrupted model**, and that's the point:
- The headline is now a believable ~+5pp over an honest 50.7% baseline (not a fake 92.8% over an
  inflated 57.8%).
- The **no-shift control collapses to ~baseline** (51.5%, AUC 0.513) — so the edge genuinely
  comes from the overnight shift, not a stale target. Lift is **positive** (it was *negative* in
  the ALI model).
- Top features are the real leads — `shfe_cu_overnight_ret`, `asianminers_5d_mom`,
  `2600_HK_overnight_ret` (Chalco), `S32_AX_overnight_div` — with modest coefficients spread
  across many features, not one tautological monster (old top coef was `next_gap_z20` at 2.30).
- Leak features (`lme_overnight*`, `next_gap*`) confirmed absent from the selected set.

**Honest read:** day-ahead LME aluminum direction is still close to a coin flip; the night-before
model has essentially no edge. The morning model has a **real but small** edge (~55.5%, AUC 0.57)
sourced from the Asian/SHFE overnight session leading the London settle — exactly the size of
signal a legitimate cross-asset metals model should produce. It is now safe to layer the optional
fundamentals (SHFE aluminum, LME cancelled warrants, power cost) on this honest baseline.

---

## 10. Rebuild result — steel (2026-06-29)

Steel has no liquid HRC mark in the repo (the China complex needs Bloomberg, not run), so the
rebuild took two honest angles. New scripts:

- `steel_direction_model_v1/steel_clean_morning_model.py` — keeps HRC=F as the target but
  **deletes the `next_gap_return` open-gap leak** and adds SGX iron ore (TIO=F) as a genuine
  leak-free overnight lead; no-shift control.
- `steel_direction_model_v1/iron_ore_direction_model.py` — uses the only liquid, real-trading
  *ferrous price* available, **SGX iron ore (TIO=F)**, as the target (steel's LME analog;
  2.1% flat vs HRC's 35.6%), night-before structure.

**Walk-forward results:**

| Model | Accuracy | Baseline | AUC | No-shift control | Shift lift |
|---|---|---|---|---|---|
| HRC de-leaked (morning) | 60.7% | **64.1%** | 0.559 | 62.9% / 0.565 | **−2.2pp acc / −0.006 AUC** |
| Iron ore TIO=F (night-before) | 51.9% | 51.7% | 0.529 | — | — |

**This is the decisive confirmation that the original steel 73.4% was pure leak.** Once
`next_gap_return` is removed:
- The HRC morning model falls **below its own majority baseline** (60.7% < 64.1%) and the
  no-shift control *beats* it (negative shift lift) — so the overnight steelmakers + iron ore
  add nothing at the daily horizon. The 73.4% was entirely the open-gap tautology, exactly as
  Section 2 predicted. Selected features confirm the leak is gone (top feature is now
  `close_location` at −0.28; no `next_gap*`/`us_open*` survive).
- The clean liquid iron-ore target is an honest **coin flip** (51.9% vs 51.7%, AUC 0.53) — a
  whisper of ranking signal, no usable directional edge from free data.

**Why steel ≠ aluminum.** The aluminum rebuild found a real edge because the LME is a single
liquid global benchmark that *settles after* its Asian/SHFE leads — a clean ~5-hour lead.
Steel has no such anchor: HRC=F is thin and US-settled with no liquid global cousin, and iron
ore's natural leads are Asian/contemporaneous. So the honest conclusion stands: **a credible
steel direction model genuinely requires the China ferrous complex** (SHFE rebar/HRC, DCE iron
ore & coking coal, SGX iron ore via Bloomberg) — free-data daily direction on steel is ~noise.
The iron-ore TIO=F target is at least a clean, uncorrupted base to build that on.
