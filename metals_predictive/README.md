# Metals Directional Models — What We Tried, and Why Most of It Didn't Work

**Status: archived research record, not a working system.** This folder is a copy of
everything built during the copper/aluminum/steel daily-direction modeling effort
(2026 Q2–Q3), preserved specifically so the failure modes below don't get re-invented.
Read this before starting a new directional-model attempt on any commodity.

---

## 1. What we were trying to do

Predict, once a day, whether a metal's futures/settlement price would close **up or down**
the next session — first from data available "night-before" (prior close only), then from
data available "morning-of" (after overseas markets have moved but before the US settle).
The idea: a small, honest directional edge could feed scrap-buying/selling decisions.

Three metals were attempted, all cloned from one pipeline (`copper/`):

| Metal | Target series | Files |
|---|---|---|
| Copper | COMEX HG=F | `copper/copper_close_direction_model.py`, `copper_close_morning_model.py`, `copper_close_morning_v3.py` |
| Aluminum | COMEX ALI=F → later LME aluminum | `aluminum/archive_old_comex/*`, `aluminum/lme_aluminum_*_model.py` |
| Steel | Yahoo HRC=F → later SGX iron ore | `steel/archive_old_hrc/*`, `steel/iron_ore_*_model.py`, `steel/steel_clean_morning_model.py` |

Plus a non-modeling piece, `aluminum_scrap_tracker/`, which just scrapes/logs scrap
premiums — not part of the failure discussed here.

The pipeline itself (walk-forward expanding window, per-fold L1 feature selection,
isotonic calibration on a disjoint slice, block-bootstrap CIs, permutation nulls,
a no-shift control) is genuinely sound engineering. **The failures below are not bugs in
that machinery — they are what happens when a sound pipeline is pointed at the wrong
target, or asked a question the market doesn't have an answer to.**

---

## 2. The one result that was real

**Copper morning-of model: 84.7–86.7% accuracy, AUC ~0.92–0.94.** This held up under audit
(`copper/MODEL_AUDIT_REPORT.md`) and a full no-shift control (control collapses to ~53%,
i.e. the night-before baseline — a +33pp true lift). The mechanism: LME copper's Ring
settlement (~7:15am ET) precedes the COMEX settle (~1pm ET) by ~6 hours on the *same*
underlying commodity, and COMEX HG=F is genuinely liquid (1.8% no-range days, median volume
535). This is a real, leak-free arbitrage-window lead, not a forecast of anything new — it's
"LME already told you which way copper moved this morning."

**This is the one thing worth reusing — and the one thing hardest to reuse.** It only exists
because copper has a liquid global anchor (LME) that settles hours before the US target. It
does not automatically transfer to other metals (see below). The night-before copper model
(technicals only, no LME) scores 52.3%/AUC 0.53 — real but marginal, exactly what you'd
expect from daily direction on a liquid, efficient market.

---

## 3. Everything else, and why it failed

### 3.1 Aluminum and steel morning models were fake — a stale-target tautology, not skill

The first-pass aluminum (COMEX `ALI=F`) and steel (Yahoo `HRC=F`) morning models reported
92.8% and 73.4% accuracy respectively. Both were artifacts:

- `ALI=F` has **no intraday range on 84.6% of days** and a median volume of **0 contracts**.
  When a contract doesn't trade between open and close, `open[t+1] ≈ close[t+1]`. The
  headline feature (`next_gap_return = open[t+1]/close[t] - 1`) then agrees with the label
  97.5% of the time — the model isn't forecasting tomorrow, it's reading tomorrow's answer
  off the opening print and copying it down.
- `HRC=F` (Yahoo) is thinner still: 35.6% flat-close days, ~11-lot median volume.
- **The models' own no-shift control caught this and was ignored in the headline.**
  Rebuilding the "overnight" features from same-day (rather than next-day) data should
  destroy a real signal; instead the aluminum control **beat** the real model (+0.008 acc)
  and the steel control was statistically flat (+0.0005 acc). That's the pipeline correctly
  proving there was nothing there — see `ALUMINUM_STEEL_MODEL_REVIEW.md` §2.

**Lesson: an 85–95% daily-direction accuracy claim on a commodity is not a discovery. It is
almost always a leaked or tautological feature. Treat it as a bug to find, not a result to
report, until a no-shift control (or equivalent) proves otherwise.**

### 3.2 Fixing the target didn't produce a fixable model for steel

After swapping in a real liquid mark and deleting the gap-leak feature:

| Model | Accuracy | Baseline | AUC | No-shift lift |
|---|---|---|---|---|
| Steel, HRC de-leaked (morning) | 60.7% | 64.1% | 0.559 | **−2.2pp** (control beats model) |
| Steel, SGX iron ore TIO=F (night-before, clean liquid target) | 51.9% | 51.7% | 0.529 | — |
| Aluminum, LME target (night-before) | 51.97% | 50.71% | 0.516 | — |
| Aluminum, LME target (morning-of) | **55.49%** | 50.71% | **0.570** | **+4.0pp / +0.057 AUC** (genuinely positive) |

Once the tautology is removed, **steel has nothing left — even below its own baseline.**
Iron ore (steel's cleanest available liquid proxy) is a coin flip both night-before and
morning-of. Steel has no liquid global anchor that settles ahead of a US steel mark the way
LME leads COMEX copper; its natural overnight leads (SHFE, DCE) are Asian-hours-contemporaneous,
not a clean pre-settle lead. **A credible steel model needs the China ferrous complex
(SHFE rebar/HRC, DCE iron ore & coking coal) via a real data feed (Bloomberg) — free-data
daily direction on steel is noise, full stop.** See `ALUMINUM_STEEL_MODEL_REVIEW.md` §9–10.

Aluminum is the partial exception: once retargeted to LME aluminum (which actually trades —
1.4% flat vs. 84.6% no-range for ALI=F), the morning model found a **small, genuine** edge
(+4pp over baseline, no-shift control collapses to baseline as it should) sourced from
Asian/SHFE producers settling before the London Ring. Night-before aluminum, like night-before
steel, has no edge.

### 3.3 Night-before (technicals-only) direction is a coin flip across the board

Every night-before model we built — copper (52.3%), aluminum (52.0%), steel (62.9% vs. 64.1%
baseline), iron ore (51.9%) — landed at AUC 0.51–0.53, indistinguishable from noise. Daily
direction, one day ahead, from price/volume technicals and cross-asset divergence alone, does
not have exploitable edge on any of these three metals. The only edge found anywhere in this
project came from a same-day cross-exchange timing lead (LME settling before COMEX/before
London), not from forecasting.

---

## 4. Recurring failure modes (the actual reusable lesson)

1. **Check target liquidity before building anything.** `mean(high==low)`, `mean(close==
   prior close)`, and median volume take one line to compute and would have blocked the
   aluminum and steel morning models before they ever reported a number. Rule of thumb used
   after the fact: no-range >20% or trivial median volume → the series is not tradeable price
   discovery, don't build a binary target on it.
2. **A gap/overnight feature is only leak-free if the underlying market actually moves
   between that gap and the close.** The same feature (`next_gap_return`) was legitimate on
   liquid COMEX copper and a near-perfect tautology on stale ALI=F/HRC=F. The feature isn't
   the problem — the target's liquidity is.
3. **Always run a no-shift (or equivalent placebo) control, and make it a gate, not a
   footnote.** It's cheap, it was already implemented in every model here, and it correctly
   flagged both fake results. The mistake was computing it and still leading with the
   inflated headline number.
4. **Flat closes bucketed into "down" inflate the majority baseline and distort both
   directions of comparison** — it makes an honest model look artificially bad *and* a stale
   model look artificially good, from the same defect. Either drop flat days or make it a
   3-class problem.
5. **A validated pipeline does not transfer its edge by cloning to a new asset.** The
   copper morning model's edge is a specific market-structure fact (LME settles ~6h before
   COMEX, same commodity). Porting the code to aluminum worked because aluminum has the same
   structural fact via LME. Porting it to steel did not, because steel has no analogous liquid
   global anchor — the code ran fine, the assumption underneath it didn't hold.
6. **Validation-set double use is a real, separate bug to watch for**: fitting both the
   C-grid search and the probability calibration on the same held-out slice contaminates the
   calibration. Fixed here by splitting `val` into `val_c` (2/3) and `val_cal` (1/3) —
   confirmed in `copper/MODEL_AUDIT_REPORT.md` §1.2.

---

## 5. If you're about to build another one of these

- Don't start from "clone the copper pipeline onto metal X." Start by asking whether metal X
  has a liquid target *and* a structural, pre-settle information lead analogous to LME→COMEX.
  If neither exists, this class of model has nothing to find at a daily horizon.
- Run the liquidity check (§4.1) and the no-shift control (§4.3) *before* looking at the
  headline accuracy, not after.
- Expect night-before, technicals-only, daily direction to be a coin flip. That's the honest
  finding here, repeated four times independently. It is not this project's failure to fix —
  it's very likely a fact about efficient daily markets.
- The one path with real promise left unexplored: wiring in the China ferrous complex (SHFE/
  DCE/SGX via Bloomberg) for steel, and layering LME cancelled-warrants/power-cost fundamentals
  onto the now-honest aluminum baseline. Both require paid data feeds this project didn't have
  wired up (`steel/bloombergSteel.py`, `aluminum/DATA_NEEDED.md` are the stubs).

---

## 6. Where the evidence lives in this folder

- `ALUMINUM_STEEL_MODEL_REVIEW.md` — full audit of the aluminum/steel failure and rebuild.
- `copper/MODEL_AUDIT_REPORT.md` — full audit confirming the copper morning model is clean.
- `copper/MODEL_DOCUMENTATION.md`, `copper/README.md` — plain-English writeup of the one
  model that worked, including its 2025 exception.
- `*/outputs/*/evaluation_summary.md`, `metrics_summary.csv`, `overnight_shift_validation.csv`
  — the raw numbers behind every table above, per model variant.
- `steel/archive_old_hrc/`, `aluminum/archive_old_comex/` — the original, corrupted-target
  versions, kept so the before/after comparison is auditable rather than asserted.
