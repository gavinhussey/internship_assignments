# Steel Direction Model v1 — CME US HRC

Ported from the validated `copper_direction_model_v2` pipeline (same walk-forward,
L1-logistic + isotonic calibration, divergence features, confidence gate, no-shift
control, block-bootstrap CIs). Only the **target series** and the **cross-asset cousins**
change. Predicts: *will CME US HRC steel close higher tomorrow than today?*

## Files
- `steel_close_direction_model.py` — night-before model (guess at prior close).
- `steel_close_morning_model.py` — morning-of model (guess after Asian/EU steel sessions,
  before the ~1pm ET HRC settle; adds HRC's own overnight Globex gap + overnight steelmakers).
- `prefetch_data.py` — one-shot yfinance fetch → `data/raw/steel.csv` + macro/overnight caches.
- `bloombergSteel.py` — **(optional, Bloomberg)** pulls the China/Asia ferrous complex
  (SHFE rebar/HRC, DCE iron ore/coking coal, SGX iron ore) → `data/external/*.csv`.
- `data/raw/steel.csv` — HRC=F OHLCV (Yahoo Finance), 2014→present.

## Run
```bash
python prefetch_data.py                       # fetch + cache all free series (once)
python bloombergSteel.py                       # OPTIONAL: China complex (Terminal required)
python steel_close_direction_model.py         # night-before  -> outputs/steel_close_direction/
python steel_close_morning_model.py --lr-only # morning-of    -> outputs/steel_close_morning/
```

## China complex (the real overnight lead — wired, awaiting data)
`china_features()` in the morning model is steel's analog to copper's LME block. The
SHFE/DCE day sessions close ~3 PM Beijing (~1-2 AM CT) and SGX iron ore settles in
Singapore — all **before** the ~1 PM ET HRC settle — so each is a leak-free overnight
signal (`shift(-1)`, validated by the existing no-shift control). Run `bloombergSteel.py`
to populate the CSVs; the model auto-detects them and adds
`<series>_overnight_ret/_z20` + `china_steel_composite`. Until then the block is a clean
no-op and the model runs on free data. **Verify the Bloomberg yellow-key tickers on your
Terminal first** (see header of `bloombergSteel.py`) — the Chinese-contract symbols vary.

## Results (walk-forward, ~2,024 OOS days, 2018–2026)

| Model | Accuracy | Majority baseline | AUC | Verdict |
|---|---|---|---|---|
| Night-before | 62.9% | **64.1%** | **0.56** | No edge — below baseline; AUC barely over coin flip |
| Morning-of | 73.4% | 64.1% | **0.76** | Real ranking signal, but see caveats |

## The two findings that matter

**1. HRC=F (Yahoo) is extremely thin → the binary target is degraded.**
Median daily volume is ~11 contracts; **35.6% of days are completely flat** (close == prior
close). Flat days bucket into "DOWN", so the class split is 30% UP / 34% DOWN / 36% FLAT and
the majority baseline ("always not-up") is already 64%. The night-before model can't beat
that. This is the "thin, index-settled" risk flagged before the build — it's real.

**2. The morning edge is the HRC self-gap, not cross-asset prediction.**
The morning model's lone meaningful feature is `next_gap_return` (coef 0.86, ~10× everything
else) — HRC's own overnight Globex open vs. prior close. The **no-shift control scores
identically** (0.7332 vs 0.7337): the overnight foreign steelmaker basket adds **nothing**.
So the morning model says, honestly: *"if HRC already opened up this morning, it'll close up."*
Same structural story as copper's morning model (which leaned on LME), except steel has no
arbitraged global anchor — and part of the 73% is inflated by the same staleness (on flat
days, open == close == yesterday's stale print).

## Honest read
This is a faithful clone that **runs end-to-end and the validation machinery correctly
exposes what's real vs. artifact.** The usable takeaway is narrow: HRC's own overnight Globex
move is a weak-to-moderate morning tell (AUC ~0.76, but only ~+9pp over a 64% baseline on a
thin series). Cross-asset steel divergence did *not* clear the bar at the daily horizon.

## Next steps to make this commercially real
- **Fix the target series.** Get a liquid daily HRC mark (CME settlement feed / Barchart /
  a continuous back-adjusted front month) instead of Yahoo's 11-lot front month. Drop or
  separately model flat days.
- **Add the China complex (the real lead).** SHFE rebar/HRC + DCE iron ore + coking coal
  (Bloomberg) — closes ~1–2am CT, the closest steel has to copper's LME anchor. Validate
  with the existing no-shift control before trusting it.
- **Consider the right horizon for the yard.** For ferrous scrap (busheling/shred, monthly
  index-settled), a *weekly* direction model off HRC + the China complex is more honest than
  forcing a daily up/down on a stale contract.

*Research model — not financial advice.*
