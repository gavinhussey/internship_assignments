# Aluminum Direction Model v1

Predicts the **next day's up/down direction** of the aluminum price, ported from the
validated `copper_direction_model_v2` pipeline. Same machinery (L1-pruned logistic
regression, isotonic calibration, expanding-window walk-forward, permutation testing,
deployable confidence gate); the only things that change are aluminum's cross-asset
"cousins" and the overnight leads.

> ⚠️ Research/educational model, not financial advice. Day-ahead direction is close to a
> coin flip; a real edge here is small. Validate before trusting any number.

## The two models (same question, different timing)

| Model | File | When you guess | Extra info it may use |
|---|---|---|---|
| **Night-before** | `aluminum_close_direction_model.py` | at today's close (~24h ahead) | aluminum price block + cross-asset divergence |
| **Morning-of** | `aluminum_close_morning_model.py` | next morning (~5h ahead) | the above **+ LME aluminum overnight settlement + Asian producers** |

The morning model is expected to be the stronger one: **the LME is the global aluminum
benchmark and settles (~7am ET) before the US session closes**, so the LME overnight move is
a large, clean, leak-free lead — an even better setup than copper's Asian-miner lead.

## Status

Pipeline built and import-clean. **It needs one input from you to run: the aluminum primary
price file.** See **`DATA_NEEDED.md`** — that's the to-do list.

## What's tailored for aluminum vs. copper

- **Divergence cousins** (`DIVERGENCE_TICKERS`): Alcoa (AA), Century (CENX), Kaiser (KALU),
  Constellium (CSTM), Norsk Hydro (NHYDY), Chalco (ACH), Rio/BHP, plus base-metal & mining
  ETFs (DBB, PICK, XME, SLX). China demand proxies (FXI/MCHI/ASHR/KWEB) and macro
  (VIX/DXY/TNX/CNY) retained.
- **Overnight producers** (`OVERNIGHT_TICKERS`): BHP.AX, RIO.AX, S32.AX, China Hongqiao
  (1378.HK), Chalco (2600.HK), Rusal (0486.HK).
- **LME block**: LME **aluminum** is the primary overnight lead; LME copper/zinc and SHFE
  copper are cross-metal context.
- Identical leakage controls, walk-forward protocol, and diagnostics as the copper model.

## How to run (after adding `data/raw/aluminum.csv`)

```bash
pip install -r requirements.txt
python aluminum_close_direction_model.py   # night-before  -> outputs/aluminum_close_direction/
python aluminum_close_morning_model.py     # morning-of     -> outputs/aluminum_close_morning/
python overfitting_check.py                # honesty checks -> outputs/aluminum_close_direction/
```

Outputs per model: `evaluation_summary.md`, `selected_features.csv`,
`walk_forward_predictions.csv`, `metrics_summary.{csv,json}`, `model.pkl`, and charts.

## Provenance

Adapted from `copper_direction_model_v2/` (see that folder's README for the full
plain-English explanation of every feature and the validation methodology — all of it
applies here unchanged except the metal-specific tickers).
