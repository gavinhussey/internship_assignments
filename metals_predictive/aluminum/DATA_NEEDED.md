# Inputs needed from you (to finish wiring up the aluminum model)

The whole pipeline is built and imports cleanly. It is missing **one required input** and
has **a few optional ones** that would sharpen it. Here's the list, most important first.

---

## 1. REQUIRED — the aluminum primary price series  → `data/raw/aluminum.csv`

This is the equivalent of copper's `data/raw/copper.csv`. It defines the target
(`close[t+1] > close[t]`). Daily, with at least `Date,Close`; `Open,High,Low,Volume`
strongly preferred (Open is what powers the morning model's LME-gap proxy).

I need you to tell me **which aluminum benchmark you want to predict**, because aluminum —
unlike copper — has no single dominant US futures market. Options:

| Option | What it is | Pros | Cons |
|---|---|---|---|
| **A. LME aluminum 3-month** | The global benchmark | Most liquid/meaningful; you already have the close in `data/external/lme_aluminum.csv` | London close, not a US session → the "US open gap / LME-leads-US" morning trick doesn't apply; and we'd need OHLC, not just close |
| **B. COMEX aluminum (ALI)** | CME's USD/tonne US contract | US session → the morning model's LME-leads-US logic works exactly as for copper | Thinner/younger than copper; need to source OHLCV |
| **C. An aluminum ETF/ETN** (e.g. a broad metals or aluminum-linked fund) | US-listed, easy OHLCV via yfinance | Free, daily OHLCV out of the box | A proxy, not the metal itself; tracking error |

**What I need from you:** either (a) drop a CSV at `data/raw/aluminum.csv` in the format in
that folder's README, **or** tell me which option above you want and where you can get the
data (a provider/terminal export, a NASDAQ Data Link dataset name, or "just use yfinance
ticker X"). If you want, I can wire an auto-downloader for option B or C.

> Note on units: if your primary is in **USD/tonne**, leave `LME_UNIT_DIV = 1.0` in
> `aluminum_close_morning_model.py`. If it's in **$/lb**, set it to `2204.62` so the
> LME-vs-US basis features compare like-for-like. (Returns are unit-invariant; only the
> basis cares.) Tell me the units and I'll set it.

---

## 2. ALREADY INCLUDED (no action needed)

These came over from the copper project and are reused as-is:

- `data/external/lme_aluminum.csv` — LME aluminum daily close → the morning model's
  **primary overnight lead** (the biggest expected edge).
- `data/external/lme_copper.csv`, `lme_zinc.csv`, `shfe_copper.csv` — cross-metal context.
- `data/external/external_drivers.csv` — DXY, rates, VIX, oil, etc. (point-in-time merged).
- Cross-asset "cousin" equities + China/sector ETFs and the overnight Asian aluminum
  producers are **downloaded automatically from yfinance** on first run and cached. Needs
  internet once. (Tickers are listed/commented in the two model files.)

---

## 3. OPTIONAL — would likely add real signal (aluminum-specific)

Aluminum's economics differ from copper's in ways worth exploiting in v2:

1. **Energy / power cost** — smelting is ~30-40% electricity. A daily power or natural-gas
   proxy (e.g. EU TTF / Henry Hub, or a European power index) is a genuinely *aluminum-
   specific* driver copper doesn't have. Give me a daily series (or a yfinance/Data Link
   ticker) and I'll add it as a driver + divergence feature.
2. **LME aluminum inventory / cancelled warrants** — daily LME warehouse stocks and the
   cancelled-warrant ratio are a classic aluminum tightness signal. Daily CSV `[date, value]`.
3. **SHFE aluminum** (not just SHFE copper) — closes ~1-2am ET, a clean China overnight
   lead. Daily CSV `[date, shfe_al_close]` and I'll wire it next to the SHFE copper slot.
4. **Midwest premium** (US physical delivery premium) — if you can get a daily/weekly series,
   it captures US-specific supply tightness.

None of these block a first run — they're upgrades.

---

## Once I have item 1, I will:
1. Run the night-before model → baseline accuracy/AUC + walk-forward + confidence gate.
2. Run the morning model → the LME-lead version (expected to be the stronger model).
3. Run `overfitting_check.py` → per-year stability, config robustness, permutation null.
4. Report honest numbers and we decide whether the edge is real before adding item-3 features.
