# Copper Direction Models — A Complete, Plain-English Guide

This guide explains everything about these models, assuming you've never heard of them and
don't have a finance or coding background. By the end you'll understand **what they do, how
every piece works, and why each ingredient is there.** Read it top to bottom.

> ⚠️ These are **research/learning models, not financial advice.** They are right about
> 54–60% of the time — better than a coin flip, but nowhere near a crystal ball. Do not
> trade real money based on them.

---

## 1. The one question these models answer

Copper is a metal that gets traded on a big exchange (called **COMEX**). Its price goes up
and down every day. We try to answer exactly one yes/no question:

> **Will copper's price be higher at the end of tomorrow than it was at the end of today?**

That's it. We are **not** trying to guess the *price* (like "$4.12"). We only guess the
**direction**: UP or DOWN. In code this is written as:

```
target_up = 1 if (tomorrow's closing price > today's closing price) else 0
```

- `1` means "UP" (tomorrow closes higher).
- `0` means "DOWN" (tomorrow closes lower or the same).

**Why direction instead of the exact price?** Guessing the exact price is a trap: a lazy
guess of "tomorrow ≈ today" looks accurate (the price barely moves day to day) but tells
you nothing useful. Direction is the honest, hard question — and the one a person betting
"up or down" actually cares about.

**Why is this hard?** Daily price moves are *almost* random — close to a coin flip. A model
that's right 54% of the time has found a small but real edge. Getting much above that is
extremely difficult, because thousands of professionals are trying to do the same thing.

---

## 2. The big picture: how the model works in 6 steps

Think of the model like a weather forecaster for copper. It doesn't *know* the future — it
looks at lots of clues from the past and present and makes an educated guess.

1. **Gather data.** Copper's daily prices, plus the prices of related things (mining
   company stocks, the Chinese currency, oil, the stock-market "fear gauge," etc.).
2. **Turn data into clues ("features").** Raw prices aren't useful by themselves. We
   compute things like "how much did copper rise this week?" or "are mining stocks moving
   differently than copper?" Each clue is a number. (Section 4 explains every clue.)
3. **Learn from history.** We show the model thousands of past days — each day's clues *and*
   what actually happened the next day — so it can find which clues tend to come before UP
   days vs DOWN days.
4. **Make a prediction.** For a new day, the model combines all the clues into a single
   probability, like "62% chance tomorrow is UP."
5. **Turn the probability into a guess.** Above 50% → guess UP. Below 50% → guess DOWN.
   (Optionally, only "commit" when the model is *confident* — see Section 7.)
6. **Test it honestly.** We check how often it was right on days it had *never seen during
   learning*, the same way a real forecaster is judged on future weather, not past weather.

---

## 3. Where the data comes from

- **Copper prices**: `data/raw/copper.csv` — the daily open, high, low, close, and trading
  volume of COMEX copper futures, back to 2015.
- **Big-picture economic data** (`data/external/external_drivers.csv`): the US dollar's
  strength, interest rates, an oil price, a stock-market fear gauge, etc.
- **Related markets** (downloaded automatically from Yahoo Finance and cached in
  `data/external/macro_cache/`): mining-company stocks and other metals/currencies.

Every piece of data is a daily number lined up by date. The code is careful to only ever
use information that would have been **known at the time** — never future information (more
on this in Section 6, "no cheating").

---

## 4. The features (the clues) — explained one by one

A **feature** is just a single number that describes something about the market on a given
day. The model uses ~160 candidate features, but it automatically throws most away and keeps
only ~12–17 that actually help (Section 5 explains how). Below, the features are grouped by
idea. For each, we say **what it measures** and **why it might hint at tomorrow's direction.**

A few words you'll see a lot:
- **Return** = percent change in price. "Copper rose 2%" is a return of +0.02.
- **Momentum** = "is the price higher or lower than it was N days ago?" (a measure of trend).
- **Moving average** = the average price over the last N days. It smooths out daily noise so
  you can see the underlying trend, like averaging your test scores.
- **Volatility** = how wild/jumpy the price has been recently (big swings = high volatility).

### Group A — Copper's own behavior

These describe what copper itself has been doing.

- **Returns over many time spans** (`log_return_1d, 2d, 3d, 5d, 10d, 20d`, and
  `simple_return_1d/5d/20d`): how much copper has gone up or down over the last day, week,
  month, etc. *Why it helps:* prices have mild "memory" — short bursts can continue
  (momentum) or snap back (mean-reversion), and the model learns which.
  *(`log_return_10d` — copper's roughly two-week trend — is one of the kept features.)*
- **Momentum** (`momentum_5d, 10d, 20d, 60d`): is copper higher than it was N days ago?
  *Why it helps:* a strong trend sometimes keeps going; a stretched one sometimes reverses.
- **Trend position** (`price_vs_ma5/20/60`, `ma5_vs_ma20`, `ma20_vs_ma60`): how far today's
  price is above or below its recent average, and whether short-term averages are above
  long-term ones. *Why it helps:* being far above the average can mean "overheated"
  (due for a pullback) or "strong" (trend intact).
- **Volatility** (`volatility_5d, 10d, 20d, 60d`): how jumpy copper has been over each span.
  **Volatility spikes** (`volatility_spike_5_20`, `volatility_spike_10_60`): is short-term
  jumpiness rising faster than long-term? *Why it helps:* sudden calm vs. sudden chaos
  changes how the next move tends to behave. *(`volatility_spike_5_20` is a kept feature.)*
- **Drawdown** (`drawdown_20d, 60d`): how far below its recent peak copper is sitting.
  *Why it helps:* deep pullbacks can signal either fear (more falling) or a bounce.
- **Candle shape** (today's bar):
  - `high_low_range`: how wide today's price swing was. *Big range = an emotional day.*
  - `close_open_return`: did copper rise or fall *within* today (open to close)?
  - **`close_location`** (a kept feature): where today's close landed inside the day's
    range. Near the top = buyers won the day; near the bottom = sellers won. *Why it helps:*
    who "won" today gives a faint hint about tomorrow.
  - `gap_return`: did copper open higher or lower than yesterday's close (an overnight gap)?
- **Volume** (`volume_pct_change_1d/5d`, `volume_vs_ma20`, `volume_zscore_20d`): how much
  copper was traded vs. normal. *Why it helps:* a big move on heavy volume is taken more
  seriously than one on light volume.
- **Economic backdrop** (from the drivers file — US dollar `dxy`, interest rates `ust_2y`,
  `ust_10y`, the yield curve, the `vix` fear gauge, oil `brent`/`wti`, `aluminum`): for each
  we compute its recent returns, momentum, and volatility. *Why it helps:* copper is an
  industrial metal, so it rises and falls with the health of the global economy.
  **Oil features are especially useful** — `brent_momentum_20d`, `brent_return_1d`, and
  `brent_volatility_20d` are all kept, because oil and copper both track global demand.
  The **stock-market fear gauge** `VIX_vol_20d` is kept too: when investors get scared,
  industrial metals usually suffer. The **Chinese yuan** `CNY_ret_1d` is kept because China
  buys about half the world's copper, so the yuan and copper move together.

### Group B — Cross-asset "divergence" (the most important idea)

This is the heart of the model, so it's worth a careful explanation.

Copper has "cousins" — things that should move *with* it because they depend on the same
forces: **mining-company stocks** (Freeport `FCX`, Southern Copper `SCCO`, BHP, Rio Tinto,
Vale), **copper-miner ETFs** (`COPX`), **China stock funds** (`FXI`, `MCHI`, `ASHR`,
`KWEB`), **base-metal ETFs** (`DBB`, `PICK`, metals & mining `XME`, steel `SLX`), and a
**copper ETF** (`CPER`).

A **divergence** feature measures: *did the cousin move differently than copper today?*

```
divergence = (cousin's return) − (copper's return)
```

**Why this is powerful:** sometimes mining stocks or Chinese markets react to news *before*
copper's own price fully catches up. If copper miners jumped but copper itself didn't, that
gap ("divergence") can hint that copper is about to follow. It's like noticing the smoke
before you see the fire. For each cousin we compute:
- `_div_1d, _div_5d, _div_20d`: the divergence over 1, 5, and 20 days.
- `_div_z20`: a **z-score** of the divergence — i.e., "how unusual is today's gap compared
  to a normal day?" (A z-score of 2 means "much bigger than usual.") *(`XME_div_z20` — the
  metals-&-mining sector pulling away from copper — is a top kept feature.)*
- `_ratio_mom20`: the trend in the cousin's price *relative to* copper. *(`CPER_ratio_mom20`
  is kept.)*

The kept divergence features include `XME_div_z20`, `VALE_div_5d`, `DBB_div_5d`,
`CPER_div_5d` — a mix of the metals sector, a big miner, base metals, and the copper ETF all
"disagreeing" with copper in ways that slightly predict its next move. **This divergence
idea is the single thing that lifted the model above a coin flip.**

We also include a few **plain** market features (not divergences): the fear gauge `VIX`, the
dollar `DX-Y.NYB`, the 10-year interest rate `^TNX`, and the Chinese yuan `CNY=X`, each as
returns/momentum/volatility.

### Group C — The "overnight" signal (only in the morning model)

This is the special ingredient that makes the second model much better, and it relies on a
clever (but completely fair) timing trick. **Read this slowly.**

The world's stock markets are open at different times. While it's the middle of the night in
New York, it's daytime in **Asia and Australia**, and the big copper-mining companies there
are actively trading: **BHP and Rio Tinto in Australia**, and **Zijin Mining (`2899.HK`),
Jiangxi Copper (`0358.HK`), and MMG (`1208.HK`) in Hong Kong**.

Here's the timeline for any given day (all times New York time):
- ~1:00 am – 4:00 am: **Asian/Australian miners finish trading** for the day.
- ~1:00 pm: **COMEX copper in New York closes** for the day.

So by the time New York's copper market closes in the afternoon, those Asian miners already
finished hours earlier *that same morning*. Asian markets often react to overnight world
news first, and **copper's New York price then tends to follow.** It's like seeing how a
movie was reviewed overseas before it opens in your country.

The overnight features capture this:
- `2899_HK_overnight_ret`, `0358_HK_overnight_ret`, etc.: how each Asian miner moved in its
  most recent (this-morning's) session.
- `asianminers_overnight_mean`: the average move across all the Asian miners — a single
  "what did Asia do last night?" number.
- `..._overnight_div`: the Asian miner's overnight move *minus* copper's recent move (the
  same divergence idea, applied to the overnight session).

**These overnight features have the biggest influence in the morning model.** The top two
are `2899_HK_overnight_div` (Zijin Mining's overnight move vs copper) and
`asianminers_overnight_mean`. In plain terms: *if Asian copper miners had a strong morning,
copper in New York is meaningfully more likely to close up that afternoon.*

⚠️ **The important catch (timing):** this only works if you make your guess **in the
morning**, after Asia has closed but before New York closes. If you have to lock your guess
in the *night before*, those Asian sessions haven't happened yet, so you can't use them.
That's exactly why there are **two models** (Section 8).

---

## 5. How the model decides which clues to keep (feature selection)

We start with ~160 candidate features, but most are noise — using all of them would make the
model "memorize" random quirks instead of learning real patterns (this is called
**overfitting**). So we do two things:

1. **Throw out junk on purpose.** Raw price *levels* (like "the 20-day average price was
   $3.84") are removed, because copper traded around $3 in the past and $6 now — a level the
   model learned in training would be meaningless today. We also remove exact duplicate
   features.
2. **Let the math prune the rest.** We use a technique called **L1 (Lasso)** that forces the
   model to set most feature weights to exactly **zero** unless a feature genuinely earns its
   place. Out of ~160 candidates, it keeps only **~12 (night-before)** or **~17 (morning)**.
   Fewer, stronger clues = a simpler model that holds up better on new data.

---

## 6. The model itself, and why we trust the test

### What kind of model is it?

It's a **logistic regression** — one of the simplest, most transparent prediction methods.
Think of it as a weighted scorecard:

```
score = (weight₁ × feature₁) + (weight₂ × feature₂) + ... 
probability_up = squish(score)   # "squish" turns any score into a 0–100% probability
```

Each kept feature gets a **weight** (its `coefficient`). A positive weight means "more of
this clue → more likely UP"; negative means "→ more likely DOWN." For example, in the
morning model, `2899_HK_overnight_div` has a large positive weight (Asian miners up →
copper more likely up), while `brent_momentum_20d` has a negative weight.

**Why such a simple model?** We tried fancier ones (random forests, gradient boosting). The
simple linear model **won.** When the real signal is weak, fancy models tend to chase noise.
Simple and honest beat complicated and overfit.

Two finishing touches:
- **Calibration (isotonic):** makes the model's stated probabilities trustworthy, so when it
  says "60%," it really is right about 60% of the time.
- **Per-fold tuning:** the strength of the L1 pruning is re-chosen for each time period using
  only past data, so we never tune it using the answers we're trying to predict.

### "No cheating" rules (avoiding data leakage)

The #1 way these models go wrong is accidentally using future information. We prevent it:
- The answer (tomorrow's price) is never an input.
- Every clue uses only data available **at guess time**.
- The "smoothing" steps (filling gaps, scaling numbers) are computed from **past data only**,
  separately for each time period.

### How we test honestly: walk-forward

We never test on data the model studied. Instead we **walk forward through history**:

> Train on 2015–2018 → predict 2019. Then train on 2015–2019 → predict 2020. And so on.

Each prediction is for a day the model had never seen — just like real life, where you only
ever predict the future. We collect ~1,884 such "never-seen" predictions (2019–2026) and
measure accuracy on those. This is the only number that matters.

### Proving the edge is real (the permutation test)

To be sure the model isn't fooling us, we **shuffle the answers randomly** and re-run
everything. With scrambled answers there's nothing real to learn, so a sound model should
score ~50% (a coin flip). Ours does: with shuffled answers it gets **~50%**, but with the
real answers it gets **54–60%**. That gap is the proof the edge is real, not luck or a bug.

---

## 7. Reading the model's confidence (the "confidence gate")

The model outputs a probability, and not all days are equally clear. On days when it's only
51% sure, it's basically guessing. On days it's 65% sure, it's more reliable.

A **confidence gate** means: *only act on the days the model is confident about.* You still
look at every day, but you treat low-confidence days as "no strong opinion." For the
night-before model, accuracy is fairly flat across confidence levels (~54%), so the gate
mostly just tells you which calls to weight more. This is the honest, usable version of a
"dead-band" — we trust the model's conviction, not the size of recent price moves (which we
tested and found don't help pick the predictable days).

---

## 8. The two models — same question, different timing

Both answer "will copper close up tomorrow?" The only difference is **when you're allowed to
guess**, which changes **what information is fair to use**:

| Model | File | When you guess | What it may use | Accuracy | AUC |
|---|---|---|---|---|---|
| **Night-before** | `copper_close_direction_model.py` | at today's close (~24 hours ahead) | everything through today | **~53.9%** | 0.54 |
| **Morning-of** | `copper_close_morning_model.py` | tomorrow morning (~5 hours ahead) | the above **+ last night's Asian markets** | **~59.8%** | 0.63 |

The morning model is far more accurate **because** it's allowed to peek at the overnight
Asian miners — a fresher, shorter forecast. It's not cheating (those markets really did close
before copper does), but it *is* a different, easier question. **Use the model that matches
when you actually have to commit your guess.**

The morning model reuses the night-before model's entire pipeline (it `import`s it) and just
adds the overnight features — so there's no duplicated logic.

---

## 9. Results — the honest scorecard

**Night-before model** (predicting ~24 hours ahead, 1,884 test days):
- Accuracy **53.9%** (a coin flip is 50%; "always guess up" is ~51.4%).
- AUC **0.536** (0.50 = no skill; this is a small but statistically real edge).
- Honest caveat: only a hair better than always guessing "up," and it was inconsistent
  year-to-year before the overnight features existed.

**Morning-of model** (predicting ~5 hours ahead, 1,884 test days):
- Accuracy **59.8%**, AUC **0.632** — a big, clear edge.
- **Consistent every single year** (2019–2026), all between 56.8% and 63.0%:

  | Year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
  |---|---|---|---|---|---|---|---|---|
  | Accuracy | 56.8% | 59.7% | 60.7% | 63.0% | 57.8% | 59.9% | 59.9% | 61.9% |

- Validated by the permutation test (real 59.8% vs. shuffled ~50%) — the edge is real.

---

## 10. Things we tried that did NOT work (and why)

Honesty matters as much as results. These were tested and **rejected**:

- **Trader positioning data (CFTC):** only published weekly and with a delay — too stale to
  help a daily guess.
- **Producer-country currencies (Chilean peso, Australian dollar, Peruvian sol) and
  relative-value vs. stocks/gold/oil (SPY, GLD, USO):** these *sound* relevant, but they
  overlapped with features we already had (the dollar, the fear gauge, oil). Redundant clues
  add noise, and they slightly *hurt* accuracy, so we removed them.
- **Predicting "candle color" (did copper close above its own opening price?):** essentially
  a coin flip (~50%) for every model — that kind of move is unpredictable from prior-day data.

The lesson: a new clue only helps if it carries **genuinely new information**, not a repeat
of something the model already sees. That's exactly why the overnight Asian data worked — it
was the one thing not already baked in.

---

## 11. Limitations — what these models are NOT

- **Not a money machine.** A 54–60% edge on direction is real but small, and says nothing
  about *how much* copper moves. Trading costs alone could erase a thin edge.
- **The morning model's edge depends on the timing** — guess in the morning, or it doesn't
  hold.
- **Markets change.** A pattern that worked for years can fade as others discover it.
- **It only knows what it's shown.** Surprise news (a mine collapse, a policy shock) isn't in
  the data.
- **This is a learning/research project, not financial advice.**

---

## 12. How to run it

```bash
pip install -r requirements.txt          # needs: pandas, numpy, scikit-learn, matplotlib, joblib, yfinance

python copper_close_direction_model.py   # night-before model  -> outputs/copper_close_direction/
python copper_close_morning_model.py     # morning-of model    -> outputs/copper_close_morning/
python overfitting_check.py              # honesty checks (per-year, robustness, shuffle test)
```

Each model writes, into its `outputs/` folder:
- `evaluation_summary.md` — the human-readable results.
- `selected_features.csv` — the exact clues it kept, with their weights.
- `walk_forward_predictions.csv` — every test-day prediction vs. what actually happened.
- `metrics_summary.csv` / `.json` — the accuracy/AUC numbers.
- `model.pkl` — the trained model, saved for reuse.
- charts (confidence gate, etc.).

The data is found automatically; related-market prices are downloaded once and cached, so
re-runs work offline.

---

## 13. Mini-glossary

- **COMEX**: the New York exchange where copper futures trade.
- **Close / open / high / low**: a day's ending / starting / highest / lowest price.
- **Return**: percent change in price.
- **Momentum**: whether price is higher or lower than N days ago (trend strength).
- **Moving average**: the average price over the last N days (smooths out noise).
- **Volatility**: how jumpy/wild recent prices have been.
- **Divergence**: when a related market moves differently than copper — a possible early
  warning.
- **Z-score**: how unusual a value is vs. normal (2 = "much bigger than usual").
- **Feature**: one input clue (a single number) the model looks at.
- **Logistic regression**: a simple weighted-scorecard model that outputs a probability.
- **L1 / Lasso**: a method that zeroes out useless features, keeping only the helpful few.
- **Calibration**: making the model's "60%" actually mean a 60% chance.
- **Walk-forward testing**: only testing on future days the model never trained on.
- **Overfitting**: memorizing random noise instead of learning real patterns.
- **Permutation (shuffle) test**: scrambling the answers to confirm the edge is real, not
  luck.
- **AUC**: a 0.5–1.0 score for ranking ability; 0.5 = no skill, higher = better.
- **Leakage**: accidentally using future information — the cardinal sin we guard against.

---

*Built with scikit-learn only (no deep learning needed). Research and educational use only —
not financial advice.*
