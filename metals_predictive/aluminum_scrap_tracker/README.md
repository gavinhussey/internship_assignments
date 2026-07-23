# Aluminum scrap daily tracker — market data only

Prices each scrap grade off a **traded market benchmark** — no internal ticket data.

```
wrought/clean (anchor = p1020):
    US delivered P1020 ($/lb) = LME aluminum ($/lb) + Midwest Premium ($/lb)
    scrap = realization × delivered P1020
cast/secondary (anchor = alloy):
    secondary anchor ($/lb)   = LME NASAAC / Aluminium Alloy ($/lb) + alloy premium
    scrap = realization × secondary anchor
```

- **LME aluminum** drives the wrought grades; **LME NASAAC/alloy** drives the cast grades.
- The **per-grade realization factor** is the only thing that differs by grade. It is a
  **fixed public-spread estimate** in `config.json` — updated from published market spreads,
  not from your tickets (by design).

## Run

```bash
python scrap_tracker.py                 # build history + today's board + liquidity check
python scrap_tracker.py --latest        # print board only (no file writes)
python scrap_tracker.py --set-mwp 0.40  # what-if: Midwest Premium at $0.40/lb for all dates
python scrap_tracker.py --check         # anchor liquidity/staleness check only
```

Outputs: `outputs/daily_scrap_prices.csv` (full history) and `outputs/latest_snapshot.md`.

## Inputs you maintain (all market data)

| File | Columns | Notes |
|---|---|---|
| `data/lme_aluminum.csv` | `date, lme_al_close` | LME aluminum **USD/tonne**. Provided (2014→2026). |
| `data/midwest_premium.csv` | `date, mwp_usd_per_lb` | US Midwest Premium **USD/lb**. Last value carried forward; config default before first row. |
| `data/lme_alloy.csv` | `date, alloy_close` | **LME NASAAC / Aluminium Alloy, USD/tonne — you supply this.** Empty by default; until present, cast grades fall back to P1020 (clearly labeled `p1020(fallback)`). |
| `config.json` | — | premiums + per-grade `{anchor, realization}`. |

There is **no free daily API** for LME aluminum or NASAAC. Refresh them from your data
provider — a **Bloomberg fetcher is included** (see below), or use NASDAQ Data Link /
Westmetall / a manual CSV drop.

## Pulling the alloy data from Bloomberg

`bloomberg_fetch.py` pulls LME aluminium alloy (and primary) history straight into the
tracker's CSVs. It needs a live Bloomberg session (logged-in Terminal or B-PIPE) and the
Bloomberg Python API; it auto-detects `xbbg` or raw `blpapi`.

```bash
pip install -r requirements-bloomberg.txt    # + blpapi from Bloomberg's index (see that file)

python bloomberg_fetch.py --check            # connect + print last 5 alloy rows (verify ticker)
python bloomberg_fetch.py                     # write data/lme_alloy.csv  (cast anchor)
python bloomberg_fetch.py --security all      # also refresh data/lme_aluminum.csv (primary)
python scrap_tracker.py --check               # staleness guard on the freshly pulled data
```

**Confirm the security strings first.** The tickers in `bloomberg_config.json`
(`LMAADS03 Comdty` for alloy, `LMAHDS03 Comdty` for primary) are common forms but **verify
the exact security in your Terminal** — for US cast scrap you likely want the **NASAAC**
(North American Special Aluminium Alloy) contract; set its security in the config or pass
`--ticker "<SEC> Comdty"`. Field defaults to `PX_LAST` (override with `--field PX_SETTLE`).
LME quotes are USD/tonne, matching the tracker.

## ⚠️ NASAAC is thin — the tracker checks for staleness automatically

The LME alloy/NASAAC contract trades far less than primary aluminum, so it **may be stale**
(repeated/echoed settlements) the way COMEX ALI=F was. Every run prints a liquidity check:

```
- LME aluminum:    1.4% zero-change days, max flat streak 2d   ✓ clean
- LME alloy/NASAAC: …% zero-change days, …                      ⚠️ POSSIBLY STALE …
```

If NASAAC flags as stale (>5% zero-change days), don't trust the cast estimates as a daily
signal — treat them as a slow level only, and consider anchoring cast grades back to P1020
(`"anchor": "p1020"` in config.json) with a lower realization.

## Realization factors — fixed, public-spread based

Values in `config.json` are mid-points of typical published scrap-vs-anchor spreads. Note
that an `alloy`-anchored grade's realization is measured **against the alloy anchor** (which
is already a secondary price), so it's higher than the same grade's ratio vs primary P1020.
Update these from public market spreads when they move; they intentionally use no ticket data.

## Scope & honest caveats

- **Estimates a fair-value level; it does not forecast.** Daily *direction* of each estimate
  is just its anchor's direction.
- **Fully-free limitation:** there is no free daily public *scrap* price, so the scrap
  discount (realization) is a static market-spread assumption, not a live daily scrap feed.
  A live per-grade scrap feed would require a PRA subscription (Davis Index / Fastmarkets /
  Platts) — out of scope for this market-data-only build.
- **Do not** use COMEX aluminum (ALI=F) as an anchor — it's a stale LME echo (85% no-trade
  days). LME aluminum (and NASAAC for cast, if it passes the staleness check) are correct.
