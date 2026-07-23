# Aluminum scrap board — 2026-06-26

- LME aluminum: **$3,179.5/t**  ($1.4422/lb)
- Midwest Premium: **$0.2000/lb**  (source: csv)
- **US delivered P1020: $1.6422/lb**  (1d +0.0068, +0.42% · 5d -5.65%)
- LME alloy/NASAAC (cast anchor): **not supplied** — cast grades using P1020 fallback (drop data/lme_alloy.csv to activate)

| Grade | anchor | realization | est. scrap $/lb |
|---|---|---|---|
| 6061_6063_extrusion_clean | p1020 | 0.90 | **$1.4780** |
| clean_sheet_litho | p1020 | 0.85 | **$1.3959** |
| ubc | p1020 | 0.80 | **$1.3138** |
| turnings_borings | p1020 | 0.52 | **$0.8539** |
| zorba | p1020 | 0.58 | **$0.9525** |
| mixed_low_copper | p1020(fallback) | 0.82 | **$1.3466** |
| old_cast | p1020(fallback) | 0.88 | **$1.4451** |

## Anchor liquidity check (staleness guard — caught COMEX ALI=F)
- LME aluminum: 3052 rows, 1.4% zero-change days, max flat streak 2d  ✓ clean

> Market data only. scrap = realization × anchor. realization factors are fixed public-spread estimates (config.json) — no ticket data used.
