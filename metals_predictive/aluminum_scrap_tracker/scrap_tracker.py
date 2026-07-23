"""Daily aluminum scrap-price tracker — fully market-data driven (no ticket data).

Each scrap grade is priced off a TRADED market benchmark:

    wrought/clean grades (anchor='p1020'):
        US delivered P1020 ($/lb) = LME aluminum ($/lb) + Midwest Premium ($/lb)
        scrap = realization x delivered P1020
    cast/secondary grades (anchor='alloy'):
        secondary anchor ($/lb)   = LME NASAAC / Aluminium Alloy ($/lb) + alloy premium
        scrap = realization x secondary anchor

LME aluminum is the daily DRIVER for wrought; the LME alloy (NASAAC) contract is the daily
driver for cast. realization factors are FIXED public-spread estimates (config.json) — this
build deliberately uses no internal ticket data.

Inputs (CSV, date-indexed):
  data/lme_aluminum.csv      [date, lme_al_close]   LME aluminum, USD/tonne (provided)
  data/midwest_premium.csv   [date, mwp_usd_per_lb] US Midwest Premium, USD/lb (optional)
  data/lme_alloy.csv         [date, alloy_close]    LME NASAAC/alloy, USD/tonne (you supply;
                             empty by default. Cast grades fall back to P1020 until present.)
  config.json                premiums + per-grade {anchor, realization}

Outputs:
  outputs/daily_scrap_prices.csv   full history: anchors + each grade ($/lb) + P1020 % moves
  outputs/latest_snapshot.md       latest-day board (+ anchor liquidity check)

Run:
  python scrap_tracker.py                 # build history + today's board + liquidity check
  python scrap_tracker.py --set-mwp 0.35  # what-if: Midwest Premium for all dates
  python scrap_tracker.py --latest        # print board only, no file writes
  python scrap_tracker.py --check         # print liquidity/staleness check for each anchor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
LME_CSV = BASE / "data" / "lme_aluminum.csv"
MWP_CSV = BASE / "data" / "midwest_premium.csv"
ALLOY_CSV = BASE / "data" / "lme_alloy.csv"
CONFIG = BASE / "config.json"
OUT_DIR = BASE / "outputs"

LB_PER_TONNE = 2204.62  # metric tonne -> lb
STALE_ZERO_CHG = 0.05   # >5% zero-change days => flag as possibly stale (ALI=F was 16%)


def log(m: str) -> None:
    print(f"[scrap_tracker] {m}", flush=True)


def load_config() -> dict:
    with open(CONFIG) as fh:
        cfg = json.load(fh)
    if not cfg.get("grades"):
        raise ValueError("config.json has no 'grades'.")
    return cfg


def _load_lme_series(path: Path, value_col: str) -> pd.DataFrame | None:
    """Load a [date, value_col] USD/tonne series -> DataFrame[date, <value_col>_per_lb]."""
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    if value_col not in df.columns or df[value_col].dropna().empty:
        return None
    df = (df[["date", value_col]].dropna()
          .sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True))
    df[f"{value_col}_per_lb"] = df[value_col] / LB_PER_TONNE
    return df


def liquidity_stats(s: pd.Series, name: str) -> dict:
    """Staleness check on a close-only series — the same test that caught COMEX ALI=F."""
    chg = s.pct_change()
    zero = float((chg == 0).mean())
    flat = (chg == 0).astype(int)
    streak = mx = 0
    for v in flat:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return {"name": name, "rows": int(len(s)), "zero_change_frac": zero,
            "max_flat_streak": int(mx), "stale": zero > STALE_ZERO_CHG}


def attach_mwp(df: pd.DataFrame, default_mwp: float, override: float | None) -> pd.DataFrame:
    df = df.copy()
    if override is not None:
        df["mwp_usd_per_lb"] = float(override)
        df["mwp_source"] = f"override({override})"
        return df
    if MWP_CSV.exists():
        mwp = pd.read_csv(MWP_CSV, parse_dates=["date"])
        mwp = mwp.rename(columns={c: c.lower().strip() for c in mwp.columns})
        if "mwp_usd_per_lb" not in mwp.columns:
            raise ValueError("midwest_premium.csv must have columns [date, mwp_usd_per_lb].")
        mwp = (mwp[["date", "mwp_usd_per_lb"]].dropna()
               .sort_values("date").drop_duplicates("date", keep="last"))
        df = df.merge(mwp, on="date", how="left")
        df["mwp_source"] = df["mwp_usd_per_lb"].map(lambda v: "csv" if pd.notna(v) else "default")
        df["mwp_usd_per_lb"] = df["mwp_usd_per_lb"].ffill().fillna(default_mwp)
    else:
        df["mwp_usd_per_lb"] = default_mwp
        df["mwp_source"] = "default"
    return df


def build(cfg: dict, override_mwp: float | None):
    lme = _load_lme_series(LME_CSV, "lme_al_close")
    if lme is None:
        raise FileNotFoundError(f"missing/empty {LME_CSV} (LME aluminum, USD/tonne).")
    df = attach_mwp(lme, cfg["midwest_premium_default_usd_per_lb"], override_mwp)
    df["delivered_p1020_per_lb"] = df["lme_al_close_per_lb"] + df["mwp_usd_per_lb"]

    # Cast-side anchor: LME NASAAC / Aluminium Alloy (you supply). Align to LME dates, ffill.
    alloy = _load_lme_series(ALLOY_CSV, "alloy_close")
    alloy_present = alloy is not None
    if alloy_present:
        df = df.merge(alloy[["date", "alloy_close_per_lb"]], on="date", how="left")
        df["alloy_close_per_lb"] = df["alloy_close_per_lb"].ffill()
        df["secondary_anchor_per_lb"] = df["alloy_close_per_lb"] + cfg.get("alloy_premium_usd_per_lb", 0.0)

    notes = {"alloy_present": alloy_present, "alloy_fallback_grades": []}
    for grade, spec in cfg["grades"].items():
        anchor = spec.get("anchor", "p1020")
        if anchor == "alloy" and alloy_present:
            base = df["secondary_anchor_per_lb"]
            used = "alloy"
        else:
            if anchor == "alloy":
                notes["alloy_fallback_grades"].append(grade)  # no alloy data yet
            base = df["delivered_p1020_per_lb"]
            used = "p1020(fallback)" if anchor == "alloy" else "p1020"
        df[f"scrap_{grade}_per_lb"] = spec["realization"] * base
        df[f"scrap_{grade}_anchor"] = used

    df["p1020_chg_1d"] = df["delivered_p1020_per_lb"].pct_change(1)
    df["p1020_chg_5d"] = df["delivered_p1020_per_lb"].pct_change(5)

    liq = [liquidity_stats(df["lme_al_close_per_lb"], "LME aluminum")]
    if alloy_present:
        liq.append(liquidity_stats(df["alloy_close_per_lb"].dropna(), "LME alloy/NASAAC"))
    return df, notes, liq


def liquidity_report(liq: list[dict]) -> str:
    lines = ["## Anchor liquidity check (staleness guard — caught COMEX ALI=F)"]
    for s in liq:
        flag = "  ⚠️ POSSIBLY STALE — treat with caution" if s["stale"] else "  ✓ clean"
        lines.append(f"- {s['name']}: {s['rows']} rows, "
                     f"{s['zero_change_frac']:.1%} zero-change days, "
                     f"max flat streak {s['max_flat_streak']}d{flag}")
    return "\n".join(lines)


def latest_board(df, cfg, notes, liq) -> str:
    r = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else r
    d1 = r["delivered_p1020_per_lb"] - prev["delivered_p1020_per_lb"]
    L = []
    L.append(f"# Aluminum scrap board — {r['date'].date()}")
    L.append("")
    L.append(f"- LME aluminum: **${r['lme_al_close']:,.1f}/t**  (${r['lme_al_close_per_lb']:.4f}/lb)")
    L.append(f"- Midwest Premium: **${r['mwp_usd_per_lb']:.4f}/lb**  (source: {r['mwp_source']})")
    L.append(f"- **US delivered P1020: ${r['delivered_p1020_per_lb']:.4f}/lb**  "
             f"(1d {d1:+.4f}, {r['p1020_chg_1d']:+.2%} · 5d {r['p1020_chg_5d']:+.2%})")
    if notes["alloy_present"]:
        L.append(f"- LME alloy/NASAAC (cast anchor): **${r['secondary_anchor_per_lb']:.4f}/lb**")
    else:
        L.append("- LME alloy/NASAAC (cast anchor): **not supplied** — cast grades using "
                 "P1020 fallback (drop data/lme_alloy.csv to activate)")
    L.append("")
    L.append("| Grade | anchor | realization | est. scrap $/lb |")
    L.append("|---|---|---|---|")
    for grade, spec in cfg["grades"].items():
        L.append(f"| {grade} | {r[f'scrap_{grade}_anchor']} | {spec['realization']:.2f} | "
                 f"**${r[f'scrap_{grade}_per_lb']:.4f}** |")
    L.append("")
    L.append(liquidity_report(liq))
    L.append("")
    L.append("> Market data only. scrap = realization × anchor. realization factors are fixed "
             "public-spread estimates (config.json) — no ticket data used.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-mwp", type=float, default=None, help="override Midwest Premium ($/lb) all dates")
    ap.add_argument("--latest", action="store_true", help="print board only, no file writes")
    ap.add_argument("--check", action="store_true", help="print anchor liquidity check and exit")
    args = ap.parse_args()

    cfg = load_config()
    df, notes, liq = build(cfg, args.set_mwp)

    if args.check:
        print(liquidity_report(liq))
        return 0

    board = latest_board(df, cfg, notes, liq)
    if args.latest:
        print(board)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "daily_scrap_prices.csv"
    df.to_csv(out_csv, index=False)
    (OUT_DIR / "latest_snapshot.md").write_text(board + "\n", encoding="utf-8")
    log(f"wrote {out_csv}  ({len(df)} rows, {df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()})")
    if notes["alloy_fallback_grades"]:
        log(f"NOTE: no alloy data yet — these cast grades used P1020 fallback: "
            f"{', '.join(notes['alloy_fallback_grades'])}")
    print()
    print(board)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
