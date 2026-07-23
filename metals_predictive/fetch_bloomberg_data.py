"""Single-entry Bloomberg fetch script — run this on the Bloomberg PC."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────────
# STEEL — China ferrous complex
# Outputs -> steel/data/external/
# ⚠️  Verify each ticker: open Bloomberg, type ticker <GO>, then DES <GO>
# ──────────────────────────────────────────────────────────────────────────────
STEEL_OUT = BASE / "steel" / "data" / "external"

STEEL_TICKERS = [
    # ticker               column           filename                  description
    # ✓ GOOD (2,931 rows 2014-2026) — skip unless re-fetching everything
    ("IOE1 Comdty",  "dce_io_close",   "dce_iron_ore.csv",     "DCE iron ore 1st generic (CNY/t)"),
    # ✓ GOOD (3,115 rows 2014-2026) — skip unless re-fetching everything
    ("SCO1 Comdty",  "sgx_io_close",   "sgx_iron_ore.csv",     "SGX iron ore 62% Fe (USD/t)"),
    # ⚠️  RE-FETCH NEEDED: RBTA/RBA were wrong. Try RBT1 Comdty — verify DES <GO> first.
    ("RBT1 Comdty",  "shfe_rb_close",  "shfe_rebar.csv",       "SHFE rebar 1st generic (CNY/t)"),
    # ⚠️  RE-FETCH NEEDED: HCA/HC1 were wrong. Try ROC1 Comdty (same pattern as RBT1).
    # NOTE: Do NOT use CMX HRC (US Midwest) — we need SHFE HRC (China). Verify DES <GO>.
    ("ROC1 Comdty",  "shfe_hrc_close", "shfe_hrc.csv",         "SHFE hot-rolled coil 1st generic (CNY/t)"),
    # ❌ BROKEN: JMA/JM01 failed. Try CKC1 Comdty (DCE coking coal 1st generic).
    ("CKC1 Comdty",   "dce_cc_close",   "dce_coking_coal.csv",  "DCE coking coal 1st generic (CNY/t)"),
]

# To re-fetch ONLY the broken tickers (faster), set STEEL_TICKERS_REFIX to True and run.
# The good DCE/SGX iron ore data will be skipped automatically by --skip-good flag below.
STEEL_TICKERS_BROKEN_ONLY = [
    # Verify each with DES <GO> before running
    ("RBT1 Comdty",  "shfe_rb_close",  "shfe_rebar.csv",       "SHFE rebar 1st generic (CNY/t)"),
    ("ROC1 Comdty",  "shfe_hrc_close", "shfe_hrc.csv",         "SHFE hot-rolled coil 1st generic (CNY/t)"),
    ("CKC1 Comdty",  "dce_cc_close",   "dce_coking_coal.csv",  "DCE coking coal 1st generic (CNY/t)"),
]

# ──────────────────────────────────────────────────────────────────────────────
# ALUMINUM — SHFE aluminum, LME inventory, European gas
# Outputs -> aluminum/data/external/
# ⚠️  Verify each ticker before running
# ──────────────────────────────────────────────────────────────────────────────
ALUMINUM_OUT = BASE / "aluminum" / "data" / "external"

ALUMINUM_TICKERS = [
    # ticker               column            filename                       description
    ("ANO1 Comdty",   "shfe_al_close",  "shfe_aluminum.csv",           "SHFE aluminum 1st generic (CNY/t)"),
    ("MEPRAICW Index","lme_al_stocks",  "lme_al_inventory.csv",        "LME aluminum on-warrant stocks (t)"),
    ("MEPRALCW Index","lme_al_cw",     "lme_al_cancelled_warrants.csv","LME aluminum cancelled warrants (t)"),
    ("TTFGDAHD Index","ttf_close",     "ttf_gas.csv",                  "TTF gas day-ahead (EUR/MWh)"),
]

# ──────────────────────────────────────────────────────────────────────────────
# TICKER NOTES — common alternatives if a ticker fails DES <GO> check
# ──────────────────────────────────────────────────────────────────────────────
TICKER_NOTES = """
Ticker verification guide (type each in Bloomberg, then DES <GO>):

  RBTA Comdty   → SHFE rebar. Alternatives: RBA Comdty, RBT1 Comdty
  HCA Comdty    → SHFE hot-rolled coil. Alternatives: HRC1 Comdty, SHFRHRCA Index
  IOE1 Comdty   → DCE iron ore. Alternatives: I01 Comdty, IOA Comdty
  JMA Comdty    → DCE coking coal. Alternatives: JM01 Comdty, CKC1 Comdty
  SCO1 Comdty   → SGX iron ore. Alternatives: TIO1 Comdty

  ALA Comdty    → SHFE aluminum 1st generic. Alternatives: ALUA Comdty, ALU1 Comdty
  LMAHSTKS Index → LME aluminum on-warrant stocks (fetched via VALUE field, not PX_LAST)
  LMAHCWRTS Index → LME aluminum cancelled warrants (fetched via VALUE field, not PX_LAST)
  TTFHNGDY Index → TTF Hub day-ahead gas. Alternatives: TTFGASNLD Index, NGERNLH Index
"""

START = "20140601"
END = pd.Timestamp.today().strftime("%Y%m%d")


def _normalise(df) -> pd.DataFrame:
    """Coerce whatever xbbg/pdblp returns to a clean two-column [date, value] DataFrame."""
    if hasattr(df, "to_native"):
        df = df.to_native()
    elif hasattr(df, "to_pandas"):
        df = df.to_pandas()
    else:
        df = pd.DataFrame(df)
    if df is None or len(df) == 0:
        raise RuntimeError("Bloomberg returned empty data")
    # Long format: columns include 'date', 'field', 'value'
    if {"date", "field", "value"}.issubset(df.columns):
        return df[["date", "value"]].copy()
    # Wide format with MultiIndex (ticker, field) columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1).reset_index()
    else:
        df = df.reset_index()
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    return df


def fetch_xbbg(ticker: str) -> pd.DataFrame:
    from xbbg import blp
    # Try PX_LAST first (prices), fall back to VALUE (LME inventory / index series)
    for field in ("PX_LAST", "VALUE"):
        try:
            result = blp.bdh(tickers=ticker, flds=field, start_date="2014-06-01")
            df = _normalise(result)
            if len(df.dropna()) > 0:
                return df
        except Exception:
            pass
    raise RuntimeError(f"xbbg: no data for {ticker} under PX_LAST or VALUE")


def fetch_pdblp(ticker: str) -> pd.DataFrame:
    import pdblp
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    try:
        for field in ("PX_LAST", "VALUE"):
            try:
                df = con.bdh(tickers=[ticker], flds=[field],
                             start_date=START, end_date=END)
                df = df.droplevel(0, axis=1).reset_index()
                df.columns = ["date", "value"]
                if len(df.dropna()) > 0:
                    return df
            except Exception:
                pass
        raise RuntimeError(f"pdblp: no data for {ticker} under PX_LAST or VALUE")
    finally:
        con.stop()


def fetch_one(ticker: str) -> pd.DataFrame:
    for name, fn in [("xbbg", fetch_xbbg), ("pdblp", fetch_pdblp)]:
        try:
            return fn(ticker)
        except ImportError:
            pass
        except Exception as e:
            print(f"      {name}: {e}")
    raise RuntimeError(f"All Bloomberg methods failed for {ticker}")


def run_group(label: str, tickers: list, out_dir: Path, dry_run: bool) -> tuple[int, int]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for ticker, col, filename, desc in tickers:
        out = out_dir / filename
        print(f"\n  {ticker}  —  {desc}")
        if dry_run:
            print(f"    [dry-run] would write -> {out.relative_to(BASE)}")
            continue
        print(f"    fetching...", end="", flush=True)
        try:
            raw = fetch_one(ticker)
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw.dropna().sort_values("date").drop_duplicates("date")
            raw.columns = ["date", col]
            raw.to_csv(out, index=False)
            print(f"  {len(raw)} rows  "
                  f"({raw['date'].min().date()} → {raw['date'].max().date()})")
            print(f"    -> {out.relative_to(BASE)}")
            ok += 1
        except Exception as e:
            print(f"\n    FAILED: {e}")
            print(f"    *** Verify '{ticker}' on the Terminal: DES <GO> ***")
            fail += 1
    return ok, fail


def main():
    parser = argparse.ArgumentParser(description="Fetch Bloomberg data for metal direction models")
    parser.add_argument("--skip-steel",       action="store_true", help="Skip all steel/iron ore tickers")
    parser.add_argument("--skip-aluminum",    action="store_true", help="Skip all aluminum tickers")
    parser.add_argument("--steel-broken-only", action="store_true",
                        help="Re-fetch only the 3 broken steel tickers (RBA/HC1/JM01). "
                             "Skips the good DCE/SGX iron ore data.")
    parser.add_argument("--dry-run",          action="store_true", help="List tickers without fetching")
    args = parser.parse_args()

    print(TICKER_NOTES)

    if args.dry_run:
        print("DRY RUN — no data will be fetched\n")

    total_ok, total_fail = 0, 0

    if not args.skip_steel:
        steel_list = STEEL_TICKERS_BROKEN_ONLY if args.steel_broken_only else STEEL_TICKERS
        label = "STEEL — broken tickers only (RBA/HC1/JM01)" if args.steel_broken_only else "STEEL — China ferrous complex"
        ok, fail = run_group(label, steel_list, STEEL_OUT, args.dry_run)
        total_ok += ok; total_fail += fail

    if not args.skip_aluminum:
        ok, fail = run_group("ALUMINUM — fundamentals", ALUMINUM_TICKERS, ALUMINUM_OUT, args.dry_run)
        total_ok += ok; total_fail += fail

    print(f"\n{'='*60}")
    if args.dry_run:
        print("  Dry run complete — run without --dry-run to fetch data")
    else:
        print(f"  Done: {total_ok} succeeded, {total_fail} failed")
        if total_fail:
            print("  Fix the failed tickers (see *** lines above) and re-run.")
            print("  Successful CSVs are already saved — use --skip-* flags to skip them.")
        if total_ok:
            print("""
  Next steps on this PC:
    git add direction_models/steel/data/external/
    git add direction_models/aluminum/data/external/
    git commit -m "Add Bloomberg data: China ferrous + aluminum fundamentals"
    git push

  Then on your Mac:
    git pull
    cd steel  &&  python iron_ore_direction_model.py
    cd steel  &&  python iron_ore_morning_model.py
    cd aluminum  &&  python lme_aluminum_morning_model.py
""")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
