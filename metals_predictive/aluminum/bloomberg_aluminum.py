"""Fetch aluminum-specific Bloomberg data.

These are the key aluminum drivers absent from the free yfinance model:
  - SHFE aluminum (China's domestic futures, closes ~3pm Beijing / 2am ET)
  - LME aluminum inventory (on-warrant stocks) and cancelled warrants
  - European TTF natural gas (aluminum smelting cost proxy)

⚠️  VERIFY ALL TICKERS on your Terminal before trusting output.
    For each entry in TICKERS, type the ticker in Bloomberg and run DES <GO> to
    confirm it is the right series. The values below are best-guess generics that
    follow Bloomberg naming conventions but may require adjustment.

Requires Bloomberg Terminal running and logged in (port 8194).
Install one of:
    pip install xbbg
    pip install pdblp

Run:  python bloomberg_aluminum.py  (from inside aluminum_direction_model_v1/)

Outputs are written to data/external/ and picked up automatically by
lme_aluminum_morning_model.py and lme_aluminum_direction_model.py on next run.
"""
from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
START = "20140601"
END = pd.Timestamp.today().strftime("%Y%m%d")

# (bloomberg_ticker, output_column_name, relative_output_path, human_description)
#
# TICKERS TO VERIFY ON YOUR TERMINAL:
#
#   ALUA Comdty    — SHFE aluminum 1st generic front month (CNY/tonne)
#                    Verify: type  ALUA Comdty <GO>  then  DES <GO>
#                    Alternative if ALUA fails: try  ALU1 Comdty  or  ALUA1 Comdty
#
#   LMAHSTKS Index — LME aluminum total on-warrant warehouse stocks (tonnes)
#                    Verify: type  LMAHSTKS Index <GO>  then  DES <GO>
#                    Alternative: search  LME ALUMINIUM STOCKS  in Bloomberg search
#
#   LMAHCWRTS Index — LME aluminum cancelled warrants (% of total stocks)
#                     Verify: type  LMAHCWRTS Index <GO>  then  DES <GO>
#                     Alternative: search  LME ALUMINIUM CANCELLED WARRANTS
#
#   TTFMMA0 Comdty — European TTF natural gas front-month generic (EUR/MWh)
#                    Verify: type  TTFMMA0 Comdty <GO>  then  DES <GO>
#                    Alternative: try  TTFGAS NL  or  GASNLH Comdty
#                    Note: TTF is EUR/MWh; the model uses % returns so units don't matter.
#
TICKERS = [
    ("AN01 Comdty",     "shfe_al_close",  "data/external/shfe_aluminum.csv",
     "SHFE aluminum 1st generic (CNY/tonne)"),
    ("MEPRAICW Index",  "lme_al_stocks",  "data/external/lme_al_inventory.csv",
     "LME aluminum on-warrant stocks (tonnes)"),
    ("MEPRALCW Index",  "lme_al_cw",      "data/external/lme_al_cancelled_warrants.csv",
     "LME aluminum cancelled warrants (tonnes)"),
    ("TTFGDAHD Index",  "ttf_close",      "data/external/ttf_gas.csv",
     "TTF gas day-ahead (EUR/MWh)"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Bloomberg fetch (identical pattern to bloombergCopper.py / bloombergSteel.py)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(df) -> pd.DataFrame:
    if hasattr(df, "to_native"):
        df = df.to_native()
    elif hasattr(df, "to_pandas"):
        df = df.to_pandas()
    else:
        df = pd.DataFrame(df)
    if df is None or len(df) == 0:
        raise RuntimeError("Bloomberg returned empty data")
    if {"date", "field", "value"}.issubset(df.columns):
        return df[["date", "value"]].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1).reset_index()
    else:
        df = df.reset_index()
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    return df


def fetch_xbbg(ticker: str) -> pd.DataFrame:
    from xbbg import blp
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


def fetch(ticker: str) -> pd.DataFrame:
    for name, fn in [("xbbg", fetch_xbbg), ("pdblp", fetch_pdblp)]:
        try:
            return fn(ticker)
        except ImportError:
            pass
        except Exception as e:
            print(f"  {name} failed for {ticker}: {e}")
    raise RuntimeError(f"All methods failed for {ticker}. Is Bloomberg Terminal running?")


def main():
    any_failed = False
    for ticker, col_name, out_path, desc in TICKERS:
        out = BASE_DIR / out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {ticker}  ({desc})")
        print(f"  -> {out_path}")
        try:
            raw = fetch(ticker)
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw.dropna().sort_values("date").drop_duplicates("date")
            raw.columns = ["date", col_name]
            raw.to_csv(out, index=False)
            print(f"  Saved {len(raw)} rows "
                  f"({raw['date'].min().date()} - {raw['date'].max().date()})")
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  *** Verify the ticker '{ticker}' on your Terminal: DES <GO> ***")
            any_failed = True
        print()

    if any_failed:
        print("Some tickers failed — see *** lines above.")
        print("Fix the ticker symbol in TICKERS and re-run. Successful files are already saved.")
        sys.exit(1)
    print("All done. Re-run lme_aluminum_morning_model.py to use the new features.")


if __name__ == "__main__":
    main()
