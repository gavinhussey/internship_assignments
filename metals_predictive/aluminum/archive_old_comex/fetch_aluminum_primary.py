"""Download the aluminum PRIMARY price series -> data/raw/aluminum.csv.

Primary = COMEX aluminum futures (CME ticker ALI), pulled from yfinance as 'ALI=F'.
This is the USD/tonne US-session contract, so the morning model's "LME settles before the
US session" overnight-gap logic applies exactly as it does for copper.

Writes Date,Open,High,Low,Close,Volume — the format the model auto-discovers.

Run:  python fetch_aluminum_primary.py
      python fetch_aluminum_primary.py --ticker ALI=F --start 2014-06-01
      python fetch_aluminum_primary.py --fallback   # also try ETF proxies if ALI=F is thin

Notes
-----
* COMEX aluminum (ALI=F) is far less liquid than COMEX copper; yfinance history can be
  short or gappy. The script reports coverage so you can judge if it's enough (the model
  needs ~1000 rows before walk-forward begins). If ALI=F is too thin, pass --fallback to
  also try liquid US-listed proxies and pick the one with the longest daily history.
* This is intentionally the ONLY network step for the primary; cross-asset cousins and
  overnight producers are fetched/cached by the models themselves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "data" / "raw" / "aluminum.csv"

# US-session aluminum proxies, in preference order, used only with --fallback.
# JJU = iPath aluminum ETN (direct but illiquid); the rest are broad-metal proxies.
FALLBACK_TICKERS = ["JJU", "DBB", "PICK"]


def log(m: str) -> None:
    print(f"[fetch_primary] {m}", flush=True)


def _download(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if d is None or len(d) == 0:
        return pd.DataFrame()
    # yfinance can return a MultiIndex column frame for a single ticker — flatten it.
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns={"Adj Close": "AdjClose"}).reset_index()
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    out = d[keep].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None).dt.date
    out = out.dropna(subset=["Close"]).drop_duplicates("Date", keep="last").sort_values("Date")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ALI=F", help="primary yfinance ticker (default COMEX aluminum)")
    ap.add_argument("--start", default="2014-06-01")
    ap.add_argument("--fallback", action="store_true",
                    help="if the primary ticker is thin, also try ETF proxies and keep the longest")
    args = ap.parse_args()

    candidates = [args.ticker] + (FALLBACK_TICKERS if args.fallback else [])
    best, best_df = None, pd.DataFrame()
    for tk in candidates:
        try:
            df = _download(tk, args.start)
        except Exception as e:
            log(f"{tk}: download failed ({type(e).__name__}: {e})")
            continue
        log(f"{tk}: {len(df)} rows"
            + (f"  {df['Date'].iloc[0]}..{df['Date'].iloc[-1]}" if len(df) else "")
            + ("  [has OHLC]" if {"Open", "High", "Low"}.issubset(df.columns) else "  [close only]"))
        if len(df) > len(best_df):
            best, best_df = tk, df

    if best_df.empty:
        log("FAILED: no data returned for any candidate. Check connectivity / ticker, or "
            "supply data/raw/aluminum.csv manually (see data/raw/PUT_ALUMINUM_PRICE_FILE_HERE.md).")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(OUT, index=False)
    log(f"WROTE {OUT}  (source ticker: {best})  rows={len(best_df)}")
    if len(best_df) < 1000:
        log(f"WARNING: only {len(best_df)} rows — the walk-forward needs ~1000 before it starts "
            f"scoring. Consider --fallback or a longer primary series.")
    if not {"Open", "High", "Low"}.issubset(best_df.columns):
        log("WARNING: no OHLC — candle features and the morning model's US-open LME-gap proxy "
            "will be skipped. A full OHLCV primary is strongly preferred.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
