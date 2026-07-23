"""Pre-fetch all yfinance series into the exact caches the steel models expect.
Writes data/raw/steel.csv, macro_cache/macro_prices_expanded.csv, overnight_prices.csv.
Run once; the models then run offline from cache."""
import time, sys, pathlib
import pandas as pd, numpy as np
import yfinance as yf
import warnings; warnings.filterwarnings("ignore")

import steel_close_direction_model as base
import steel_close_morning_model as morning

BASE = pathlib.Path(__file__).resolve().parent

def safe(tk): return base._safe_name(tk)

def dl(tk):
    for a in range(4):
        try:
            d = yf.download(tk, start="2014-06-01", auto_adjust=True, progress=False, threads=False)
            if d is not None and len(d):
                return d
        except Exception as e:
            print(f"  {tk} attempt {a+1}: {type(e).__name__}")
        time.sleep(3)
    print(f"  {tk}: FAILED")
    return None

def close_series(d, tk):
    c = d["Close"]
    s = c.iloc[:,0] if isinstance(c, pd.DataFrame) else c
    return s.rename(safe(tk))

# --- 1. Target: HRC=F OHLCV -> data/raw/steel.csv ---
print("Fetching HRC=F (target)...")
hrc = dl("HRC=F")
oh = hrc[["Open","High","Low","Close","Volume"]].copy()
oh.columns = ["Open","High","Low","Close","Volume"]
oh = oh.reset_index().rename(columns={"Date":"Date","index":"Date"})
oh["Date"] = pd.to_datetime(oh["Date"]).dt.tz_localize(None).dt.date
oh = oh.dropna(subset=["Close"])
(BASE/"data"/"raw").mkdir(parents=True, exist_ok=True)
oh.to_csv(BASE/"data"/"raw"/"steel.csv", index=False)
print(f"  steel.csv: {len(oh)} rows {oh['Date'].min()} -> {oh['Date'].max()}")

# --- 2. Macro cache (divergence + plain tickers) ---
print(f"Fetching {len(base.ALL_TICKERS)} macro tickers...")
frames = {}
for tk in base.ALL_TICKERS:
    d = dl(tk)
    if d is not None:
        frames[safe(tk)] = close_series(d, tk)
wide = pd.concat(frames.values(), axis=1, sort=True).reset_index().rename(columns={"Date":"date","index":"date"})
wide["date"] = pd.to_datetime(wide["date"]).dt.tz_localize(None)
(BASE/"data"/"external"/"macro_cache").mkdir(parents=True, exist_ok=True)
wide.to_csv(base.MACRO_CACHE, index=False)
print(f"  macro cache: {wide.shape[0]} rows, {wide.shape[1]-1} tickers: {sorted(frames)}")

# --- 3. Overnight cache (steelmakers) ---
print(f"Fetching {len(morning.OVERNIGHT_TICKERS)} overnight tickers...")
ofr = {}
for tk in morning.OVERNIGHT_TICKERS:
    d = dl(tk)
    if d is not None:
        ofr[safe(tk)] = close_series(d, tk)
ow = pd.concat(ofr.values(), axis=1, sort=True).reset_index().rename(columns={"Date":"date","index":"date"})
ow["date"] = pd.to_datetime(ow["date"]).dt.tz_localize(None)
ow.to_csv(morning.OVERNIGHT_CACHE, index=False)
print(f"  overnight cache: {ow.shape[0]} rows, {ow.shape[1]-1} tickers: {sorted(ofr)}")
print("PREFETCH DONE")
