# Pull BTCUSDT daily OHLCV from Binance's official public data archive.
#
# WHY THIS EXISTS: the original data/Binance_BTCUSDT_d.csv came from CryptoDataDownload
# and was MISSING 24 DAYS, all of them in 2026 - including consecutive runs
# (2026-03-20..22 and 2026-05-21..28). Gaps are silent poison for this project: every
# rolling window and every .diff() that straddles a hole is quietly wrong, and the holes
# sat exactly where the reserved out-of-sample test window has to live.
#
# data.binance.vision is the SOURCE CryptoDataDownload resells. It is free, needs no API
# key, and is not geo-blocked (api.binance.com returns HTTP 451 from a US IP; this does
# not). Validated against the CryptoDataDownload file: identical Close to 0.000000% on all
# 3,229 overlapping days, plus the 24 days CDD dropped. Strict superset, same numbers.
#
# Re-running this is SAFE and idempotent - unlike fetch_onchain.py in models/, this archive
# does not roll forward, so a re-pull can only ever return MORE data, never less.
#
# Usage:  cd magnitude_of_move && python fetch_ohlcv.py

import datetime as dt
import io
import zipfile

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/spot"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START = dt.date(2017, 8, 1)          # first month Binance listed BTCUSDT
OUT = "data/Binance_BTCUSDT_d.csv"

# Binance kline schema. Older dumps are headerless; dumps from 2025 on ship a header row.
KLINE_COLS = [
    "open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

KEEP = ["Date", "Open", "High", "Low", "Close", "Volume", "quote_volume", "trades"]


def read_archive(url):
    """Download one zip and return its klines as a DataFrame, or None if absent (404)."""
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode()
    has_header = raw.split("\n", 1)[0].lower().startswith("open_time")
    df = pd.read_csv(io.StringIO(raw), header=0 if has_header else None)
    df.columns = KLINE_COLS[: len(df.columns)]
    return df


def month_iter(start, end):
    m = start.replace(day=1)
    while m <= end.replace(day=1):
        yield m
        m = (m.replace(day=28) + dt.timedelta(days=8)).replace(day=1)


def fetch(today=None):
    today = today or dt.date.today()
    frames = []

    # Completed months come as one zip each.
    for m in month_iter(START, today - dt.timedelta(days=today.day)):
        df = read_archive(f"{BASE}/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{m:%Y-%m}.zip")
        if df is not None:
            frames.append(df)

    # The current, still-running month has no monthly zip yet - fall back to daily zips.
    # The most recent day or two will 404 until Binance publishes them; that is expected.
    d = today.replace(day=1)
    while d <= today:
        df = read_archive(f"{BASE}/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{d:%Y-%m-%d}.zip")
        if df is not None:
            frames.append(df)
        d += dt.timedelta(days=1)

    if not frames:
        raise RuntimeError("Fetched nothing. Check network / that data.binance.vision is reachable.")

    raw = pd.concat(frames, ignore_index=True)

    # 🔴 open_time is epoch MILLIseconds in older dumps and epoch MICROseconds in newer
    # ones. The unit changes PARTWAY THROUGH the archive, so it must be decided per row.
    # Deciding it once for the whole frame reads every pre-switch row as 1970 - silently,
    # with no error, destroying ~80% of the history.
    ot = raw["open_time"].astype("int64")
    epoch_ms = ot.where(ot <= 1e15, ot // 1000)
    raw["Date"] = pd.to_datetime(epoch_ms, unit="ms").dt.normalize()

    return (
        raw[KEEP]
        .drop_duplicates(subset="Date")
        .sort_values("Date")           # ascending, oldest first
        .reset_index(drop=True)
    )


def assert_continuous(df):
    """The whole point of this script. A gap here silently corrupts every rolling feature."""
    span = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing = span.difference(df["Date"])
    if len(missing):
        raise ValueError(
            f"{len(missing)} missing day(s) in the series, e.g. "
            f"{[str(x.date()) for x in missing[:10]]}. Do not build features on this."
        )
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates in the series.")
    if df.isna().any().any():
        raise ValueError("NaNs in the series.")


if __name__ == "__main__":
    ohlcv = fetch()
    assert_continuous(ohlcv)

    ohlcv.to_csv(OUT, index=False)

    print(f"wrote {OUT}")
    print(f"span : {ohlcv['Date'].min().date()} -> {ohlcv['Date'].max().date()}")
    print(f"rows : {len(ohlcv)}  (continuous, no gaps, no dupes, no NaNs)")
