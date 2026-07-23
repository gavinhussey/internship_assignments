"""Pull LME aluminium alloy (and primary) history from Bloomberg -> tracker CSVs.

Writes the exact format scrap_tracker.py consumes:
    data/lme_alloy.csv     [date, alloy_close]    (USD/tonne)  -> cast-side anchor
    data/lme_aluminum.csv  [date, lme_al_close]   (USD/tonne)  -> primary anchor (refresh)

Connection: needs the Bloomberg Python API and a live session — a Bloomberg Terminal
(logged in) or a B-PIPE/Server-API endpoint, reachable at host:port (default
localhost:8194). Works through either backend, auto-detected:
  * xbbg  (preferred — pandas wrapper around blpapi), or
  * blpapi (the official low-level API; a raw HistoricalDataRequest is implemented here).

Install the API (special index URL):
    pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
    pip install xbbg          # optional but recommended

Usage:
    python bloomberg_fetch.py --check                 # connect + print last 5 rows for alloy
    python bloomberg_fetch.py                          # refresh alloy (default security)
    python bloomberg_fetch.py --security lme_aluminum  # refresh the primary instead
    python bloomberg_fetch.py --security all           # refresh both
    python bloomberg_fetch.py --ticker "LMNADS03 Comdty" --field PX_SETTLE   # ad-hoc override

After fetching, run:  python scrap_tracker.py --check   (verifies the new alloy isn't stale).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "bloomberg_config.json"


def log(m: str) -> None:
    print(f"[bbg_fetch] {m}", flush=True)


def load_config() -> dict:
    with open(CONFIG) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Backends — return a tidy DataFrame[date, value] for one security/field.
# ---------------------------------------------------------------------------
def _fetch_xbbg(ticker: str, field: str, start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    raw = blp.bdh(tickers=ticker, flds=field, start_date=start, end_date=end)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "value"])
    # xbbg returns MultiIndex columns (ticker, field); collapse to a single series.
    s = raw.iloc[:, 0]
    out = s.rename("value").reset_index().rename(columns={"index": "date"})
    out.columns = ["date", "value"]
    return out


def _fetch_blpapi(ticker: str, field: str, start: str, end: str,
                  host: str, port: int) -> pd.DataFrame:
    import blpapi
    opts = blpapi.SessionOptions()
    opts.setServerHost(host)
    opts.setServerPort(port)
    session = blpapi.Session(opts)
    if not session.start():
        raise ConnectionError(f"blpapi: could not start session to {host}:{port} "
                              "(is the Terminal running / are you logged in?)")
    try:
        if not session.openService("//blp/refdata"):
            raise ConnectionError("blpapi: could not open //blp/refdata service.")
        svc = session.getService("//blp/refdata")
        req = svc.createRequest("HistoricalDataRequest")
        req.getElement("securities").appendValue(ticker)
        req.getElement("fields").appendValue(field)
        req.set("startDate", pd.to_datetime(start).strftime("%Y%m%d"))
        req.set("endDate", pd.to_datetime(end).strftime("%Y%m%d"))
        req.set("periodicitySelection", "DAILY")
        session.sendRequest(req)
        rows = []
        while True:
            ev = session.nextEvent(500)
            for msg in ev:
                sd = msg.getElement("securityData") if msg.hasElement("securityData") else None
                if sd is None:
                    continue
                if sd.hasElement("securityError"):
                    raise ValueError(f"blpapi securityError for {ticker}: "
                                     f"{sd.getElement('securityError')}")
                fd = sd.getElement("fieldData")
                for i in range(fd.numValues()):
                    pt = fd.getValueAsElement(i)
                    if pt.hasElement(field):
                        rows.append((pt.getElementAsDatetime("date"),
                                     pt.getElementAsFloat(field)))
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
        return pd.DataFrame(rows, columns=["date", "value"])
    finally:
        session.stop()


def fetch(ticker: str, field: str, start: str, end: str, host: str, port: int) -> pd.DataFrame:
    """Try xbbg first, then raw blpapi. Raises a clear error if neither is usable."""
    try:
        import xbbg  # noqa: F401
        log(f"backend=xbbg  {ticker}  {field}  {start}..{end}")
        return _fetch_xbbg(ticker, field, start, end)
    except ImportError:
        pass
    try:
        import blpapi  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "No Bloomberg backend available. Install the API:\n"
            "  pip install --index-url=https://blpapi.bloomberg.com/repository/releases/"
            "python/simple/ blpapi\n"
            "  pip install xbbg   # optional wrapper\n"
            "and ensure a Bloomberg Terminal (logged in) or B-PIPE session is running."
        ) from e
    log(f"backend=blpapi  {ticker}  {field}  {start}..{end}")
    return _fetch_blpapi(ticker, field, start, end, host, port)


# ---------------------------------------------------------------------------
def _tidy(df: pd.DataFrame, out_col: str) -> pd.DataFrame:
    df = df.dropna().copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = (df.sort_values("date").drop_duplicates("date", keep="last")
          .rename(columns={"value": out_col}).reset_index(drop=True))
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce")
    return df.dropna(subset=[out_col])[["date", out_col]]


def write_csv(df: pd.DataFrame, out_csv: Path, out_col: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    z = float((df[out_col].pct_change() == 0).mean()) if len(df) > 1 else float("nan")
    log(f"wrote {out_csv}  rows={len(df)}  "
        f"{df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}  "
        f"zero-change={z:.1%}" + ("  ⚠️ POSSIBLY STALE (>5%)" if z > 0.05 else ""))


def run_security(cfg: dict, key: str, field: str | None,
                 ticker_override: str | None, end: str) -> None:
    spec = cfg["securities"][key]
    ticker = ticker_override or spec["ticker"]
    fld = field or cfg.get("field", "PX_LAST")
    raw = fetch(ticker, fld, cfg.get("start_date", "2014-06-01"), end,
                cfg.get("host", "localhost"), int(cfg.get("port", 8194)))
    if raw.empty:
        log(f"{key}: NO DATA returned for {ticker} / {fld}. Verify the security string in "
            f"the Terminal (the placeholder tickers in bloomberg_config.json are guesses).")
        return
    tidy = _tidy(raw, spec["out_col"])
    write_csv(tidy, BASE / spec["out_csv"], spec["out_col"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--security", default="lme_alloy",
                    help="which config security: lme_alloy | lme_aluminum | all")
    ap.add_argument("--ticker", default=None, help="ad-hoc Bloomberg security override")
    ap.add_argument("--field", default=None, help="override field (e.g. PX_SETTLE)")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--check", action="store_true",
                    help="connectivity test: pull last 5 rows for the chosen security and print")
    args = ap.parse_args()

    cfg = load_config()

    if args.check:
        key = "lme_alloy" if args.security == "all" else args.security
        spec = cfg["securities"][key]
        ticker = args.ticker or spec["ticker"]
        fld = args.field or cfg.get("field", "PX_LAST")
        start = (pd.Timestamp.today() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        df = fetch(ticker, fld, start, args.end, cfg.get("host", "localhost"),
                   int(cfg.get("port", 8194)))
        if df.empty:
            log(f"CHECK: connected but no data for {ticker}/{fld} — verify the security string.")
            return 1
        log(f"CHECK OK — {ticker} {fld}, last rows:")
        print(_tidy(df, "value").tail(5).to_string(index=False))
        return 0

    keys = list(cfg["securities"]) if args.security == "all" else [args.security]
    if any(k not in cfg["securities"] for k in keys):
        log(f"unknown --security '{args.security}'. Options: "
            f"{', '.join(cfg['securities'])}, all")
        return 2
    for k in keys:
        run_security(cfg, k, args.field, args.ticker if args.security != "all" else None, args.end)
    log("done. Next: python scrap_tracker.py --check  (staleness guard on the new data).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        sys.exit(1)
