"""Fetch LME Copper 3-month settlement + COMEX 2nd-month generic future via Bloomberg Terminal.

Requires Bloomberg Terminal running and logged in (port 8194).

Tickers:
    LMCADS03 Comdty — LME Copper 3-month official settlement, USD/tonne.
        -> data/external/lme_copper.csv [date, lme_close]
        Picked up by copper_close_morning_model.py.
    HG2 Comdty — COMEX Copper 2nd-nearby generic future settlement, USD/lb.
        -> data/external/comex_copper_hg2.csv [date, hg2_close]
        Paired with the front-month close (data/raw/copper.csv) to build a
        calendar-spread (contango/backwardation) feature for the NIGHT-BEFORE
        model — both legs settle same-day on COMEX, so this is available at
        night-before prediction time with no overnight-session dependency.
        Picked up by copper_close_direction_model.py.

Install one of:
    pip install pdblp
    pip install xbbg
"""
import sys
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/external")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "20140601"
END   = pd.Timestamp.today().strftime("%Y%m%d")

SERIES = [
    ("LMCADS03 Comdty", "lme_close", OUT_DIR / "lme_copper.csv"),
    ("HG2 Comdty",      "hg2_close", OUT_DIR / "comex_copper_hg2.csv"),
]


def fetch_pdblp(ticker: str, col: str) -> pd.DataFrame:
    import pdblp
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    try:
        df = con.bdh(tickers=[ticker], flds=["PX_LAST"],
                     start_date=START, end_date=END)
        df = df.droplevel(0, axis=1).reset_index()
        df.columns = ["date", col]
        return df
    finally:
        con.stop()


def fetch_xbbg(ticker: str, col: str) -> pd.DataFrame:
    from xbbg import blp
    result = blp.bdh(tickers=ticker, flds="PX_LAST", start_date="2014-06-01")
    # xbbg may return a narwhals DataFrame in newer versions — convert to pandas
    if hasattr(result, "to_native"):
        df = result.to_native()
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
    else:
        df = pd.DataFrame(result)
    if df is None or len(df) == 0:
        raise RuntimeError("Bloomberg returned empty data — is Terminal running and logged in?")
    # Handle long format: [ticker, date, field, value]
    if "value" in df.columns and "date" in df.columns:
        df = df[df["field"] == "PX_LAST"][["date", "value"]].copy()
        df.columns = ["date", col]
        return df
    # Handle wide format with MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1).reset_index()
    else:
        df = df.reset_index()
    df.columns = ["date", col]
    return df


def fetch_one(ticker: str, col: str) -> pd.DataFrame | None:
    for name, fn in [("pdblp", fetch_pdblp), ("xbbg", fetch_xbbg)]:
        try:
            print(f"  Trying {name} for {ticker}...")
            df = fn(ticker, col)
            print(f"    {name} succeeded")
            return df
        except ImportError:
            print(f"    {name} not installed — skipping")
        except Exception as e:
            print(f"    {name} failed: {e}")
    return None


def main():
    any_ok = False
    for ticker, col, out_path in SERIES:
        print(f"Fetching {ticker} -> {out_path.name}")
        df = fetch_one(ticker, col)
        if df is None:
            print(f"  FAILED: both methods failed for {ticker}. Bloomberg Terminal running/logged in?")
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=[col]).sort_values("date").drop_duplicates("date")
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows ({df['date'].min().date()} - {df['date'].max().date()}) to {out_path}")
        any_ok = True

    if not any_ok:
        print("\nAll tickers failed. Install a Bloomberg Python library:  pip install pdblp  OR  pip install xbbg")
        sys.exit(1)

    print("Re-run copper_close_morning_model.py / copper_close_direction_model.py to incorporate the new features.")


if __name__ == "__main__":
    main()
