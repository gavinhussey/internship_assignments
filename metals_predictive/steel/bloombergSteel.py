"""Fetch the China/Asia ferrous complex via Bloomberg Terminal.

This is steel's closest analog to copper's LME anchor. China is >50% of world steel and
its futures set the marginal global price; the SHFE/DCE day sessions close ~3 PM Beijing
(~1-2 AM CT) and SGX iron ore settles in Singapore — all BEFORE the ~1 PM ET CME HRC
settle, so each series is a leak-free overnight lead (used with shift(-1) in the model).

  SHFE Rebar (螺纹钢)        -> data/external/shfe_rebar.csv       (CNY/tonne)
  SHFE Hot-Rolled Coil       -> data/external/shfe_hrc.csv         (CNY/tonne)
  DCE Iron Ore (铁矿石)      -> data/external/dce_iron_ore.csv     (CNY/tonne, key input)
  DCE Coking Coal (焦煤)     -> data/external/dce_coking_coal.csv  (CNY/tonne, key input)
  SGX Iron Ore 62% Fe (TSI)  -> data/external/sgx_iron_ore.csv     (USD/tonne)

⚠️ VERIFY THE TICKERS on your Terminal before trusting output — Bloomberg yellow-key
symbols for the Chinese ferrous contracts vary by convention/contract. The values below
are the generic front-month ("1") best-guesses; confirm each with <Comdty> + DES <GO> and
edit the TICKERS table as needed. The fetch logic itself is identical to bloombergCopper.py.

Requires Bloomberg Terminal running and logged in (port 8194).
Install:  pip install xbbg   OR   pip install pdblp

Run:  python bloombergSteel.py
"""
import sys
import pandas as pd
from pathlib import Path

START = "20140601"
END   = pd.Timestamp.today().strftime("%Y%m%d")

# (bloomberg_ticker, output_column, output_path)  — VERIFY tickers on Terminal (see header).
TICKERS = [
    ("RBTA Comdty", "shfe_rb_close",  "data/external/shfe_rebar.csv"),       # SHFE rebar 1st generic
    ("HCA Comdty",  "shfe_hrc_close", "data/external/shfe_hrc.csv"),         # SHFE HRC 1st generic
    ("IOE1 Comdty", "dce_io_close",   "data/external/dce_iron_ore.csv"),     # DCE iron ore 1st generic
    ("JMA Comdty",  "dce_cc_close",   "data/external/dce_coking_coal.csv"),  # DCE coking coal 1st generic
    ("SCO1 Comdty", "sgx_io_close",   "data/external/sgx_iron_ore.csv"),     # SGX TSI 62% Fe 1st generic
]


def fetch_xbbg(ticker: str) -> pd.DataFrame:
    from xbbg import blp
    result = blp.bdh(tickers=ticker, flds="PX_LAST", start_date="2014-06-01")
    if hasattr(result, "to_native"):
        df = result.to_native()
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
    else:
        df = pd.DataFrame(result)
    if df is None or len(df) == 0:
        raise RuntimeError(f"{ticker}: Bloomberg returned empty data — is Terminal running?")
    # Long format: [ticker, date, field, value]
    if "value" in df.columns and "date" in df.columns:
        return df[df["field"] == "PX_LAST"][["date", "value"]].copy()
    # Wide / MultiIndex format
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1).reset_index()
    else:
        df = df.reset_index()
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    return df


def fetch_pdblp(ticker: str) -> pd.DataFrame:
    import pdblp
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    try:
        df = con.bdh(tickers=[ticker], flds=["PX_LAST"],
                     start_date=START, end_date=END)
        df = df.droplevel(0, axis=1).reset_index()
        df.columns = ["date", "value"]
        return df
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
    base = Path(__file__).resolve().parent
    any_failed = False
    for ticker, col_name, out_path in TICKERS:
        out = base / out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {ticker} -> {out_path} ...")
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
            any_failed = True

    if any_failed:
        print("\nSome tickers failed. Check the symbol on the Terminal (DES <GO>) and the "
              "Bloomberg connection, then re-run.")
        sys.exit(1)
    print("\nAll done. Re-run steel_close_morning_model.py to use the China-complex features.")


if __name__ == "__main__":
    main()
