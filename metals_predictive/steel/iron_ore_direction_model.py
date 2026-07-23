"""Iron Ore Direction Model — the LIQUID ferrous-benchmark rebuild (steel's LME analog).

WHY THIS EXISTS
  Steel has no liquid US HRC mark in this repo (HRC=F is 47% no-range; the China complex needs
  Bloomberg). The aluminum fix worked because LME aluminum is the liquid benchmark that actually
  trades. The steel-complex analog that IS available and liquid is **SGX 62% Fe iron ore
  (TIO=F, yfinance)** — the dominant priced ferrous commodity and the key steel/scrap input
  (~2.1% flat days, balanced ~50.7% baseline, vs HRC's 35.6% flat / 30% up-rate).

  This is NOT "HRC steel" — it is the iron-ore leg of the ferrous complex. But it is a clean,
  liquid, honestly-validatable direction target, whereas HRC=F is not.

TARGET:  target_up = 1 if iron_ore_close[t+1] > iron_ore_close[t] else 0.

TIMING / LEAKAGE
  Night-before structure: decision at the day-t US close (~4pm ET); target is the next SGX iron
  ore settle on day t+1. Every day-t feature (US steelmaker/miner cousins close 4pm ET, macro,
  iron-ore price block) precedes the day-(t+1) Asian settle, so all features are leak-free.
  (No same-Asian-day equity leads are used — those settle ~contemporaneously with SGX and would
  risk a leak. So there is intentionally no "morning-of" variant here.)

Run:  python iron_ore_direction_model.py  ->  outputs/iron_ore_direction/
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import steel_close_direction_model as base

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "iron_ore_direction"
IRON_ORE_TK = "TIO=F"
EXT_CSV = BASE_DIR / "data" / "external" / "external_drivers.csv"   # may be absent

# Bloomberg China ferrous CSVs — populated by bloombergSteel.py (optional; skipped if absent)
SHFE_RB_CSV  = BASE_DIR / "data" / "external" / "shfe_rebar.csv"       # column: shfe_rb_close
SHFE_HRC_CSV = BASE_DIR / "data" / "external" / "shfe_hrc.csv"         # column: shfe_hrc_close
DCE_IO_CSV   = BASE_DIR / "data" / "external" / "dce_iron_ore.csv"     # column: dce_io_close
DCE_CC_CSV   = BASE_DIR / "data" / "external" / "dce_coking_coal.csv"  # column: dce_cc_close
SGX_IO_CSV   = BASE_DIR / "data" / "external" / "sgx_iron_ore.csv"     # column: sgx_io_close


def log(m): print(f"[iron_ore] {m}", flush=True)


def _align_series(df_dates: pd.DataFrame, csv_path: Path,
                  price_col: str) -> pd.Series:
    """Load a Bloomberg CSV with [date, <price_col>], align to df_dates, ffill gaps."""
    raw = pd.read_csv(csv_path, parse_dates=["date"])
    raw = raw.rename(columns={c: c.lower().strip() for c in raw.columns})
    raw = (raw[["date", price_col]].dropna()
           .sort_values("date").drop_duplicates("date", keep="last"))
    merged = df_dates[["date"]].reset_index(drop=True).merge(
        raw[["date", price_col]], on="date", how="left"
    )
    merged[price_col] = merged[price_col].ffill()
    return merged[price_col].reset_index(drop=True)


def china_ferrous_features(df: pd.DataFrame, shift_d: int = 0,
                           exclude_sgx: bool = False,
                           include_prefixes: list | None = None) -> pd.DataFrame:
    """China ferrous complex features from Bloomberg CSVs (all optional).

    TIMING GUIDE — which shift_d to use:
      shift_d=0   (night-before model, decide at 4pm ET day t):
        SHFE/DCE closes at ~3pm Beijing = ~2am ET on US day t.
        That is 14 hours before the night-before decision on day t,
        so day-t SHFE/DCE data is genuinely available. Use shift_d=0.

      shift_d=-1  (morning model, decide after DCE/SHFE day t+1 session):
        Use the day-(t+1) SHFE/DCE session (shift(-1)) as an overnight
        lead for TIO=F which settles ~5am ET on day t+1. Use shift_d=-1.

    Run bloombergSteel.py first to populate the CSVs.
    """
    close = df["close"].reset_index(drop=True)
    vol20 = close.pct_change(1).rolling(20).std()
    cols = {}
    n_loaded = 0

    series_map = [
        (SHFE_RB_CSV,  "shfe_rb_close",  "shfe_rb"),
        (SHFE_HRC_CSV, "shfe_hrc_close", "shfe_hrc"),
        (DCE_IO_CSV,   "dce_io_close",   "dce_io"),
        (DCE_CC_CSV,   "dce_cc_close",   "dce_cc"),
        # SGX iron ore Bloomberg = same contract as TIO=F yfinance; only valid at shift_d=0
        # (day-t SGX predicting day t+1 TIO). At shift_d=-1 it's a self-leak: same instrument
        # same settlement date. exclude_sgx=True is set by iron_ore_morning_model.py.
        *([( SGX_IO_CSV,   "sgx_io_close",   "sgx_io")] if not exclude_sgx else []),
    ]

    overnight_rets = []
    for csv_path, col, prefix in series_map:
        if include_prefixes is not None and prefix not in include_prefixes:
            continue
        if not csv_path.exists():
            continue
        px = _align_series(df, csv_path, col)
        ret = px.pct_change(1)
        sig = ret.shift(shift_d) if shift_d != 0 else ret
        cols[f"{prefix}_ret"]    = sig.to_numpy()
        cols[f"{prefix}_ret_z20"]= base._safe_div(sig, vol20).to_numpy()
        cols[f"{prefix}_div"]    = (sig - close.pct_change(1)).to_numpy()
        mom5  = ret.rolling(5).mean()
        mom20 = ret.rolling(20).mean()
        cols[f"{prefix}_mom5"]   = (mom5.shift(shift_d) if shift_d != 0 else mom5).to_numpy()
        cols[f"{prefix}_5_20"]   = base._safe_div(mom5, mom20).subtract(1).shift(
            shift_d if shift_d != 0 else 0).to_numpy()
        overnight_rets.append(sig)
        n_loaded += 1
        log(f"China ferrous: {prefix} ({px.notna().sum()} rows, shift={shift_d})")

    if overnight_rets:
        agg = pd.concat(overnight_rets, axis=1).mean(axis=1)
        cols["china_ferrous_mean"]  = agg.to_numpy()
        cols["china_ferrous_div"]   = (agg - close.pct_change(1)).to_numpy()
        cols["china_ferrous_z20"]   = base._safe_div(agg, vol20).to_numpy()

    if n_loaded == 0:
        log("China ferrous CSVs not found — run bloombergSteel.py to add China complex features")
    else:
        log(f"China ferrous: {len(cols)} features from {n_loaded} series")

    return pd.DataFrame(cols, index=range(len(df))).replace([np.inf, -np.inf], np.nan)


def load_iron_ore() -> tuple[pd.DataFrame, list[str]]:
    """Build the iron-ore primary series (Date, close) from the macro cache TIO=F column."""
    macro = base.fetch_macro()
    col = IRON_ORE_TK if IRON_ORE_TK in macro.columns else base._safe_name(IRON_ORE_TK)
    if col not in macro.columns:
        raise RuntimeError(f"{IRON_ORE_TK} not found in macro cache columns")
    df = (macro[["date", col]].rename(columns={col: "close"})
          .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
          .dropna(subset=["date"]).sort_values("date")
          .drop_duplicates("date", keep="last").reset_index(drop=True))
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df, []


def build_panel():
    df, ext_cols = load_iron_ore()
    df = base.create_target(df)
    Xc = base.steel_features(df, ext_cols).reset_index(drop=True)
    Xd = base.divergence_features(base.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    # All five China ferrous series now have verified tickers with full 2014-2026 history.
    # shift_d=0: SHFE/DCE close ~2am ET, night-before decision at 4pm ET — no leak.
    Xf = china_ferrous_features(df, shift_d=0).reset_index(drop=True)
    X = pd.concat([Xc, Xd, Xf], axis=1)
    y = df["target_up"].to_numpy(int)
    return df, X, y


def per_year(df, y, probs, mask):
    years = pd.to_datetime(df["date"]).dt.year.to_numpy()
    rows = []
    for yr in sorted(np.unique(years[mask])):
        sel = mask & (years == yr)
        if sel.sum() < 30:
            continue
        ys, ps = y[sel], probs[sel]
        rows.append({"year": int(yr), "n": int(sel.sum()),
                     "accuracy": round(accuracy_score(ys, (ps >= 0.5).astype(int)), 4),
                     "auc": round(roc_auc_score(ys, ps), 4) if len(np.unique(ys)) > 1 else np.nan})
    return pd.DataFrame(rows)


def write_summary(ctx, path):
    me, g, py = ctx["metrics"], ctx["gate"], ctx["per_year"]
    best_gate = g.iloc[g["accuracy"].idxmax()]
    md = f"""# Iron Ore (SGX 62% Fe, TIO=F) Direction Model — NIGHT-BEFORE

## Question
Will SGX iron ore close higher tomorrow than today? The liquid ferrous-benchmark rebuild
(steel's LME analog) replacing the thin HRC=F target. See ALUMINUM_STEEL_MODEL_REVIEW.md.

## Data
- Target: TIO=F (SGX 62% Fe iron ore, yfinance) — liquid, ~2.1% flat days.
- Drivers: cross-asset ferrous cousins (steelmakers, iron-ore miners, ETFs) + macro.
- Rows: {ctx['n_rows']} · Date range: {ctx['date_min']} → {ctx['date_max']}

## Timing (leak-free)
Night-before: decision at day-t US close; target is the day-(t+1) SGX settle. All day-t
features precede it.

## Performance (walk-forward, {me['oos_days']} OOS days)
- **Accuracy: {me['accuracy']:.4f}** (95% CI [{me['acc_ci_low']:.4f}, {me['acc_ci_high']:.4f}])
- AUC: {me['auc']:.4f} · Balanced accuracy: {me['balanced_accuracy']:.4f}
- Up base rate: {me['up_base_rate']:.4f} (always-guess-majority = {me['majority_baseline_acc']:.4f})
- Up precision/recall: {me['up_precision']:.3f} / {me['up_recall']:.3f}
- Down precision/recall: {me['down_precision']:.3f} / {me['down_recall']:.3f}

## Deployable confidence gate
| min \\|p-0.5\\| | coverage | accuracy | 95% CI |
|---|---|---|---|
""" + "\n".join(
        f"| {r.min_confidence} | {r.coverage:.1%} | {r.accuracy:.4f} | [{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}] |"
        for r in g.itertuples()) + f"""

Best confident slice: **{best_gate['accuracy']:.4f} on {best_gate['coverage']:.1%} of days**.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
""" + "\n".join(f"| {r.year} | {r.n} | {r.accuracy} | {r.auc} |" for r in py.itertuples()) + f"""

## Model
- {len(ctx['selected_features'])} features selected (L1, C={ctx['chosen_C']}) of {ctx['candidates']}
  candidates. Top: {', '.join(ctx['selected_features'][:8])}

---
*Research model — not financial advice.*
"""
    path.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y = build_panel()
    log(f"Rows={len(y)} features={X.shape[1]} up_rate={y.mean():.4f} "
        f"{df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)
    log(f"acc={metrics['accuracy']:.4f} CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] "
        f"auc={metrics['auc']:.4f} baseline={metrics['majority_baseline_acc']:.4f} OOS={metrics['oos_days']}")
    log("Per-year: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py.itertuples()))

    oos = pd.DataFrame({
        "date": df["date"].dt.date.values[mask], "close": df["close"].values[mask],
        "actual_up": y[mask], "predicted_p_up": probs[mask],
        "predicted_dir": np.where(probs[mask] >= 0.5, "UP", "DOWN"),
        "confidence": np.abs(probs[mask] - 0.5),
        "correct": (((probs[mask] >= 0.5).astype(int)) == y[mask]).astype(int)})
    oos.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    gate.to_csv(OUT_DIR / "confidence_gate.csv", index=False)
    py.to_csv(OUT_DIR / "per_year_stability.csv", index=False)
    pd.DataFrame([{k: (v if not isinstance(v, list) else str(v)) for k, v in metrics.items()}]).to_csv(
        OUT_DIR / "metrics_summary.csv", index=False)

    final = base.fit_final_model(X, y, features)
    joblib.dump(final, OUT_DIR / "model.pkl")
    pd.DataFrame([{"feature": f, "coefficient": c} for f, c in final["coefficients"].items()]).to_csv(
        OUT_DIR / "selected_features.csv", index=False)
    try:
        base.plot_gate(gate, metrics["accuracy"], OUT_DIR / "confidence_gate_chart.png")
    except Exception as e:
        log(f"plot failed: {e}")

    ctx = {"metrics": metrics, "gate": gate, "per_year": py,
           "selected_features": final["selected_features"], "chosen_C": final["chosen_C"],
           "candidates": freport["candidate_features"], "n_rows": len(df),
           "date_min": df["date"].iloc[0].date(), "date_max": df["date"].iloc[-1].date()}
    write_summary(ctx, OUT_DIR / "evaluation_summary.md")
    with open(OUT_DIR / "metrics_summary.json", "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in metrics.items() if not isinstance(v, list)}, fh, indent=2, default=str)
    log(f"Artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
