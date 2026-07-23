"""Steel Equity Basket Direction Model — NIGHT-BEFORE (liquid target rebuild).

WHY THIS EXISTS
  HRC=F (CME US HRC steel futures on Yahoo) is too illiquid for a daily direction signal:
  35.6% of days are flat (close == prior close), median volume ~11 contracts, and 47% of
  days have high == low. Any model on HRC=F below 64% is below the always-flat baseline.

  This model replaces HRC=F with an **equal-weight basket of liquid US steel equity prices**
  (NUE + STLD + MT), which:
    - Trade genuinely (millions of shares daily, no stale prints)
    - Are highly correlated with steel price direction (~0.75 with HRC)
    - Have a balanced up-rate (~51%), giving a fair 50% baseline
    - Are available on yfinance for free

TARGET:  target_up = 1 if basket_return[t+1] > 0 else 0
         where basket_return = equal-weight daily return of (NUE, STLD, MT)

TIMING / LEAKAGE
  Night-before: decision at US close (~4pm ET) on day t; target is the basket return on day
  t+1 (all three stocks close at ~4pm ET). All day-t features precede the target. Leak-free.

BASKET CONSTRUCTION
  basket_close[t] = (1 + mean_daily_ret).cumprod() * 100
  Uses mean of pct_change(1) across available tickers each day (handles NaN if one stock
  is missing). Target is sign of next basket return — equivalent to sign of basket_close[t+1]
  vs basket_close[t].

Run:  python steel_equity_basket_model.py  ->  outputs/steel_equity_basket/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import steel_close_direction_model as base

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "outputs" / "steel_equity_basket"
MACRO_CACHE = BASE_DIR / "data" / "external" / "macro_cache" / "macro_prices_expanded.csv"

BASKET_TICKERS = ["NUE", "STLD", "MT"]


def log(m): print(f"[steel_basket] {m}", flush=True)


def build_basket() -> pd.DataFrame:
    """Construct equal-weight NUE+STLD+MT basket from the steel macro cache.

    The macro cache already has these tickers downloaded (they're in DIVERGENCE_TICKERS).
    Returns a DataFrame with columns [date, close] where close is the synthetic basket
    price index (starts at 100, compounds daily equal-weight returns).
    """
    macro = base.fetch_macro()
    mp = macro.set_index("date").sort_index()
    available = [tk for tk in BASKET_TICKERS if tk in mp.columns
                 and mp[tk].notna().sum() > 200]
    if not available:
        raise RuntimeError(f"None of {BASKET_TICKERS} found in macro cache {MACRO_CACHE}")
    log(f"Basket tickers available: {available}")
    rets = pd.concat([mp[tk].pct_change(1) for tk in available], axis=1)
    rets.columns = available
    basket_ret = rets.mean(axis=1, skipna=True)
    basket_close = (1 + basket_ret.fillna(0)).cumprod() * 100
    df = pd.DataFrame({
        "date": basket_close.index,
        "close": basket_close.values,
    }).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    log(f"Basket rows={len(df)} up_rate={((basket_ret > 0).mean()):.4f} "
        f"({df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()})")
    return df


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
    md = f"""# Steel Equity Basket Direction Model — NIGHT-BEFORE

## Question
Will the equal-weight NUE+STLD+MT steel equity basket close higher tomorrow?
`target_up = 1 if basket_return[t+1] > 0 else 0`. Liquid-target rebuild replacing
the illiquid HRC=F futures target (see ALUMINUM_STEEL_MODEL_REVIEW.md §10).

## Why a basket instead of HRC=F
HRC=F (Yahoo): 35.6% flat days, 47% no-range, median vol ~11 contracts. Any model
runs below the 64.1% "always-flat" baseline. NUE+STLD+MT: genuinely liquid, balanced
~51% up-rate, free on yfinance, ~0.75 corr with HRC direction.

## Target construction
Equal-weight daily return: `basket_ret[t] = mean(NUE_ret[t], STLD_ret[t], MT_ret[t])`.
Basket price index: `(1 + basket_ret).cumprod() * 100` (starts 2014). Target is sign
of the next-day basket return — identical to close[t+1] > close[t] on the index.

## Timing (leak-free)
Night-before: decision at day-t US close (~4pm ET). All features use day-t data.
Target is day-(t+1) basket close — all three stocks trade 9:30am–4pm ET next day.

## Data
- Basket: NUE (Nucor), STLD (Steel Dynamics), MT (ArcelorMittal) equal-weight
- Rows: {ctx['n_rows']} · Date range: {ctx['date_min']} → {ctx['date_max']}

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
- {len(ctx['selected_features'])} features selected (L1, C={ctx['chosen_C']}) of
  {ctx['candidates']} candidates.
- Top features: {', '.join(ctx['selected_features'][:10])}

---
*Research model — not financial advice.*
"""
    path.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_basket()
    df = base.create_target(df)
    macro = base.fetch_macro()
    Xc = base.steel_features(df, []).reset_index(drop=True)
    Xd = base.divergence_features(macro, df["close"], df["date"]).reset_index(drop=True)
    X = pd.concat([Xc, Xd], axis=1)
    y = df["target_up"].to_numpy(int)
    log(f"Rows={len(y)} features={X.shape[1]} up_rate={y.mean():.4f}")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)
    log(f"acc={metrics['accuracy']:.4f} CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] "
        f"auc={metrics['auc']:.4f} baseline={metrics['majority_baseline_acc']:.4f} OOS={metrics['oos_days']}")
    log("Per-year: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py.itertuples()))

    oos = pd.DataFrame({
        "date": df["date"].dt.date.values[mask], "basket_close": df["close"].values[mask],
        "actual_up": y[mask], "predicted_p_up": probs[mask],
        "predicted_dir": np.where(probs[mask] >= 0.5, "UP", "DOWN"),
        "confidence": np.abs(probs[mask] - 0.5),
        "correct": (((probs[mask] >= 0.5).astype(int)) == y[mask]).astype(int)})
    oos.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    gate.to_csv(OUT_DIR / "confidence_gate.csv", index=False)
    py.to_csv(OUT_DIR / "per_year_stability.csv", index=False)
    pd.DataFrame([{k: (v if not isinstance(v, list) else str(v))
                   for k, v in metrics.items()}]).to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    final = base.fit_final_model(X, y, features)
    joblib.dump(final, OUT_DIR / "model.pkl")
    pd.DataFrame([{"feature": f, "coefficient": c}
                  for f, c in final["coefficients"].items()]).to_csv(
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
                   for k, v in metrics.items() if not isinstance(v, list)},
                  fh, indent=2, default=str)
    log(f"Artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
