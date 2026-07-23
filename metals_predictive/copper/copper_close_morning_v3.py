"""Copper Close Direction Model — v3, LME-features only.

Strips the feature set down to only the LME/metals signals:
  lme_overnight_ret     LME copper Ring settlement return (day t+1, available ~7:15 AM CT)
  lme_overnight_z20     Same return scaled to 20-day vol
  lme_comex_basis       LME - COMEX basis (arb pressure, day t)
  lme_comex_basis_chg   Daily change in basis
  lme_al_overnight_ret  LME aluminum overnight return (if lme_aluminum.csv present)
  lme_zn_overnight_ret  LME zinc overnight return (if lme_zinc.csv present)
  shfe_cu_overnight_ret SHFE copper return (if shfe_copper.csv present)
  metals_composite      Equal-weight average of all overnight metal returns

No COMEX technicals, no macro divergence, no Asian miners.
Run:  python copper_close_morning_v3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import copper_close_direction_model as base
import copper_close_morning_model as morning

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "outputs" / "copper_close_morning_v3"


def log(m): print(f"[v3_lme_only] {m}", flush=True)


def build_panel_lme_only():
    primary, reason = base.find_dataset(BASE_DIR)
    external = base.find_external_file(BASE_DIR, primary)
    df, meta = base.load_and_prepare(primary, external)
    df = base.create_target(df)

    Xl = morning.lme_features(df).reset_index(drop=True)
    if Xl.shape[1] == 0:
        raise RuntimeError(
            "No LME features found. Run bloombergCopper.py first to populate "
            "data/external/lme_copper.csv (required)."
        )
    y = df["target_up"].to_numpy(int)
    return df, Xl, y, reason


def per_year(df, y, probs, mask):
    years = pd.to_datetime(df["date"]).dt.year.to_numpy()
    rows = []
    for yr in sorted(np.unique(years[mask])):
        sel = mask & (years == yr)
        if sel.sum() < 30:
            continue
        ys, ps = y[sel], probs[sel]
        rows.append({
            "year": int(yr), "n": int(sel.sum()),
            "accuracy": round(accuracy_score(ys, (ps >= 0.5).astype(int)), 4),
            "auc": round(roc_auc_score(ys, ps), 4) if len(np.unique(ys)) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, reason = build_panel_lme_only()
    log(reason)
    log(f"Rows={len(y)}  LME features={X.shape[1]}  up_rate={y.mean():.4f}")
    log(f"Features: {list(X.columns)}")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)

    log(f"[LR]  acc={metrics['accuracy']:.4f} "
        f"CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] "
        f"auc={metrics['auc']:.4f}  OOS={metrics['oos_days']}")
    log("Per-year: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py.itertuples()))
    log(f"Features selected: {len(features)} — {features}")

    oos = pd.DataFrame({
        "date":          df["date"].dt.date.values[mask],
        "close":         df["close"].values[mask],
        "actual_up":     y[mask],
        "predicted_p_up": probs[mask],
        "predicted_dir": np.where(probs[mask] >= 0.5, "UP", "DOWN"),
        "confidence":    np.abs(probs[mask] - 0.5),
        "correct":       ((probs[mask] >= 0.5).astype(int) == y[mask]).astype(int),
    })
    oos.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    gate.to_csv(OUT_DIR / "confidence_gate.csv", index=False)
    py.to_csv(OUT_DIR / "per_year_stability.csv", index=False)
    pd.DataFrame([{k: (v if not isinstance(v, list) else str(v))
                   for k, v in metrics.items()}]).to_csv(
        OUT_DIR / "metrics_summary.csv", index=False)
    with open(OUT_DIR / "metrics_summary.json", "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in metrics.items() if not isinstance(v, list)},
                  fh, indent=2, default=str)

    final = base.fit_final_model(X, y, features)
    joblib.dump(final, OUT_DIR / "model.pkl")
    pd.DataFrame([{"feature": f, "coefficient": c}
                  for f, c in final["coefficients"].items()]).to_csv(
        OUT_DIR / "selected_features.csv", index=False)
    log(f"Final model: C={final['chosen_C']}, {len(final['selected_features'])} features.")

    try:
        base.plot_gate(gate, metrics["accuracy"], OUT_DIR / "confidence_gate_chart.png")
    except Exception as e:
        log(f"plot failed: {e}")

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
