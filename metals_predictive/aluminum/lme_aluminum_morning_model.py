"""LME Aluminum Direction Model — MORNING-OF (honest rebuild).

Same target as lme_aluminum_direction_model (next-day LME 3-month aluminum settlement), but the
guess is made on the MORNING of day t+1, after the Asian/Australian session has closed and
BEFORE the ~7:15am ET LME Ring settlement. That lets it use the overnight Asian producer +
SHFE moves of day t+1, which precede (and plausibly lead) the LME settle.

WHAT IS DELIBERATELY EXCLUDED (the old model's leaks):
  * NO lme_overnight_ret of t+1 — that IS the target (LME-on-LME self-leak).
  * NO next_gap_return / us_open_features — there is no US open before the LME settle, and on a
    real trading series the gap would not be a tautology anyway. Both gone.
The only overnight signal is genuinely-leading foreign sessions that close before the LME settle.

LEAK-FREE TIMING (all precede the ~7:15am ET LME settle on day t+1):
  ASX producers (BHP.AX, RIO.AX, S32.AX) close ~1am ET; HK (1378/2600/0486.HK) ~4am ET;
  SHFE day session ~3am ET. Each is shift(-1): row t carries the day-(t+1) session.

Validation: a no-shift control (same-day session) must NOT beat the night-before baseline; if it
does, the "lead" is really contemporaneous and the result is suspect.

Run:  python lme_aluminum_morning_model.py  ->  outputs/lme_aluminum_close_morning/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import aluminum_close_direction_model as base
import aluminum_close_morning_model as morn
import lme_aluminum_direction_model as night

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "lme_aluminum_close_morning"
SHFE_CU_CSV  = BASE_DIR / "data" / "external" / "shfe_copper.csv"
SHFE_AL_CSV  = BASE_DIR / "data" / "external" / "shfe_aluminum.csv"     # from bloomberg_aluminum.py
LME_AL_INV   = BASE_DIR / "data" / "external" / "lme_al_inventory.csv"  # from bloomberg_aluminum.py
LME_AL_CW    = BASE_DIR / "data" / "external" / "lme_al_cancelled_warrants.csv"  # from bloomberg_aluminum.py
TTF_GAS_CSV  = BASE_DIR / "data" / "external" / "ttf_gas.csv"            # from bloomberg_aluminum.py


def log(m): print(f"[lme_aluminum_morning] {m}", flush=True)


def shfe_overnight(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """SHFE copper day-(t+1) return — China industrial-demand lead, closes ~3am ET (pre-LME).

    Leak-free China overnight lead for the LME settle. shift=False is the control variant.
    """
    if not SHFE_CU_CSV.exists():
        return pd.DataFrame(index=range(len(df)))
    sh = morn._align_series(df, SHFE_CU_CSV, "shfe_cu_close")
    ret = sh.pct_change(1)
    sig = ret.shift(-1) if shift else ret
    return pd.DataFrame({"shfe_cu_overnight_ret": sig.to_numpy()},
                        index=range(len(df))).replace([np.inf, -np.inf], np.nan)


def bloomberg_features(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Bloomberg-sourced aluminum fundamentals (all optional — skipped if CSV absent).

    Timing for the morning model: all these data points are available before the ~7:15am ET
    LME Ring settle.
      SHFE aluminum     : closes ~3pm Beijing = ~2am ET day t+1  (shift=-1 → overnight lead)
      LME inventory     : updated daily, published by ~6am ET    (shift=0  → same-day level)
      LME cancelled wts : same publication schedule              (shift=0  → same-day level)
      TTF gas           : European settlement, available ~7am ET  (shift=-1 → overnight)

    For the no-shift control (shift=False), SHFE aluminum and TTF use shift(0) instead, which
    should NOT beat the night-before baseline if the overnight lead is real.

    Run bloomberg_aluminum.py first to populate the CSVs.
    """
    close = df["close"].reset_index(drop=True)
    vol20 = close.pct_change(1).rolling(20).std()
    cols = {}
    n_loaded = 0

    # SHFE aluminum — direct China overnight lead for LME aluminum
    if SHFE_AL_CSV.exists():
        al = morn._align_series(df, SHFE_AL_CSV, "shfe_al_close")
        ret = al.pct_change(1)
        sig = ret.shift(-1) if shift else ret
        cols["shfe_al_overnight_ret"] = sig.to_numpy()
        cols["shfe_al_overnight_z20"] = base._safe_div(sig, vol20).to_numpy()
        cols["shfe_al_overnight_div"] = (sig - close.pct_change(1)).to_numpy()
        n_loaded += 1
        log(f"Bloomberg: SHFE aluminum ({al.notna().sum()} rows)")

    # LME on-warrant stocks — inventory level (shift=0 in both real and control)
    if LME_AL_INV.exists():
        inv = morn._align_series(df, LME_AL_INV, "lme_al_stocks")
        inv_ret  = inv.pct_change(1)
        inv_z20  = base._safe_div(inv_ret, inv_ret.rolling(20).std())
        cols["lme_al_inv_chg"]  = inv_ret.to_numpy()
        cols["lme_al_inv_z20"]  = inv_z20.to_numpy()
        cols["lme_al_inv_lvl"]  = (inv / inv.rolling(60).mean() - 1).to_numpy()
        n_loaded += 1
        log(f"Bloomberg: LME aluminum inventory ({inv.notna().sum()} rows)")

    # LME cancelled warrants — rising % → drawdown pending → bullish signal (shift=0)
    if LME_AL_CW.exists():
        cw = morn._align_series(df, LME_AL_CW, "lme_al_cw")
        cw_chg = cw.diff(1)
        cw_z20 = base._safe_div(cw_chg, cw_chg.rolling(20).std())
        cols["lme_al_cw_pct"]  = cw.to_numpy()
        cols["lme_al_cw_chg"]  = cw_chg.to_numpy()
        cols["lme_al_cw_z20"]  = cw_z20.to_numpy()
        n_loaded += 1
        log(f"Bloomberg: LME aluminum cancelled warrants ({cw.notna().sum()} rows)")

    # TTF natural gas — aluminum smelting cost proxy (~40% of production cost)
    if TTF_GAS_CSV.exists():
        ttf = morn._align_series(df, TTF_GAS_CSV, "ttf_close")
        ttf_ret  = ttf.pct_change(1)
        ttf_vol  = ttf_ret.rolling(20).std()
        sig_ttf  = ttf_ret.shift(-1) if shift else ttf_ret
        cols["ttf_overnight_ret"] = sig_ttf.to_numpy()
        cols["ttf_vol_20d"]       = ttf_vol.to_numpy()
        cols["ttf_ret_5d"]        = ttf_ret.rolling(5).mean().to_numpy()
        n_loaded += 1
        log(f"Bloomberg: TTF gas ({ttf.notna().sum()} rows)")

    if n_loaded == 0:
        log("Bloomberg CSVs not found — run bloomberg_aluminum.py to add aluminum fundamentals")
    else:
        log(f"Bloomberg features: {len(cols)} columns from {n_loaded} sources")

    return pd.DataFrame(cols, index=range(len(df))).replace([np.inf, -np.inf], np.nan)


def build_panel(shift: bool = True):
    """Night-before features (day t) + leak-free overnight leads (day t+1 Asian session)."""
    df, ext_cols = night.load_lme()
    df = base.create_target(df)
    Xc = base.metal_features(df, ext_cols).reset_index(drop=True)
    Xd = base.divergence_features(base.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    overnight_prices = morn.fetch_overnight()
    if shift:
        Xo = morn.overnight_features(overnight_prices, df["close"], df["date"]).reset_index(drop=True)
    else:
        Xo = morn.overnight_features_no_shift(overnight_prices, df["close"], df["date"]).reset_index(drop=True)
    Xs = shfe_overnight(df, shift=shift).reset_index(drop=True)
    Xb = bloomberg_features(df, shift=shift).reset_index(drop=True)
    X = pd.concat([Xc, Xd, Xo, Xs, Xb], axis=1)
    y = df["target_up"].to_numpy(int)
    n_over = Xo.shape[1] + Xs.shape[1] + Xb.shape[1]
    return df, X, y, n_over


def write_summary(ctx, path):
    me, g, py = ctx["metrics"], ctx["gate"], ctx["per_year"]
    md = f"""# LME Aluminum Close Direction Model — MORNING-OF

## What this is
Predicts the next-day **LME 3-month aluminum** settlement, guessed on the morning of day t+1
after the Asian/AU session closes and before the ~7:15am ET LME Ring settle. Honest rebuild of
the corrupted COMEX-ALI morning model: the LME-on-LME self-leak and the open-gap tautology are
both removed (see ALUMINUM_STEEL_MODEL_REVIEW.md).

## Overnight features (leak-free, all close before the LME settle)
ASX/HK aluminum producers (BHP.AX, RIO.AX, S32.AX, 1378.HK, 2600.HK, 0486.HK) + SHFE copper,
day-(t+1) session, shift(-1)-aligned.

## No-shift control (the honesty check)
Real model acc {me['accuracy']:.4f} / AUC {me['auc']:.4f}.
No-shift control acc {ctx['ctrl_acc']:.4f} / AUC {ctx['ctrl_auc']:.4f}.
**Overnight shift lift: {me['accuracy'] - ctx['ctrl_acc']:+.4f} acc / {me['auc'] - ctx['ctrl_auc']:+.4f} AUC.**
A positive, meaningful lift is the only thing that makes the morning timing worthwhile.

## Performance (walk-forward, {me['oos_days']} OOS days)
- **Accuracy: {me['accuracy']:.4f}** (95% CI [{me['acc_ci_low']:.4f}, {me['acc_ci_high']:.4f}])
- AUC: {me['auc']:.4f} · Balanced accuracy: {me['balanced_accuracy']:.4f}
- Up base rate: {me['up_base_rate']:.4f} (always-guess-majority = {me['majority_baseline_acc']:.4f})
- Up precision/recall: {me['up_precision']:.3f} / {me['up_recall']:.3f}
- Down precision/recall: {me['down_precision']:.3f} / {me['down_recall']:.3f}

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
""" + "\n".join(f"| {r.year} | {r.n} | {r.accuracy} | {r.auc} |" for r in py.itertuples()) + f"""

## Model
- {len(ctx['selected_features'])} features selected (L1, C={ctx['chosen_C']}) of {ctx['candidates']}
  candidates; LME price block + cross-asset divergence + leak-free overnight Asian/SHFE leads.
- Top features: {', '.join(ctx['selected_features'][:8])}

---
*Research model — not financial advice. The morning-of timing is only valid if the overnight
shift lift above is positive.*
"""
    path.write_text(md, encoding="utf-8")


def per_year(df, y, probs, mask):
    return night.per_year(df, y, probs, mask)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, n_over = build_panel(shift=True)
    log(f"Rows={len(y)} features={X.shape[1]} (overnight={n_over}) up_rate={y.mean():.4f}")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)
    log(f"acc={metrics['accuracy']:.4f} CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] "
        f"auc={metrics['auc']:.4f} baseline={metrics['majority_baseline_acc']:.4f} OOS={metrics['oos_days']}")
    log("Per-year: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py.itertuples()))

    # No-shift control
    log("--- No-shift control (same-day session; should NOT beat the baseline) ---")
    _, X_ctrl, _, _ = build_panel(shift=False)
    probs_ctrl, _, _ = base.walk_forward(X_ctrl, y)
    mctrl = ~np.isnan(probs_ctrl)
    ctrl_acc = accuracy_score(y[mctrl], (probs_ctrl[mctrl] >= 0.5).astype(int))
    ctrl_auc = roc_auc_score(y[mctrl], probs_ctrl[mctrl])
    log(f"No-shift control: acc={ctrl_acc:.4f} auc={ctrl_auc:.4f}  "
        f"(real: acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f})")
    log(f"Overnight lift: acc {metrics['accuracy'] - ctrl_acc:+.4f}  auc {metrics['auc'] - ctrl_auc:+.4f}")
    pd.DataFrame([{"ctrl_acc": ctrl_acc, "ctrl_auc": ctrl_auc,
                   "real_acc": metrics["accuracy"], "real_auc": metrics["auc"],
                   "acc_lift": metrics["accuracy"] - ctrl_acc,
                   "auc_lift": metrics["auc"] - ctrl_auc}]).to_csv(
        OUT_DIR / "overnight_shift_validation.csv", index=False)

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
           "candidates": freport["candidate_features"], "ctrl_acc": ctrl_acc, "ctrl_auc": ctrl_auc}
    write_summary(ctx, OUT_DIR / "evaluation_summary.md")
    with open(OUT_DIR / "metrics_summary.json", "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in metrics.items() if not isinstance(v, list)}, fh, indent=2, default=str)
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
