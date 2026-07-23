"""Steel (CME HRC) MORNING-OF model — HONEST de-leaked rebuild.

WHY THIS EXISTS
  The original steel_close_morning_model scored 73.4% but its lone real feature was
  next_gap_return = open[t+1]/close[t] (coef 0.86, ~10x everything else). On the thin HRC=F
  contract (47% of days have high==low) the open ~= the close, so that feature is a near-
  tautology, not a forecast — the no-shift control matched the headline (0.7332 vs 0.7337),
  proving the overnight steelmakers added nothing. See ALUMINUM_STEEL_MODEL_REVIEW.md.

WHAT THIS CHANGES
  * Keeps HRC=F as the target — it IS the US steel price, settles ~1pm ET, and has a clean
    leak-free overnight lead structure (Asian/EU steelmakers + SGX iron ore all close first).
  * DELETES the leak: no us_open_features / next_gap_return / next_gap_z20.
  * KEEPS only genuinely-leading overnight signals: Asian/EU steelmakers (shift(-1)) PLUS
    SGX iron ore TIO=F (shift(-1)) — iron ore is steel's dominant priced input and the SGX
    settle precedes the 1pm ET HRC settle.
  * Same no-shift control as the honesty gate: if same-day signals match the shifted ones,
    there is no real overnight edge.

Run:  python steel_clean_morning_model.py  ->  outputs/steel_clean_morning/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import steel_close_direction_model as base
import steel_close_morning_model as morn

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "steel_clean_morning"
IRON_ORE_TK = "TIO=F"   # SGX 62% Fe iron ore future (yfinance), in the macro cache


def log(m): print(f"[steel_clean_morning] {m}", flush=True)


def iron_ore_overnight(macro: pd.DataFrame, close: pd.Series, dates: pd.Series,
                       shift: bool = True) -> pd.DataFrame:
    """SGX iron ore (TIO=F) day-(t+1) return — leak-free lead (SGX settles before 1pm ET HRC).

    iron ore is the key priced steel input; this is the closest free analog to a real
    overnight ferrous lead. shift=False is the no-shift control variant.
    """
    name = base._safe_name(IRON_ORE_TK)
    mp = macro.set_index("date").sort_index()
    mp = mp.reindex(pd.Index(dates).union(mp.index)).ffill().reindex(dates).reset_index(drop=True)
    col = IRON_ORE_TK if IRON_ORE_TK in mp.columns else (name if name in mp.columns else None)
    if col is None:
        log("iron ore: TIO=F not in macro cache, skipping")
        return pd.DataFrame(index=range(len(dates)))
    s = pd.to_numeric(mp[col], errors="coerce")
    ret = s.pct_change(1)
    vol20 = ret.rolling(20).std()
    sig = ret.shift(-1) if shift else ret
    hrc_ret1 = close.pct_change(1).reset_index(drop=True)
    cols = {
        "iron_ore_overnight_ret": sig.to_numpy(),
        "iron_ore_overnight_z20": base._safe_div(sig, vol20).to_numpy(),
        "iron_ore_overnight_div": (sig - hrc_ret1).to_numpy(),
    }
    return pd.DataFrame(cols, index=range(len(dates))).replace([np.inf, -np.inf], np.nan)


def build_panel(shift: bool = True):
    primary, reason = base.find_dataset(BASE_DIR)
    external = base.find_external_file(BASE_DIR, primary)
    df, meta = base.load_and_prepare(primary, external)
    df = base.create_target(df)
    macro = base.fetch_macro()
    Xc = base.steel_features(df, meta["merged_external_cols"]).reset_index(drop=True)
    Xd = base.divergence_features(macro, df["close"], df["date"]).reset_index(drop=True)
    overnight_prices = morn.fetch_overnight()
    if shift:
        Xo = morn.overnight_features(overnight_prices, df["close"], df["date"]).reset_index(drop=True)
    else:
        Xo = morn.overnight_features_no_shift(overnight_prices, df["close"], df["date"]).reset_index(drop=True)
    Xio = iron_ore_overnight(macro, df["close"], df["date"], shift=shift).reset_index(drop=True)
    # NOTE: deliberately NO us_open_features (the next_gap_return leak) and NO LME (none exists).
    X = pd.concat([Xc, Xd, Xo, Xio], axis=1)
    y = df["target_up"].to_numpy(int)
    return df, X, y, Xo.shape[1] + Xio.shape[1], reason


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
    md = f"""# Steel (CME HRC) MORNING-OF — de-leaked rebuild

## What changed
Removed the `next_gap_return` open-gap leak that inflated the original 73.4% (it was a
tautology on the thin HRC=F contract). Overnight leads are now only genuinely-leading
sessions: Asian/EU steelmakers + SGX iron ore (TIO=F), all closing before the 1pm ET settle.

## No-shift control (the honesty check)
Real model acc {me['accuracy']:.4f} / AUC {me['auc']:.4f}.
No-shift control acc {ctx['ctrl_acc']:.4f} / AUC {ctx['ctrl_auc']:.4f}.
**Overnight shift lift: {me['accuracy'] - ctx['ctrl_acc']:+.4f} acc / {me['auc'] - ctx['ctrl_auc']:+.4f} AUC.**
Only a positive, meaningful lift makes the morning timing worthwhile.

## Performance (walk-forward, {me['oos_days']} OOS days)
- **Accuracy: {me['accuracy']:.4f}** (95% CI [{me['acc_ci_low']:.4f}, {me['acc_ci_high']:.4f}])
- AUC: {me['auc']:.4f} · Balanced accuracy: {me['balanced_accuracy']:.4f}
- Up base rate: {me['up_base_rate']:.4f} (always-guess-majority = {me['majority_baseline_acc']:.4f})
- Up precision/recall: {me['up_precision']:.3f} / {me['up_recall']:.3f}
- Down precision/recall: {me['down_precision']:.3f} / {me['down_recall']:.3f}

> CAVEAT: HRC=F is still thin — 35.6% of close-to-close days are exactly flat (bucketed into
> DOWN), which inflates the majority baseline to ~{me['majority_baseline_acc']:.2f}. A genuine
> edge must clear THAT, not 0.50.

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
""" + "\n".join(f"| {r.year} | {r.n} | {r.accuracy} | {r.auc} |" for r in py.itertuples()) + f"""

## Model
- {len(ctx['selected_features'])} features selected (L1, C={ctx['chosen_C']}) of {ctx['candidates']}
  candidates; HRC price block + cross-asset divergence + leak-free overnight steelmakers + iron ore.
- Top features: {', '.join(ctx['selected_features'][:8])}

---
*Research model — not financial advice.*
"""
    path.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, n_over, reason = build_panel(shift=True)
    log(reason)
    log(f"Rows={len(y)} features={X.shape[1]} (overnight+ironore={n_over}) up_rate={y.mean():.4f}")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)
    log(f"acc={metrics['accuracy']:.4f} CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] "
        f"auc={metrics['auc']:.4f} baseline={metrics['majority_baseline_acc']:.4f} OOS={metrics['oos_days']}")
    log("Per-year: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py.itertuples()))

    log("--- No-shift control (same-day session; should NOT beat baseline) ---")
    _, X_ctrl, _, _, _ = build_panel(shift=False)
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
