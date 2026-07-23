"""Iron Ore (SGX TIO=F) Direction Model — MORNING-OF.

Same target as iron_ore_direction_model.py: will SGX 62% Fe iron ore close higher
tomorrow (t+1) than today (t)? The difference is WHEN you guess and therefore WHAT
you may use as features.

  NIGHT-BEFORE   : decide at 4pm ET day t. Uses only data through day t. ~51.9%.
  THIS MODEL     : decide at ~2:30am ET day t+1, AFTER the SHFE and DCE day sessions
                   close (3pm Beijing = 2am ET), but BEFORE the SGX TIO=F final
                   settlement (~5-6am ET Singapore time = 4-5am ET). That window lets
                   it use the China ferrous complex of day t+1 as overnight leads.

WHY THE OVERNIGHT FEATURES ARE LEAK-FREE:
  DCE iron ore   : closes 3pm Beijing = ~2am ET on day t+1.   ✓ (pre-TIO=F settle)
  SHFE rebar/HRC : closes 3pm Beijing = ~2am ET on day t+1.   ✓
  SGX iron ore (bloomberg CSV): the Bloomberg close date for SGX iron ore may be
  contemporaneous with TIO=F on yfinance (both Singapore-based). If the shift control
  confirms no spurious lift, SGX-from-Bloomberg is also valid; otherwise exclude it.

VALIDATION:
  A no-shift control (same-day China ferrous, no shift(-1)) is run alongside the
  real model. If the no-shift control matches or beats the real model, the overnight
  lead is spurious and the model should not be deployed.

Requires bloombergSteel.py to have been run first to populate:
  data/external/shfe_rebar.csv
  data/external/shfe_hrc.csv
  data/external/dce_iron_ore.csv
  data/external/dce_coking_coal.csv
  data/external/sgx_iron_ore.csv   (include with caution — see timing note above)

Run:  python iron_ore_morning_model.py  ->  outputs/iron_ore_morning/
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
import iron_ore_direction_model as night

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "outputs" / "iron_ore_morning"


def log(m): print(f"[iron_ore_morning] {m}", flush=True)


def build_panel(shift: bool = True):
    """Night-before features (day t) + China ferrous overnight leads (day t+1, shift=-1).

    shift=True   → real model: China ferrous data for day t+1 (shift_d=-1), leaked-free.
    shift=False  → control: same-day China ferrous (no shift), tests whether "overnight"
                   lead is real or contemporaneous noise.
    """
    df, ext_cols = night.load_iron_ore()
    df = base.create_target(df)
    Xc = base.steel_features(df, ext_cols).reset_index(drop=True)
    Xd = base.divergence_features(base.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    shift_d = -1 if shift else 0
    # exclude_sgx=True because SGX Bloomberg and TIO=F yfinance are the same contract;
    # shift(-1) on SGX would leak the same-day settle into the feature set.
    Xf = night.china_ferrous_features(df, shift_d=shift_d, exclude_sgx=shift).reset_index(drop=True)
    X = pd.concat([Xc, Xd, Xf], axis=1)
    y = df["target_up"].to_numpy(int)
    n_china = Xf.shape[1]
    return df, X, y, n_china


def per_year(df, y, probs, mask):
    return night.per_year(df, y, probs, mask)


def write_summary(ctx, path):
    me, g, py = ctx["metrics"], ctx["gate"], ctx["per_year"]
    best_gate = g.iloc[g["accuracy"].idxmax()]
    ctrl_section = ""
    if "ctrl_acc" in ctx:
        lift_acc = me["accuracy"] - ctx["ctrl_acc"]
        lift_auc = me["auc"] - ctx["ctrl_auc"]
        ctrl_section = f"""
## No-shift control (the honesty check)
Real model acc {me['accuracy']:.4f} / AUC {me['auc']:.4f}.
No-shift control acc {ctx['ctrl_acc']:.4f} / AUC {ctx['ctrl_auc']:.4f}.
**Overnight shift lift: {lift_acc:+.4f} acc / {lift_auc:+.4f} AUC.**
Only a **positive, meaningful lift** makes the morning timing worthwhile.
If lift ≤ 0, the China ferrous "overnight" data is contemporaneous with TIO=F — do not deploy.
"""
    md = f"""# Iron Ore (SGX TIO=F) Direction Model — MORNING-OF

## What this is
Predicts whether SGX 62% Fe iron ore closes higher tomorrow, guessed after the China ferrous
complex (SHFE rebar/HRC, DCE iron ore and coking coal) closes at ~3pm Beijing / 2am ET on
day t+1 — before the TIO=F final settlement at ~5-6am ET. See iron_ore_direction_model.py
for the night-before baseline (~51.9%).
{ctrl_section}
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
  candidates; iron ore price block + cross-asset divergence + China ferrous overnight leads.
- Top features: {', '.join(ctx['selected_features'][:8])}

## China ferrous features (Bloomberg required)
Populated by bloombergSteel.py. If overnight shift lift ≤ 0, the features are not leading
TIO=F and this model should be retracted in favour of iron_ore_direction_model.py.

---
*Research model — not financial advice. Valid only if overnight shift lift is positive.*
"""
    path.write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, X, y, n_china = build_panel(shift=True)
    log(f"Rows={len(y)} features={X.shape[1]} (china_ferrous={n_china}) "
        f"up_rate={y.mean():.4f}  {df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}")

    if n_china == 0:
        log("WARNING: no China ferrous Bloomberg features found. "
            "Run bloombergSteel.py first. Proceeding with night-before features only.")

    probs, features, freport = base.walk_forward(X, y)
    metrics, gate, mask = base.evaluate(y, probs)
    py = per_year(df, y, probs, mask)
    log(f"acc={metrics['accuracy']:.4f} CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] "
        f"auc={metrics['auc']:.4f} baseline={metrics['majority_baseline_acc']:.4f} OOS={metrics['oos_days']}")
    log("Per-year: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py.itertuples()))

    # No-shift control — validates whether the China overnight lead is real or contemporaneous
    log("--- No-shift control (same-day China ferrous; should collapse to night-before baseline) ---")
    _, X_ctrl, _, _ = build_panel(shift=False)
    probs_ctrl, _, _ = base.walk_forward(X_ctrl, y)
    mctrl = ~np.isnan(probs_ctrl)
    ctrl_acc = accuracy_score(y[mctrl], (probs_ctrl[mctrl] >= 0.5).astype(int))
    ctrl_auc = roc_auc_score(y[mctrl], probs_ctrl[mctrl])
    log(f"No-shift control: acc={ctrl_acc:.4f} auc={ctrl_auc:.4f}  "
        f"(real: acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f})")
    lift_acc = metrics["accuracy"] - ctrl_acc
    lift_auc = metrics["auc"] - ctrl_auc
    log(f"Overnight shift lift: acc {lift_acc:+.4f}  auc {lift_auc:+.4f}")
    if lift_acc <= 0:
        log("*** NEGATIVE LIFT — China ferrous data does not lead TIO=F. "
            "Model should NOT be deployed. Investigate timing alignment. ***")
    pd.DataFrame([{
        "ctrl_acc": ctrl_acc, "ctrl_auc": ctrl_auc,
        "real_acc": metrics["accuracy"], "real_auc": metrics["auc"],
        "acc_lift": lift_acc, "auc_lift": lift_auc,
        "n_china_features": n_china,
        "note": ("VALID: positive overnight lift" if lift_acc > 0
                 else "INVALID: negative lift — do not deploy"),
    }]).to_csv(OUT_DIR / "overnight_shift_validation.csv", index=False)

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

    ctx = {
        "metrics": metrics, "gate": gate, "per_year": py,
        "selected_features": final["selected_features"], "chosen_C": final["chosen_C"],
        "candidates": freport["candidate_features"],
        "ctrl_acc": ctrl_acc, "ctrl_auc": ctrl_auc,
    }
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
