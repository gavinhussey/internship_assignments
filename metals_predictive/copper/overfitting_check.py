"""Overfitting / forward-test diagnostics for the pruned L1 copper model.

We made method choices (L1, the drop-list, the C-grid) while watching the same
2019-2026 walk-forward, so the 53.9% could be partly fit to that window. Three checks:

  1. TIME STABILITY   : per-year + first/second-half + most-recent-12mo OOS accuracy/AUC.
                        A real edge is reasonably consistent; an overfit one is concentrated.
  2. CONFIG ROBUSTNESS: re-run with different INIT_TRAIN / STRIDE / seed. Overfit results
                        swing wildly when you perturb the protocol.
  3. PERMUTATION NULL : shuffle the labels and re-run the FULL pipeline K times. If the
                        method is sound it collapses to ~50%; if it still "predicts",
                        there is leakage. Real accuracy should sit far above the null.

Reuses the frozen pruned pipeline from copper_close_direction_model.py.
Run:  python overfitting_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

import copper_close_direction_model as cc

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "copper_close_direction"


def log(m): print(f"[overfit] {m}", flush=True)


def build():
    primary, _ = cc.find_dataset(BASE)
    external = cc.find_external_file(BASE, primary)
    df, meta = cc.load_and_prepare(primary, external)
    df = cc.create_target(df)
    Xc = cc.copper_features(df, meta["merged_external_cols"]).reset_index(drop=True)
    Xd = cc.divergence_features(cc.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    X = pd.concat([Xc, Xd], axis=1)
    y = df["target_up"].to_numpy(int)
    return df, X, y


def _l1(C, seed):
    return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C,
                              class_weight="balanced", max_iter=5000, random_state=seed)


def walk_forward(X_all, y, init_train=1000, val_window=252, stride=63,
                 cgrid=(0.05, 0.1, 0.25, 0.5), seed=42, calibrate=True):
    feats, _ = cc.filter_features(X_all, np.arange(init_train))
    feats = cc._prune_drop(feats)
    X = X_all[feats]
    n = len(y)
    probs = np.full(n, np.nan)
    for r in range(init_train, n, stride):
        ce = r - val_window
        if ce <= 50:
            continue
        core = np.arange(ce)
        fc = np.arange(r, min(r + stride, n))
        split_idx = ce + (2 * val_window) // 3
        val_c   = np.arange(ce, split_idx)
        val_cal = np.arange(split_idx, r)
        if len(val_cal) < 30 or len(np.unique(y[val_cal])) < 2:
            val_c = val_cal = np.arange(ce, r)
        if len(np.unique(y[core])) < 2 or len(np.unique(y[val_c])) < 2:
            continue
        imp = SimpleImputer(strategy="median").fit(X.iloc[core])
        Xi = pd.DataFrame(imp.transform(X), columns=feats, index=X.index)
        sc = RobustScaler().fit(Xi.iloc[core])
        Xs = sc.transform(Xi)
        best, best_auc = None, -1
        for C in cgrid:
            m = _l1(C, seed).fit(Xs[core], y[core])
            a = roc_auc_score(y[val_c], m.predict_proba(Xs[val_c])[:, 1])
            if a > best_auc:
                best, best_auc = m, a
        if calibrate:
            cal = cc._calibrate(best, Xs[val_cal], y[val_cal])
            probs[fc] = cal.predict_proba(Xs[fc])[:, 1]
        else:
            probs[fc] = best.predict_proba(Xs[fc])[:, 1]
    return probs


def acc_auc(y, p, sel):
    ys, ps = y[sel], p[sel]
    if len(ys) < 10 or len(np.unique(ys)) < 2:
        return len(ys), np.nan, np.nan
    return len(ys), accuracy_score(ys, (ps >= 0.5).astype(int)), roc_auc_score(ys, ps)


def main():
    df, X, y = build()
    log("Running frozen pruned pipeline (baseline config)...")
    p = walk_forward(X, y)
    mask = ~np.isnan(p)
    years = pd.to_datetime(df["date"]).dt.year.to_numpy()

    # ---- 1. time stability ----
    log("=" * 76)
    log("1) TIME STABILITY (per-year OOS):")
    rows = []
    for yr in sorted(np.unique(years[mask])):
        sel = mask & (years == yr)
        n, a, au = acc_auc(y, p, sel)
        rows.append({"period": str(yr), "n": n, "accuracy": a, "auc": au})
        log(f"   {yr}:  n={n:>4}  acc={a:.4f}  auc={au:.4f}")
    idx = np.where(mask)[0]
    half = idx[len(idx) // 2]
    for label, sel in [("first_half", mask & (np.arange(len(y)) < half)),
                       ("second_half", mask & (np.arange(len(y)) >= half))]:
        n, a, au = acc_auc(y, p, sel)
        rows.append({"period": label, "n": n, "accuracy": a, "auc": au})
        log(f"   {label}: n={n:>4}  acc={a:.4f}  auc={au:.4f}")
    # most-recent ~12 months = the true forward slice
    recent = mask & (pd.to_datetime(df["date"]).to_numpy() >= (df["date"].max() - pd.Timedelta(days=365)))
    n, a, au = acc_auc(y, p, recent)
    rows.append({"period": "recent_12mo", "n": n, "accuracy": a, "auc": au})
    log(f"   recent_12mo (forward slice): n={n}  acc={a:.4f}  auc={au:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "overfit_time_stability.csv", index=False)

    # ---- 2. config robustness ----
    log("-" * 76)
    log("2) CONFIG ROBUSTNESS (perturb protocol; overfit => big swings):")
    cfgs = [dict(init_train=1000, stride=63, seed=42),
            dict(init_train=750, stride=63, seed=42),
            dict(init_train=1250, stride=63, seed=42),
            dict(init_train=1000, stride=21, seed=42),
            dict(init_train=1000, stride=126, seed=42),
            dict(init_train=1000, stride=63, seed=7),
            dict(init_train=1000, stride=63, seed=123)]
    crows = []
    for c in cfgs:
        pp = walk_forward(X, y, **c)
        mm = ~np.isnan(pp)
        n, a, au = acc_auc(y, pp, mm)
        crows.append({**c, "n": n, "accuracy": round(a, 4), "auc": round(au, 4)})
        log(f"   {c}  ->  n={n} acc={a:.4f} auc={au:.4f}")
    cdf = pd.DataFrame(crows)
    cdf.to_csv(OUT / "overfit_config_robustness.csv", index=False)
    log(f"   accuracy spread: {cdf['accuracy'].min():.4f} - {cdf['accuracy'].max():.4f} "
        f"(std {cdf['accuracy'].std():.4f})")

    # ---- 3. permutation null (full pipeline — same as main model) ----
    log("-" * 76)
    K = 20
    log(f"3) PERMUTATION NULL TEST ({K} label shuffles, full pipeline):")
    real = cc.walk_forward(X, y)
    rmask = ~np.isnan(real[0])
    real_acc = accuracy_score(y[rmask], (real[0][rmask] >= 0.5).astype(int))
    rng = np.random.default_rng(42)
    null_accs = []
    for k in range(K):
        ys = y.copy()
        rng.shuffle(ys)
        pn, _, _ = cc.walk_forward(X, ys)
        m2 = ~np.isnan(pn)
        null_accs.append(accuracy_score(ys[m2], (pn[m2] >= 0.5).astype(int)))
        log(f"   null {k+1:02d}/{K}: acc={null_accs[-1]:.4f}")
    null_accs = np.array(null_accs)
    p_value = float((null_accs >= real_acc).mean())
    log(f"   real acc (full pipeline): {real_acc:.4f}")
    log(f"   null acc mean+/-std    : {null_accs.mean():.4f} +/- {null_accs.std():.4f}")
    log(f"   null acc max           : {null_accs.max():.4f}")
    log(f"   permutation p-value    : {p_value:.3f}  (fraction of nulls >= real)")
    pd.DataFrame({"null_accuracy": null_accs}).to_csv(OUT / "overfit_permutation_null.csv", index=False)

    log("=" * 76)
    verdict = ("LEAKAGE SUSPECT" if p_value > 0.10 else
               "passes null test")
    log(f"VERDICT: real {real_acc:.3f} vs null {null_accs.mean():.3f} (p={p_value:.3f}) -> {verdict}")
    log(f"Artifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"FATAL {type(e).__name__}: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)
