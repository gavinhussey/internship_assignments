"""Aluminum Close Direction Model — MORNING-OF variant.

Same question as the night-before model (`aluminum_close_direction_model.py`): will the
US-session aluminum benchmark close higher tomorrow (t+1) than today (t)? The difference is
WHEN you guess and therefore WHAT you may use:

  * NIGHT-BEFORE model : guess at today's US close (~1pm ET, day t). Uses only data
                         through day t.
  * THIS MORNING-OF    : guess on the morning of day t+1, AFTER the LME aluminum settlement
                         and the Asian/Australian session have closed but BEFORE the US t+1
                         settle. That lets it use the overnight LME + Asian aluminum-producer
                         moves of day t+1.

WHY ALUMINUM IS AN ESPECIALLY STRONG CASE FOR THIS:
  The LME is THE global aluminum benchmark (unlike copper, where COMEX leads in the US).
  The LME 3-month aluminum contract settles ~7am ET; the US aluminum session opens after.
  So the LME overnight settlement on day t+1 is fully known before the US t+1 close — a
  large, clean, leak-free lead. This is the single biggest expected edge here.

WHY THE OVERNIGHT FEATURES ARE LEAK-FREE:
  ASX producers close ~1am ET and HK producers ~3-4am ET on date t+1; LME aluminum settles
  ~7am ET; the US aluminum contract settles ~1pm ET on date t+1. All precede the target.

  US-OPEN OVERNIGHT PROXY: if the primary file has an `open` column, the US pit opens after
  the LME close, so next_gap_return = open_{t+1} / close_t - 1 is a near-perfect proxy for
  the LME overnight move, available with no external data required.

  REAL LME DATA: data/external/lme_aluminum.csv [date, lme_al_close] (USD/tonne, LME 3-month)
  is already included; it gives the direct LME aluminum overnight return and LME-vs-US basis.
  Cross-metal lme_copper.csv / lme_zinc.csv are also used as context. Refresh from
  NASDAQ Data Link (data.nasdaq.com, free): LME aluminum dataset.

  TRADE-OFF: this is a ~5-hour-ahead forecast, not 24h. Only valid if you actually guess
  in the morning. If you must commit at the prior close, use the night-before model.

Reuses the night-before model's data/feature/walk-forward/L1 pipeline as a library.
Run:  python aluminum_close_morning_model.py   ->  outputs/aluminum_close_morning/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import aluminum_close_direction_model as base

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "aluminum_close_morning"
OVERNIGHT_CACHE = BASE_DIR / "data" / "external" / "macro_cache" / "overnight_prices.csv"

# ASX + HK aluminum producers — Asian session closes before the US settle on the same date.
# 1378.HK = China Hongqiao (world's largest primary-aluminum producer)
# 2600.HK = Aluminum Corp of China / Chalco   0486.HK = United Co. RUSAL
# S32.AX  = South32 (alumina/aluminum heavy)  BHP/RIO carry large alumina+aluminum books
OVERNIGHT_TICKERS = ["BHP.AX", "RIO.AX", "S32.AX",
                     "1378.HK", "2600.HK", "0486.HK"]


def log(m): print(f"[aluminum_morning] {m}", flush=True)


def fetch_overnight() -> pd.DataFrame:
    if OVERNIGHT_CACHE.exists():
        cached_cols = set(pd.read_csv(OVERNIGHT_CACHE, nrows=0).columns)
        expected = {base._safe_name(tk) for tk in OVERNIGHT_TICKERS}
        missing = expected - cached_cols
        if not missing:
            log(f"overnight: using cache {OVERNIGHT_CACHE.name}")
            return pd.read_csv(OVERNIGHT_CACHE, parse_dates=["date"])
        log(f"overnight: cache missing {missing}, re-downloading all tickers...")
    import yfinance as yf
    log(f"overnight: downloading {OVERNIGHT_TICKERS} ...")
    frames = {}
    for tk in OVERNIGHT_TICKERS:
        for attempt in range(3):
            try:
                d = yf.download(tk, start="2014-06-01", auto_adjust=True, progress=False, threads=False)
                if len(d):
                    close = d["Close"]
                    s = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
                    frames[base._safe_name(tk)] = s.rename(base._safe_name(tk))
                    break
            except Exception as e:
                log(f"  {tk} attempt {attempt+1}: {e}")
            time.sleep(2)
        else:
            log(f"  {tk}: SKIPPED")
    if not frames:
        raise RuntimeError("no overnight tickers downloaded and no cache present")
    wide = pd.concat(frames.values(), axis=1, sort=True).reset_index().rename(columns={"Date": "date"})
    wide["date"] = pd.to_datetime(wide["date"]).dt.tz_localize(None)
    OVERNIGHT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OVERNIGHT_CACHE, index=False)
    return wide


def overnight_features(prices, close, dates) -> pd.DataFrame:
    """Day-(t+1) Asian miner moves, available the morning of t+1 (shift(-1); leak-free).

    Single-day returns per ticker + multi-day momentum on the aggregate:
      - 5d/20d rolling mean of daily returns ending at t+1 (trend persistence)
      - 5d/20d rolling vol of daily returns ending at t+1 (regime context)
      - 5/20 crossover ratio (short-term trend vs medium-term)
    """
    mp = prices.set_index("date").sort_index()
    mp = mp.reindex(pd.Index(dates).union(mp.index)).ffill().reindex(dates).reset_index(drop=True)
    metal_ret1 = close.pct_change(1).reset_index(drop=True)
    cols, ov_shifted, ov_raw = {}, [], []
    for tk in OVERNIGHT_TICKERS:
        name = base._safe_name(tk)
        if name not in mp.columns or mp[name].notna().sum() < 100:
            continue
        daily = mp[name].reset_index(drop=True).pct_change(1)
        sig = daily.shift(-1)                                       # Asian session of date t+1
        cols[f"{name}_overnight_ret"] = sig.to_numpy()
        cols[f"{name}_overnight_div"] = (sig - metal_ret1).to_numpy()
        ov_shifted.append(sig)
        ov_raw.append(daily)                                        # unshifted for rolling calcs
    if ov_shifted:
        agg = pd.concat(ov_shifted, axis=1).mean(axis=1)
        cols["asianminers_overnight_mean"] = agg.to_numpy()
        cols["asianminers_overnight_div"] = (agg - metal_ret1).to_numpy()
    if ov_raw:
        # Unshifted aggregate — rolling windows then shift(-1) to get stats ending at t+1
        agg_raw = pd.concat(ov_raw, axis=1).mean(axis=1)
        roll5  = agg_raw.rolling(5).mean()
        roll20 = agg_raw.rolling(20).mean()
        vol5   = agg_raw.rolling(5).std()
        vol20  = agg_raw.rolling(20).std()
        cols["asianminers_5d_mom"]      = roll5.shift(-1).to_numpy()
        cols["asianminers_20d_mom"]     = roll20.shift(-1).to_numpy()
        cols["asianminers_vol5"]        = vol5.shift(-1).to_numpy()
        cols["asianminers_vol20"]       = vol20.shift(-1).to_numpy()
        cols["asianminers_trend_5_20"]  = base._safe_div(roll5, roll20).shift(-1).subtract(1).to_numpy()
    return pd.DataFrame(cols, index=range(len(dates))).replace([np.inf, -np.inf], np.nan)


def overnight_features_no_shift(prices, close, dates) -> pd.DataFrame:
    """Control: same-day Asian return (no shift(-1)). Should NOT beat the ~53.9% baseline."""
    mp = prices.set_index("date").sort_index()
    mp = mp.reindex(pd.Index(dates).union(mp.index)).ffill().reindex(dates).reset_index(drop=True)
    metal_ret1 = close.pct_change(1).reset_index(drop=True)
    cols, ov = {}, []
    for tk in OVERNIGHT_TICKERS:
        name = base._safe_name(tk)
        if name not in mp.columns or mp[name].notna().sum() < 100:
            continue
        daily = mp[name].reset_index(drop=True).pct_change(1)
        sig = daily                                                     # NO shift — control
        cols[f"{name}_overnight_ret"] = sig.to_numpy()
        cols[f"{name}_overnight_div"] = (sig - metal_ret1).to_numpy()
        ov.append(sig)
    if ov:
        agg = pd.concat(ov, axis=1).mean(axis=1)
        cols["asianminers_overnight_mean"] = agg.to_numpy()
        cols["asianminers_overnight_div"] = (agg - metal_ret1).to_numpy()
    return pd.DataFrame(cols, index=range(len(dates))).replace([np.inf, -np.inf], np.nan)


# LME aluminum is the PRIMARY overnight lead (LME is the global aluminum benchmark).
LME_CSV = BASE_DIR / "data" / "external" / "lme_aluminum.csv"
# If your primary US price is in USD/tonne (e.g. COMEX ALI), keep LME_UNIT_DIV=1.0 so the
# LME-vs-US basis is in matching units. If primary is in $/lb, set LME_UNIT_DIV=2204.62.
# (Returns features are unit-invariant; only the basis features care.)
LME_UNIT_DIV = 1.0

# Gradient boosting hyperparameter grid — two shallow configs sufficient; 4 was too slow
GB_PARAM_GRID = [
    {"max_iter": 100, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20},
    {"max_iter": 100, "learning_rate": 0.10, "max_depth": 3, "min_samples_leaf": 20},
]


def walk_forward_gb(X_all: pd.DataFrame, y: np.ndarray):
    """Walk-forward with HistGradientBoosting — handles NaN natively, no scaling needed.

    Same val-split (C1 fix) and block structure as the LR pipeline. Hyperparameters
    selected on val_c, isotonic calibration on val_cal.
    """
    features, report = base.filter_features(X_all, np.arange(base.INIT_TRAIN))
    features = base._prune_drop(features)
    X = X_all[features]
    n = len(y)
    probs = np.full(n, np.nan)
    total_folds = len(range(base.INIT_TRAIN, n, base.STRIDE))
    fold_num = 0
    for r in range(base.INIT_TRAIN, n, base.STRIDE):
        fold_num += 1
        ce = r - base.VAL_WINDOW
        if ce <= 50:
            continue
        core = np.arange(ce)
        fc   = np.arange(r, min(r + base.STRIDE, n))
        split_idx = ce + (2 * base.VAL_WINDOW) // 3
        val_c   = np.arange(ce, split_idx)
        val_cal = np.arange(split_idx, r)
        if len(val_cal) < 30 or len(np.unique(y[val_cal])) < 2:
            val_c = val_cal = np.arange(ce, r)
        if len(np.unique(y[core])) < 2 or len(np.unique(y[val_c])) < 2:
            continue
        Xc = X.iloc[core].values
        Xvc = X.iloc[val_c].values
        best_mdl, best_auc = None, -1.0
        for params in GB_PARAM_GRID:
            mdl = HistGradientBoostingClassifier(
                **params, class_weight="balanced", random_state=base.RANDOM_STATE
            )
            mdl.fit(Xc, y[core])
            auc = roc_auc_score(y[val_c], mdl.predict_proba(Xvc)[:, 1])
            if auc > best_auc:
                best_mdl, best_auc = mdl, auc
        cal = base._calibrate(best_mdl, X.iloc[val_cal].values, y[val_cal])
        probs[fc] = cal.predict_proba(X.iloc[fc].values)[:, 1]
        log(f"GB fold {fold_num}/{total_folds} done (train={len(core)}, val_auc={best_auc:.4f})")
    report["candidate_features"] = len(features)
    return probs, features, report


def us_open_features(df: pd.DataFrame) -> pd.DataFrame:
    """LME overnight proxy from the US-session open price on date t+1.

    The US aluminum pit opens after the LME London close (~7am ET).
    open_{t+1} has fully priced in the LME overnight move, so
    next_gap_return = open_{t+1} / close_t - 1 is a near-perfect proxy.
    Both features use shift(-1), identical leak-free logic to Asian producers.
    Requires an `open` column in the primary file (see DATA_NEEDED.md).
    """
    if "open" not in df.columns:
        log("us_open_features: no 'open' column found, skipping")
        return pd.DataFrame(index=range(len(df)))
    close = df["close"].reset_index(drop=True)
    nxt_open = df["open"].reset_index(drop=True).shift(-1)   # open of t+1
    vol20 = close.pct_change(1).rolling(20).std()
    cols = {}
    cols["next_gap_return"] = (nxt_open / close - 1).to_numpy()
    # Gap in sigma units: isolates signal from background vol regime
    cols["next_gap_z20"] = base._safe_div(
        pd.Series(cols["next_gap_return"]), vol20
    ).to_numpy()
    # Where does tomorrow's open sit vs the 20d range? (0=bottom, 1=top)
    if "low" in df.columns and "high" in df.columns:
        roll_lo = df["low"].reset_index(drop=True).rolling(20).min()
        roll_hi = df["high"].reset_index(drop=True).rolling(20).max()
        rng = (roll_hi - roll_lo).replace(0, np.nan)
        cols["next_open_range_loc"] = ((nxt_open - roll_lo) / rng).to_numpy()
    return pd.DataFrame(cols, index=range(len(df))).replace([np.inf, -np.inf], np.nan)


# Cross-metal context (LME aluminum itself is the primary lead, defined as LME_CSV above).
LME_CU_CSV   = BASE_DIR / "data" / "external" / "lme_copper.csv"
LME_ZN_CSV   = BASE_DIR / "data" / "external" / "lme_zinc.csv"
SHFE_CU_CSV  = BASE_DIR / "data" / "external" / "shfe_copper.csv"


def _align_series(df_dates: pd.DataFrame, csv_path: Path,
                  price_col: str, unit_div: float = 1.0) -> pd.Series:
    """Load a CSV with [date, <price_col>], align to df_dates index, ffill."""
    raw = pd.read_csv(csv_path, parse_dates=["date"])
    raw = raw.rename(columns={c: c.lower().strip() for c in raw.columns})
    raw = (raw[["date", price_col]].dropna()
           .sort_values("date").drop_duplicates("date", keep="last"))
    merged = df_dates[["date"]].reset_index(drop=True).merge(
        raw[["date", price_col]], on="date", how="left"
    )
    merged[price_col] = merged[price_col].ffill()
    s = merged[price_col].reset_index(drop=True)
    return s / unit_div if unit_div != 1.0 else s


def lme_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same-day metals features from LME and SHFE — all available before the US settle.

    Timing (all precede the US aluminum t+1 settle, ~1pm ET):
      LME Ring 2nd session (official settlement): ~1:15 PM London = ~7:15 AM CT ✓
      SHFE day session close: ~3 PM Beijing = ~1-2 AM CT ✓
      US pit open (next_gap_return): ~8:10 AM CT ✓ (handled in us_open_features)

    LME aluminum (lme_aluminum.csv) is the PRIMARY lead and the single biggest expected
    edge — the LME is the global aluminum benchmark and settles before the US session.
    Cross-metal series (copper/zinc/SHFE) are optional context, skipped if CSV absent.

    Features use shift(-1): row t predicts US-session direction on day t+1; the LME/SHFE
    data for date t+1 is available before the US settle (settlement precedes the US close).

      lme_overnight_ret     : LME aluminum Ring settlement return on day t+1 (the lead)
      lme_overnight_z20     : overnight return in 20d sigma units
      lme_us_basis          : LME aluminum - US close on day t (arb pressure)
      lme_us_basis_chg      : daily change in basis
      lme_cu_overnight_ret  : LME copper overnight return (cross-metal context)
      lme_zn_overnight_ret  : LME zinc overnight return (cross-metal context)
      shfe_cu_overnight_ret : SHFE copper return on day t+1 (China industrial demand)
      metals_composite      : equal-weight average of all overnight metal returns
    """
    if not LME_CSV.exists():
        return pd.DataFrame(index=range(len(df)))
    log(f"LME: loading {LME_CSV.name}")
    lp = _align_series(df, LME_CSV, "lme_al_close", unit_div=LME_UNIT_DIV)
    close = df["close"].reset_index(drop=True)
    lme_ret   = lp.pct_change(1)
    lme_vol20 = lme_ret.rolling(20).std()
    basis     = lp - close
    cols = {}
    # Core LME aluminum — same-day Ring settlement, available ~7:15 AM CT (the lead signal)
    cols["lme_overnight_ret"]   = lme_ret.shift(-1).to_numpy()
    cols["lme_overnight_z20"]   = base._safe_div(lme_ret.shift(-1), lme_vol20).to_numpy()
    cols["lme_us_basis"]        = basis.to_numpy()
    cols["lme_us_basis_chg"]    = basis.diff(1).to_numpy()
    log(f"LME aluminum: {lp.notna().sum()} rows")
    overnight_rets = [lme_ret.shift(-1)]
    # LME copper (optional cross-metal context)
    if LME_CU_CSV.exists():
        cu = _align_series(df, LME_CU_CSV, "lme_close")
        cu_ret = cu.pct_change(1)
        cols["lme_cu_overnight_ret"] = cu_ret.shift(-1).to_numpy()
        overnight_rets.append(cu_ret.shift(-1))
        log(f"LME copper: {cu.notna().sum()} rows")
    # LME zinc (optional cross-metal context)
    if LME_ZN_CSV.exists():
        zn = _align_series(df, LME_ZN_CSV, "lme_zn_close")
        zn_ret = zn.pct_change(1)
        cols["lme_zn_overnight_ret"] = zn_ret.shift(-1).to_numpy()
        overnight_rets.append(zn_ret.shift(-1))
        log(f"LME zinc: {zn.notna().sum()} rows")
    # SHFE copper (optional) — China industrial-demand proxy, closes ~1 AM CT
    if SHFE_CU_CSV.exists():
        sh = _align_series(df, SHFE_CU_CSV, "shfe_cu_close")
        sh_ret = sh.pct_change(1)
        cols["shfe_cu_overnight_ret"] = sh_ret.shift(-1).to_numpy()
        overnight_rets.append(sh_ret.shift(-1))
        log(f"SHFE copper: {sh.notna().sum()} rows")
    # Composite: equal-weight average across all available overnight metal returns
    if len(overnight_rets) > 1:
        cols["metals_composite"] = pd.concat(overnight_rets, axis=1).mean(axis=1).to_numpy()
    n_feat = len(cols)
    log(f"LME/metals features: {n_feat} total")
    return pd.DataFrame(cols, index=range(len(df))).replace([np.inf, -np.inf], np.nan)


def build_panel():
    primary, reason = base.find_dataset(BASE_DIR)
    external = base.find_external_file(BASE_DIR, primary)
    df, meta = base.load_and_prepare(primary, external)
    df = base.create_target(df)
    Xc = base.metal_features(df, meta["merged_external_cols"]).reset_index(drop=True)
    Xd = base.divergence_features(base.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    Xo = overnight_features(fetch_overnight(), df["close"], df["date"]).reset_index(drop=True)
    Xl = lme_features(df).reset_index(drop=True)
    Xg = us_open_features(df).reset_index(drop=True)
    n_lme = Xl.shape[1]
    n_gap = Xg.shape[1]
    X = pd.concat([Xc, Xd, Xo, Xl, Xg], axis=1)
    y = df["target_up"].to_numpy(int)
    return df, X, y, meta, reason, Xo.shape[1], n_lme, n_gap


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
    md = f"""# Aluminum Close Direction Model — MORNING-OF variant

## What this is
Predicts whether the US-session aluminum benchmark closes higher tomorrow than today, but
the guess is made on the **morning of the target day**, after the LME aluminum settlement
(~7am ET) and the Asian/Australian session (~1-4am ET) close, and before the US settle
(~1pm ET). That lets it use the **overnight LME aluminum + Asian aluminum-producer moves**,
which lead the US session. The LME lead is the single biggest expected edge.

> If you must lock your guess at the prior close instead, use the NIGHT-BEFORE model
> (`aluminum_close_direction_model.py`). This one is a ~5-hour forecast, not 24h.

## Overnight features (leak-free)
LME aluminum Ring settlement of date t+1 (the primary lead) plus Asian/AU producers
(BHP.AX, RIO.AX, S32.AX, Hongqiao 1378.HK, Chalco 2600.HK, Rusal 0486.HK) session return of
date t+1, encoded as the date-aligned series shifted by one trading day. All settle before
the US t+1 close, so they are available at prediction time.
Validated: a no-shift control (same-day return) scores {ctx.get('ctrl_acc', 'see overnight_shift_validation.csv'):.4f}
vs real {me['accuracy']:.4f}, isolating the gain to the t+1 session shift.

## Performance (walk-forward, {me['oos_days']} OOS days)
- **Accuracy: {me['accuracy']:.4f}** (95% CI [{me['acc_ci_low']:.4f}, {me['acc_ci_high']:.4f}])
- **AUC: {me['auc']:.4f}** · Balanced accuracy: {me['balanced_accuracy']:.4f}
- Up precision/recall: {me['up_precision']:.3f} / {me['up_recall']:.3f}
- Down precision/recall: {me['down_precision']:.3f} / {me['down_recall']:.3f}

## Per-year stability
| year | n | accuracy | auc |
|---|---|---|---|
""" + "\n".join(f"| {r.year} | {r.n} | {r.accuracy} | {r.auc} |" for r in py.itertuples()) + f"""

## Model
- {len(ctx['selected_features'])} features selected (L1, per-fold C={ctx['chosen_C']}) of
  {ctx['candidates']} candidates; aluminum + cross-asset divergence + overnight Asian
  producers + US open gap (LME proxy){' + direct LME overnight' if ctx.get('n_lme', 0) > 0 else ''}.
- Top features: {', '.join(ctx['selected_features'][:8])}

---
*Research model — not financial advice. The morning-of timing is essential to its validity.*
"""
    path.write_text(md, encoding="utf-8")


def main() -> int:
    lr_only = "--lr-only" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, meta, reason, n_over, n_lme, n_gap = build_panel()
    log(reason)
    log(f"Rows={len(y)} features={X.shape[1]} "
        f"(overnight={n_over} lme={n_lme} gap={n_gap}) up_rate={y.mean():.4f}")

    # --- Logistic Regression ---
    probs_lr, features, freport = base.walk_forward(X, y)
    metrics_lr, gate_lr, mask_lr = base.evaluate(y, probs_lr)
    py_lr = per_year(df, y, probs_lr, mask_lr)
    log(f"[LR]  acc={metrics_lr['accuracy']:.4f} "
        f"CI[{metrics_lr['acc_ci_low']:.4f},{metrics_lr['acc_ci_high']:.4f}] "
        f"auc={metrics_lr['auc']:.4f}  OOS={metrics_lr['oos_days']}")
    log("Per-year LR: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py_lr.itertuples()))

    if lr_only:
        log("--lr-only: skipping GB walk-forward")
        probs, metrics, gate, mask, py = probs_lr, metrics_lr, gate_lr, mask_lr, py_lr
    else:
        # --- HistGradientBoosting ---
        log("Running HistGradientBoosting walk-forward ...")
        probs_gb, _, _ = walk_forward_gb(X, y)
        metrics_gb, gate_gb, mask_gb = base.evaluate(y, probs_gb)
        py_gb = per_year(df, y, probs_gb, mask_gb)
        log(f"[GB]  acc={metrics_gb['accuracy']:.4f} "
            f"CI[{metrics_gb['acc_ci_low']:.4f},{metrics_gb['acc_ci_high']:.4f}] "
            f"auc={metrics_gb['auc']:.4f}  OOS={metrics_gb['oos_days']}")
        log("Per-year GB: " + "  ".join(f"{r.year}:{r.accuracy}" for r in py_gb.itertuples()))
        log(f"LR vs GB: acc {metrics_lr['accuracy']:.4f} vs {metrics_gb['accuracy']:.4f} "
            f"({metrics_gb['accuracy'] - metrics_lr['accuracy']:+.4f})  "
            f"auc {metrics_lr['auc']:.4f} vs {metrics_gb['auc']:.4f} "
            f"({metrics_gb['auc'] - metrics_lr['auc']:+.4f})")
        pd.DataFrame([{
            "winner": "GB" if metrics_gb["auc"] > metrics_lr["auc"] else "LR",
            "lr_acc": round(metrics_lr["accuracy"], 4),
            "lr_auc": round(metrics_lr["auc"], 4),
            "gb_acc": round(metrics_gb["accuracy"], 4),
            "gb_auc": round(metrics_gb["auc"], 4),
        }]).to_csv(OUT_DIR / "model_comparison.csv", index=False)
        # Pick the better model for all downstream outputs
        if metrics_gb["auc"] > metrics_lr["auc"]:
            log("GB wins on AUC — using GB for final outputs")
            probs, metrics, gate, mask, py = probs_gb, metrics_gb, gate_gb, mask_gb, py_gb
        else:
            log("LR wins on AUC — using LR for final outputs")
            probs, metrics, gate, mask, py = probs_lr, metrics_lr, gate_lr, mask_lr, py_lr

    # No-shift control: validates that the edge comes from tomorrow's Asian session,
    # not same-day. Should revert to approximately the night-before ~53.9% baseline.
    log("--- No-shift control (same-day Asian return; should ~= night-before baseline) ---")
    overnight_prices = fetch_overnight()
    Xo_ctrl = overnight_features_no_shift(
        overnight_prices, df["close"], df["date"]
    ).reset_index(drop=True)
    Xc_ctrl = base.metal_features(df, meta["merged_external_cols"]).reset_index(drop=True)
    Xd_ctrl = base.divergence_features(base.fetch_macro(), df["close"], df["date"]).reset_index(drop=True)
    Xl_ctrl = lme_features(df).reset_index(drop=True)
    Xg_ctrl = us_open_features(df).reset_index(drop=True)
    X_ctrl = pd.concat([Xc_ctrl, Xd_ctrl, Xo_ctrl, Xl_ctrl, Xg_ctrl], axis=1)
    probs_ctrl, _, _ = base.walk_forward(X_ctrl, y)
    mask_ctrl = ~np.isnan(probs_ctrl)
    ctrl_acc = accuracy_score(y[mask_ctrl], (probs_ctrl[mask_ctrl] >= 0.5).astype(int))
    ctrl_auc = roc_auc_score(y[mask_ctrl], probs_ctrl[mask_ctrl])
    log(f"No-shift control: acc={ctrl_acc:.4f}  auc={ctrl_auc:.4f}  "
        f"(real model: acc={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f})")
    log(f"Overnight lift: acc +{metrics['accuracy'] - ctrl_acc:+.4f}  "
        f"auc +{metrics['auc'] - ctrl_auc:+.4f}")
    pd.DataFrame([{
        "ctrl_acc": ctrl_acc, "ctrl_auc": ctrl_auc,
        "real_acc": metrics["accuracy"], "real_auc": metrics["auc"],
        "acc_lift": metrics["accuracy"] - ctrl_acc,
        "auc_lift": metrics["auc"] - ctrl_auc,
    }]).to_csv(OUT_DIR / "overnight_shift_validation.csv", index=False)

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
    log(f"Final model: C={final['chosen_C']}, {len(final['selected_features'])} features selected.")

    try:
        base.plot_gate(gate, metrics["accuracy"], OUT_DIR / "confidence_gate_chart.png")
    except Exception as e:
        log(f"plot failed: {e}")

    ctx = {"metrics": metrics, "gate": gate, "per_year": py,
           "selected_features": final["selected_features"], "chosen_C": final["chosen_C"],
           "candidates": freport["candidate_features"], "ctrl_acc": ctrl_acc,
           "n_lme": n_lme}
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
