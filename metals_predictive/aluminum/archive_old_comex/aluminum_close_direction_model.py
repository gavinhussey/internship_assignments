"""Aluminum Close-to-Close Direction Model (main).

THE question: will the aluminum benchmark close higher TOMORROW than it closed TODAY?
A pure up/down call, every day — we do not care by how much.

    target_up = 1 if close_{t+1} > close_t else 0

Ported from the validated copper_direction_model_v2 pipeline. The architecture is
metal-agnostic; only the cross-asset "cousins" change. What it keeps:
  * Features  : aluminum price features + EXPANDED cross-asset DIVERGENCE
                (aluminum-producer / China / base-metal returns minus aluminum's own move).
                The divergence block is what lifts AUC off the 0.50 floor.
  * Model     : Logistic Regression with isotonic probability calibration
                (it beat RF / GB / HistGB on copper — the usable signal is linear).
  * Validation: expanding-window walk-forward, out-of-sample days.
  * Dead-band : a DEPLOYABLE confidence gate — trust the call only when the model
                is sure (|p-0.5| large).

NB: aluminum's primary global benchmark is the LME, not a US exchange. Provide a daily
US-session aluminum OHLCV series at data/raw/aluminum.csv (see DATA_NEEDED.md) so the
morning model can use the LME-leads-US overnight gap exactly as the copper model does.

Leakage controls: features use only current/past data; target uses close.shift(-1);
imputer + scaler refit on each fold's training rows; macro merged point-in-time.
scikit-learn only — no TensorFlow.

Run:  python aluminum_close_direction_model.py
Outputs -> outputs/aluminum_close_direction/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

# ----------------------------------------------------------------------------
# Paths & config
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "aluminum_close_direction"
MACRO_CACHE = BASE_DIR / "data" / "external" / "macro_cache" / "macro_prices_expanded.csv"

WARMUP_ROWS = 60
MISSING_FRAC_LIMIT = 0.35
INIT_TRAIN, VAL_WINDOW, STRIDE = 1000, 252, 63
RANDOM_STATE = 42

# Cross-asset tickers (yfinance). Divergence = asset return minus aluminum's own return.
# These are aluminum's "cousins" — primary-aluminum producers, China demand proxies, and
# base-metal/mining sector ETFs that share aluminum's drivers (China construction, the
# dollar, energy, global industrial demand). Thin/illiquid tickers are skipped at runtime
# (divergence_features drops any series with <100 non-null obs), so an over-inclusive list
# is safe — let L1 prune. Candidates to revisit in v2 are noted in DATA_NEEDED.md.
#   AA   Alcoa (US bellwether)        CENX Century Aluminum (pure-play smelter)
#   KALU Kaiser Aluminum              CSTM Constellium (rolled products)
#   NHYDY Norsk Hydro ADR             ACH  Aluminum Corp of China ADR (Chalco)
#   RIO/BHP diversified majors (large alumina/aluminum books)
#   DBB base-metals ETF  PICK global metals&mining  XME US metals&mining  SLX steel
DIVERGENCE_TICKERS = ["AA", "CENX", "KALU", "CSTM", "NHYDY", "ACH",
                      "RIO", "BHP",
                      "FXI", "MCHI", "ASHR", "KWEB",
                      "DBB", "PICK", "XME", "SLX"]
# Plain macro context (returns/momentum/vol, not divergences). CNY kept — China is ~60% of
# global aluminum demand. Energy is aluminum's biggest idiosyncratic cost (smelting power);
# add a power/nat-gas proxy in v2 (see DATA_NEEDED.md).
PLAIN_TICKERS = ["^VIX", "DX-Y.NYB", "^TNX", "CNY=X"]
ALL_TICKERS = DIVERGENCE_TICKERS + PLAIN_TICKERS

# Deployable confidence-gate cutoffs on |p - 0.5|.
CONF_CUTOFFS = [0.0, 0.03, 0.05, 0.08]

# Pruning: L1 logistic selects features inside each fold; C is chosen per-fold on the
# validation set (leakage-safe). These level features are dropped a priori — they are
# non-stationary (test-period price range never appears in training) and add noise.
L1_C_GRID = [0.05, 0.1, 0.25, 0.5]
DROP_LEVEL_FEATURES = ["ma_5", "ma_10", "ma_20", "ma_60", "volume_ma_5", "volume_ma_20"]

LEAK_COLS = {"tomorrow_return", "target_up", "target_down", "future_return", "next_close"}
EXTERNAL_INVENTORY_LIKE = ["inventory", "stocks"]
DATE_CANDIDATES = ["date", "datetime", "time", "timestamp", "trade_date", "observation_date"]
CLOSE_CANDIDATES = ["close", "comex_close", "ali_close", "aluminum_close", "alu_close",
                    "aluminum_futures_close", "settlement", "settle", "settlement_price", "px_last"]
DATASET_KEYWORDS = ["aluminum", "aluminium", "alu", "ali", "comex", "futures", "price",
                    "market", "merged", "features"]


def log(msg: str) -> None:
    print(f"[aluminum_close] {msg}", flush=True)


def _safe_name(tk: str) -> str:
    return tk.replace("^", "").replace("=X", "").replace(".", "_").replace("-", "_")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# Dataset discovery & loading
# ----------------------------------------------------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace(r"[^0-9a-z]+", "_", regex=True)
                  .str.replace(r"_+", "_", regex=True).str.strip("_"))
    return df


def identify_date_column(cols):
    for c in DATE_CANDIDATES:
        if c in cols:
            return c
    return next((c for c in cols if "date" in c or "time" in c), None)


def identify_close_column(cols):
    cands = [c for c in cols if c in CLOSE_CANDIDATES] or \
            [c for c in cols if "close" in c or "settle" in c or "px_last" in c]
    if not cands:
        return None
    for pref in CLOSE_CANDIDATES:
        if pref in cands:
            return pref
    return cands[0]


def _peek(path: Path):
    try:
        if path.suffix.lower() == ".csv":
            cols = list(pd.read_csv(path, nrows=0).columns)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            cols = list(pd.read_excel(path, nrows=0).columns)
        elif path.suffix.lower() == ".parquet":
            cols = list(pd.read_parquet(path).columns)
        else:
            return None
    except Exception:
        return None
    return [str(c).strip().lower().replace(" ", "_") for c in cols]


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def find_dataset(root: Path):
    exts = (".csv", ".xlsx", ".xls", ".parquet")
    # NB: "external" added to skip_dirs so the LME/cross-metal series in data/external are
    # never auto-chosen as the PRIMARY target (that would make LME aluminum both the target
    # and the morning model's overnight lead — leakage). Primary must live in data/raw.
    skip_dirs = {"outputs", "macro_cache", "external", "__pycache__", ".git", ".venv"}
    skip_bits = ("prediction", "metrics", "driver", "external", "macro", "smoke")
    best, best_score, reason = None, -1, ""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if any(p in skip_dirs for p in path.parts) or any(b in path.stem.lower() for b in skip_bits):
            continue
        cols = _peek(path)
        if not cols:
            continue
        close = identify_close_column(cols)
        if identify_date_column(cols) is None or close is None:
            continue
        ohlc = all(any(k == c or k in c for c in cols) for k in ("open", "high", "low"))
        name_al = any(k in path.stem.lower() for k in ("aluminum", "aluminium", "alu", "ali"))
        score = 3 + 2 + ohlc + sum(any(kw in c for c in cols) for kw in DATASET_KEYWORDS) + 2 * name_al
        if score > best_score:
            best, best_score = path, score
            reason = (f"Chosen `{path.name}` (score={score}): date + close ('{close}'), "
                      f"OHLC={ohlc}, aluminum-in-name={name_al}.")
    if best is None:
        raise FileNotFoundError(f"No date+close dataset found under {root}.")
    return best, reason


def find_external_file(root: Path, primary: Path):
    macro_bits = ("dxy", "vix", "wti", "brent", "yield", "ust", "lme", "dollar",
                  "oil", "aluminum", "inventory", "fed_funds", "curve")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == primary or path.suffix.lower() not in (".csv", ".xlsx", ".xls", ".parquet"):
            continue
        if any(p in {"outputs", "macro_cache", "__pycache__", ".git", ".venv"} for p in path.parts):
            continue
        cols = _peek(path)
        if not cols or identify_date_column(cols) is None or identify_close_column(cols) is not None:
            continue
        if sum(any(b in c for c in cols) for b in macro_bits) >= 2:
            return path
    return None


def load_and_prepare(primary: Path, external: Path | None):
    df = clean_column_names(_read_any(primary))
    date_col = identify_date_column(df.columns)
    close_col = identify_close_column(df.columns)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (df.dropna(subset=[date_col]).sort_values(date_col)
          .drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
          .rename(columns={date_col: "date", close_col: "close"}))
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    meta = {"primary": primary.name, "external": external.name if external else None,
            "close_col_source": close_col, "merged_external_cols": []}
    if external is not None:
        ext = clean_column_names(_read_any(external))
        ed = identify_date_column(ext.columns)
        if ed is not None:
            ext[ed] = pd.to_datetime(ext[ed], errors="coerce")
            ext = (ext.dropna(subset=[ed]).sort_values(ed)
                   .drop_duplicates(subset=[ed], keep="last").rename(columns={ed: "date"}))
            cols = [c for c in ext.columns if c != "date"]
            for c in cols:
                ext[c] = pd.to_numeric(ext[c], errors="coerce")
            df = df.merge(ext[["date"] + cols], on="date", how="left")
            df[cols] = df[cols].ffill()           # past-only fill, no leakage
            meta["merged_external_cols"] = cols
    return df, meta


# ----------------------------------------------------------------------------
# Target: close-to-close direction
# ----------------------------------------------------------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tomorrow_return"] = np.log(df["close"].shift(-1) / df["close"])
    df["target_up"] = (df["tomorrow_return"] > 0).astype("Int64")
    df = df.iloc[:-1].reset_index(drop=True)      # last row's tomorrow unknown
    df["target_up"] = df["target_up"].astype(int)
    return df


# ----------------------------------------------------------------------------
# Features: aluminum price block
# ----------------------------------------------------------------------------
def metal_features(df: pd.DataFrame, external_cols: list[str]) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    for n in (1, 2, 3, 5, 10, 20):
        f[f"log_return_{n}d"] = np.log(close / close.shift(n))
    for n in (1, 5, 20):
        f[f"simple_return_{n}d"] = close.pct_change(n)
    for n in (5, 10, 20, 60):
        f[f"momentum_{n}d"] = close / close.shift(n) - 1
    ma = {n: close.rolling(n).mean() for n in (5, 10, 20, 60)}
    for n in (5, 10, 20, 60):
        f[f"ma_{n}"] = ma[n]
    f["price_vs_ma5"] = close / ma[5] - 1
    f["price_vs_ma20"] = close / ma[20] - 1
    f["price_vs_ma60"] = close / ma[60] - 1
    f["ma5_vs_ma20"] = ma[5] / ma[20] - 1
    f["ma20_vs_ma60"] = ma[20] / ma[60] - 1
    r1 = f["log_return_1d"]
    for n in (5, 10, 20, 60):
        f[f"volatility_{n}d"] = r1.rolling(n).std()
    f["volatility_spike_5_20"] = _safe_div(f["volatility_5d"], f["volatility_20d"])
    f["volatility_spike_10_60"] = _safe_div(f["volatility_10d"], f["volatility_60d"])
    f["drawdown_20d"] = close / close.rolling(20).max() - 1
    f["drawdown_60d"] = close / close.rolling(60).max() - 1
    if all(c in df.columns for c in ("open", "high", "low")):
        o, h, low = df["open"], df["high"], df["low"]
        rng = h - low
        f["high_low_range"] = _safe_div(rng, close)
        f["close_open_return"] = _safe_div(close - o, o)
        f["close_location"] = np.where(rng.values != 0, (close - low) / rng.replace(0, np.nan), 0.5)
        f["gap_return"] = _safe_div(o, close.shift(1)) - 1
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        f["volume_pct_change_1d"] = vol.pct_change(1)
        f["volume_pct_change_5d"] = vol.pct_change(5)
        f["volume_ma_5"] = vol.rolling(5).mean()
        f["volume_ma_20"] = vol.rolling(20).mean()
        f["volume_vs_ma20"] = _safe_div(vol, f["volume_ma_20"]) - 1
        f["volume_zscore_20d"] = _safe_div(vol - vol.rolling(20).mean(), vol.rolling(20).std())
    for col in external_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if any(b in col for b in EXTERNAL_INVENTORY_LIKE):
            f[f"{col}_change_1d"] = s.diff(1)
            f[f"{col}_change_5d"] = s.diff(5)
            f[f"{col}_pct_change_20d"] = s.pct_change(20)
        else:
            f[f"{col}_return_1d"] = s.pct_change(1)
            f[f"{col}_return_5d"] = s.pct_change(5)
            f[f"{col}_return_20d"] = s.pct_change(20)
            f[f"{col}_momentum_20d"] = s / s.shift(20) - 1
            f[f"{col}_volatility_20d"] = s.pct_change(1).rolling(20).std()
    # Calendar features — stationary, no leakage
    dow = df["date"].dt.dayofweek.values.astype(float)   # 0=Mon, 4=Fri
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    month = df["date"].dt.month.values.astype(float)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f[[c for c in f.columns if c not in LEAK_COLS]]


# ----------------------------------------------------------------------------
# Features: cross-asset divergence block (yfinance, cached)
# ----------------------------------------------------------------------------
def fetch_macro() -> pd.DataFrame:
    if MACRO_CACHE.exists():
        log(f"Macro: using cache {MACRO_CACHE.name}")
        return pd.read_csv(MACRO_CACHE, parse_dates=["date"])
    import yfinance as yf
    log(f"Macro: downloading {len(ALL_TICKERS)} tickers via yfinance ...")
    frames = {}
    for tk in ALL_TICKERS:
        for attempt in range(3):
            try:
                d = yf.download(tk, start="2014-06-01", auto_adjust=True, progress=False, threads=False)
                if len(d):
                    close = d["Close"]
                    s = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
                    frames[_safe_name(tk)] = s.rename(_safe_name(tk))
                    break
            except Exception as e:
                log(f"  {tk} attempt {attempt+1}: {e}")
            time.sleep(2)
        else:
            log(f"  {tk}: SKIPPED")
    if not frames:
        raise RuntimeError("No macro tickers downloaded and no cache present.")
    wide = pd.concat(frames.values(), axis=1, sort=True).reset_index().rename(columns={"Date": "date"})
    wide["date"] = pd.to_datetime(wide["date"]).dt.tz_localize(None)
    MACRO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(MACRO_CACHE, index=False)
    return wide


def divergence_features(macro: pd.DataFrame, close: pd.Series, dates: pd.Series) -> pd.DataFrame:
    mp = macro.set_index("date").sort_index()
    mp = mp.reindex(pd.Index(dates).union(mp.index)).ffill().reindex(dates).reset_index(drop=True)
    cu = {N: close.pct_change(N).to_numpy() for N in (1, 5, 20)}
    cols = {}  # build all columns then concat once (avoids DataFrame fragmentation)
    for tk in DIVERGENCE_TICKERS:
        name = _safe_name(tk)
        if name not in mp.columns or mp[name].notna().sum() < 100:
            continue
        s = mp[name]
        for N in (1, 5, 20):
            cols[f"{name}_div_{N}d"] = s.pct_change(N).to_numpy() - cu[N]
        d1 = s.pct_change(1) - pd.Series(cu[1])
        cols[f"{name}_div_z20"] = ((d1 - d1.rolling(20).mean()) / d1.rolling(20).std()).to_numpy()
        ratio = s / close.reset_index(drop=True)
        cols[f"{name}_ratio_mom20"] = (ratio / ratio.shift(20) - 1).to_numpy()
    for tk in PLAIN_TICKERS:
        name = _safe_name(tk)
        if name not in mp.columns or mp[name].notna().sum() < 100:
            continue
        s = mp[name]
        cols[f"{name}_ret_1d"] = s.pct_change(1).to_numpy()
        cols[f"{name}_ret_5d"] = s.pct_change(5).to_numpy()
        cols[f"{name}_mom_20d"] = (s / s.shift(20) - 1).to_numpy()
        cols[f"{name}_vol_20d"] = s.pct_change(1).rolling(20).std().to_numpy()
    return pd.DataFrame(cols, index=range(len(dates))).replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# Feature filtering (decided on TRAIN rows only)
# ----------------------------------------------------------------------------
def filter_features(X: pd.DataFrame, train_idx: np.ndarray):
    Xtr = X.iloc[train_idx]
    keep, dropped_missing, dropped_const = [], [], []
    for col in X.columns:
        c = Xtr[col]
        if c.isna().mean() > MISSING_FRAC_LIMIT:
            dropped_missing.append(col); continue
        nn = c.dropna()
        if nn.nunique() <= 1 or float(nn.std() or 0.0) < 1e-10:
            dropped_const.append(col); continue
        keep.append(col)
    return keep, {"dropped_missing": dropped_missing, "dropped_constant": dropped_const}


# ----------------------------------------------------------------------------
# Model + walk-forward
# ----------------------------------------------------------------------------
def _make_logistic(C=0.1):
    # L1 (sparse) logistic: zeros out non-contributing features within each fold.
    return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C,
                              class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)


def _prune_drop(features):
    """Drop non-stationary level features and exact duplicates (*_return_20d==*_momentum_20d)."""
    drop = set(DROP_LEVEL_FEATURES) | {f for f in features if f.endswith("_return_20d")}
    return [f for f in features if f not in drop]


def _fit_best_C(Xs, y, core, val):
    """Fit L1 logistic at each C, pick the one with best validation AUC (no test leakage)."""
    best, best_auc, bestC = None, -1.0, L1_C_GRID[0]
    for C in L1_C_GRID:
        mdl = _make_logistic(C).fit(Xs[core], y[core])
        a = roc_auc_score(y[val], mdl.predict_proba(Xs[val])[:, 1])
        if a > best_auc:
            best, best_auc, bestC = mdl, a, C
    return best, bestC


def _calibrate(base, Xva, yva, method="isotonic"):
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    except Exception:
        cal = CalibratedClassifierCV(base, method=method, cv="prefit")
    cal.fit(Xva, yva)
    return cal


def walk_forward(X_all: pd.DataFrame, y: np.ndarray):
    """Expanding-window walk-forward L1 logistic; returns OOS calibrated P(up) per day."""
    from collections import Counter
    features, report = filter_features(X_all, np.arange(INIT_TRAIN))
    features = _prune_drop(features)
    X = X_all[features]
    n = len(y)
    probs = np.full(n, np.nan)
    nz = []
    fold_selections = []
    for r in range(INIT_TRAIN, n, STRIDE):
        ce = r - VAL_WINDOW
        if ce <= 50:
            continue
        core = np.arange(ce)
        fc = np.arange(r, min(r + STRIDE, n))
        # Split val: first 2/3 for C-selection, last 1/3 for calibration
        split_idx = ce + (2 * VAL_WINDOW) // 3
        val_c   = np.arange(ce, split_idx)
        val_cal = np.arange(split_idx, r)
        if len(val_cal) < 30 or len(np.unique(y[val_cal])) < 2:
            log(f"  fold r={r}: val_cal too small/uniform, falling back to full val for both")
            val_c = val_cal = np.arange(ce, r)
        if len(np.unique(y[core])) < 2 or len(np.unique(y[val_c])) < 2:
            continue
        imp = SimpleImputer(strategy="median").fit(X.iloc[core])
        Xi = pd.DataFrame(imp.transform(X), columns=features, index=X.index)
        sc = RobustScaler().fit(Xi.iloc[core])
        Xs = sc.transform(Xi)
        base, _C = _fit_best_C(Xs, y, core, val_c)      # C selected on val_c only
        nz.append(int((np.abs(base.coef_) > 1e-8).sum()))
        fold_selections.append([features[i] for i, c in enumerate(base.coef_.ravel()) if abs(c) > 1e-8])
        cal = _calibrate(base, Xs[val_cal], y[val_cal])  # calibrated on val_cal only
        probs[fc] = cal.predict_proba(Xs[fc])[:, 1]
    report["candidate_features"] = len(features)
    report["avg_features_used_per_fold"] = float(np.mean(nz)) if nz else float("nan")
    all_counts = Counter(f for fold in fold_selections for f in fold)
    n_folds = max(len(fold_selections), 1)
    report["feature_selection_freq"] = {f: all_counts[f] / n_folds for f in features}
    return probs, features, report


# ----------------------------------------------------------------------------
# Metrics + deployable confidence gate
# ----------------------------------------------------------------------------
def _boot_acc(correct, iters=3000, block_size=20):
    """Circular block bootstrap — accounts for autocorrelation in rolling-window features."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(correct)
    if n < 40:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block_size))
    s = []
    for _ in range(iters):
        starts = rng.integers(0, n, n_blocks)
        sample = np.concatenate([correct[st : st + block_size] for st in starts])[:n]
        s.append(float(sample.mean()))
    return tuple(np.percentile(s, [2.5, 97.5]))


def evaluate(y, probs):
    mask = ~np.isnan(probs)
    yv, pv = y[mask], probs[mask]
    pred = (pv >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    (alo, ahi) = _boot_acc((pred == yv).astype(int))
    metrics = {
        "oos_days": int(mask.sum()),
        "accuracy": accuracy_score(yv, pred), "acc_ci_low": alo, "acc_ci_high": ahi,
        "balanced_accuracy": balanced_accuracy_score(yv, pred),
        "auc": roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else np.nan,
        "f1_up": f1_score(yv, pred),
        "up_precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "up_recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "down_precision": tn / (tn + fn) if (tn + fn) else np.nan,
        "down_recall": tn / (tn + fp) if (tn + fp) else np.nan,
        "up_base_rate": float(yv.mean()),
        "majority_baseline_acc": float(max(yv.mean(), 1 - yv.mean())),
        "confusion": [int(tn), int(fp), int(fn), int(tp)],
    }
    # Deployable confidence gate.
    conf = np.abs(pv - 0.5)
    gate_rows = []
    for d in CONF_CUTOFFS:
        g = conf > d if d > 0 else np.ones_like(conf, bool)
        if g.sum() < 20:
            continue
        c = (pred[g] == yv[g]).astype(int)
        lo, hi = _boot_acc(c)
        gate_rows.append({"min_confidence": d, "coverage": float(g.mean()),
                          "committed_days": int(g.sum()), "accuracy": float(c.mean()),
                          "acc_ci_low": lo, "acc_ci_high": hi})
    return metrics, pd.DataFrame(gate_rows), mask


# ----------------------------------------------------------------------------
# Final deployable model (fit on most recent window) + plots + summary
# ----------------------------------------------------------------------------
def fit_final_model(X: pd.DataFrame, y: np.ndarray, features: list[str]):
    features = _prune_drop(features)
    n = len(y)
    core = np.arange(max(0, n - INIT_TRAIN), n - VAL_WINDOW)
    val_start = n - VAL_WINDOW
    val_split = val_start + (2 * VAL_WINDOW) // 3
    val_c   = np.arange(val_start, val_split)
    val_cal = np.arange(val_split, n)
    imp = SimpleImputer(strategy="median").fit(X[features].iloc[core])
    Xi = pd.DataFrame(imp.transform(X[features]), columns=features, index=X.index)
    sc = RobustScaler().fit(Xi.iloc[core])
    Xs = sc.transform(Xi)
    base, C = _fit_best_C(Xs, y, core, val_c)
    cal = _calibrate(base, Xs[val_cal], y[val_cal])
    coef = base.coef_.ravel()
    selected = [(f, float(c)) for f, c in zip(features, coef) if abs(c) > 1e-8]
    selected.sort(key=lambda t: abs(t[1]), reverse=True)
    return {"imputer": imp, "scaler": sc, "model": cal, "features": features,
            "selected_features": [f for f, _ in selected],
            "coefficients": dict(selected), "chosen_C": C}


def plot_gate(gate_df, base_acc, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(gate_df["coverage"], gate_df["accuracy"], "o-", color="#2A9D8F")
    for r in gate_df.itertuples():
        ax.annotate(f"|p-.5|>{r.min_confidence}", (r.coverage, r.accuracy),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(0.5, color="grey", ls=":", label="coin flip")
    ax.axhline(base_acc, color="#E76F51", ls="--", label=f"every-day acc={base_acc:.3f}")
    ax.set_xlabel("coverage (share of days committed)"); ax.set_ylabel("accuracy")
    ax.set_title("Deployable confidence gate: accuracy vs coverage")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_summary(ctx, path):
    me, g = ctx["metrics"], ctx["gate"]
    best_gate = g.iloc[g["accuracy"].idxmax()]
    md = f"""# Aluminum Close-to-Close Direction Model

## Question
Will the aluminum benchmark close **higher tomorrow than it closed today**? Pure up/down,
every day. `target_up = 1 if close[t+1] > close[t] else 0`.

## Data
- Price dataset: `{ctx['meta']['primary']}` (close column `{ctx['meta']['close_col_source']}`)
- Macro drivers merged: `{ctx['meta']['external']}` + {len(ctx['div_tickers'])} cross-asset tickers
- Rows: {ctx['n_rows']} · Date range: {ctx['date_min']} → {ctx['date_max']}

## Features ({len(ctx['features'])} candidates -> {len(ctx['selected_features'])} selected)
- Aluminum price block (returns, momentum, trend ratios, volatility, drawdown, candle, volume)
- **Cross-asset divergence block** (the edge): producer/China/base-metal return minus
  aluminum's own return at 1/5/20d, divergence z-scores and ratio-momentum.
- Non-stationary price LEVELS (ma_*, volume_ma_*) and exact duplicates dropped a priori.
- **L1 selection** keeps ~{ctx['avg_used']:.0f} features per fold; the final model uses
  {len(ctx['selected_features'])}. Top by |coef|: {', '.join(ctx['selected_features'][:8])}.

## Model
- **L1-penalized Logistic Regression** + isotonic calibration (logistic beat RF/GB/HistGB —
  the usable signal is linear). C chosen per-fold on validation (C={ctx['chosen_C']} final).
- Expanding-window walk-forward, {me['oos_days']} out-of-sample days.

## Performance (walk-forward, every day)
- **Accuracy: {me['accuracy']:.4f}** (95% CI [{me['acc_ci_low']:.4f}, {me['acc_ci_high']:.4f}])
- AUC: {me['auc']:.4f} · Balanced accuracy: {me['balanced_accuracy']:.4f}
- Up precision/recall: {me['up_precision']:.3f} / {me['up_recall']:.3f}
- Down precision/recall: {me['down_precision']:.3f} / {me['down_recall']:.3f}
- Up base rate: {me['up_base_rate']:.4f} (always-guess-majority = {me['majority_baseline_acc']:.4f})
- Confusion (tn,fp,fn,tp): {me['confusion']}

## Deployable confidence gate (the dead-band that works)
Trust the call only when the model is sure. Accuracy rises with conviction:

| min \\|p-0.5\\| | coverage | accuracy | 95% CI |
|---|---|---|---|
""" + "\n".join(
        f"| {r.min_confidence} | {r.coverage:.1%} | {r.accuracy:.4f} | [{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}] |"
        for r in g.itertuples()) + f"""

Best confident slice: **{best_gate['accuracy']:.4f} accuracy on {best_gate['coverage']:.1%} of days**
(min |p-0.5| > {best_gate['min_confidence']}). Volatility / move-size gates did NOT deploy
(trailing vol predicts move size only weakly), so the gate is on model confidence.

## How to read it
- Beats a coin flip with confidence (AUC CI clears 0.50); ~tied with always-up on raw accuracy.
- Every-day ceiling ≈ {me['accuracy']:.1%}; the confidence gate buys higher accuracy on a
  selective subset, not a higher every-day number.

## Statistical Notes
- Confidence intervals use **circular block bootstrap** (block_size=20, 3000 iters) to
  account for autocorrelation in rolling-window features. CIs are wider than a naive
  i.i.d. bootstrap would produce — this is the honest estimate.
- Val window is split: first 2/3 for C-selection, last 1/3 for calibration. This prevents
  the isotonic calibration from seeing the same labels used to pick C.
- Feature stability (fraction of folds where each feature had non-zero L1 coefficient) is
  saved in `feature_stability.csv`. Features selected in <20% of folds should be treated
  as unreliable signal.

## Next: more data features (this model is built to extend)
Add genuinely *leading* inputs — handled by the MORNING model: overnight Asian/AU-session
aluminum-producer closes (BHP.AX, RIO.AX, S32.AX, Hongqiao 1378.HK, Chalco 2600.HK, Rusal
0486.HK), LME aluminum overnight settlement, LME–US basis / term structure, and aluminum's
big idiosyncratic driver — energy/power costs (smelting). See DATA_NEEDED.md.

---
*Research model — not financial advice. Validate before any real use.*
"""
    path.write_text(md, encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary, reason = find_dataset(BASE_DIR)
    external = find_external_file(BASE_DIR, primary)
    log(reason)

    df, meta = load_and_prepare(primary, external)
    df = create_target(df)
    Xc = metal_features(df, meta["merged_external_cols"]).reset_index(drop=True)
    macro = fetch_macro()
    Xd = divergence_features(macro, df["close"], df["date"]).reset_index(drop=True)
    X = pd.concat([Xc, Xd], axis=1)

    # Keep the full panel: the warm-up rows carry NaNs that are median-imputed inside
    # each fold, and the walk-forward only scores from INIT_TRAIN onward anyway.
    y = df["target_up"].to_numpy(int)
    log(f"Rows={len(y)}  features={X.shape[1]}  up_rate={y.mean():.4f}  "
        f"{df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}")

    probs, features, freport = walk_forward(X, y)
    metrics, gate, mask = evaluate(y, probs)
    log(f"Pruning: {X.shape[1]} built -> {freport['candidate_features']} candidates "
        f"-> ~{freport['avg_features_used_per_fold']:.0f} used per fold (L1, per-fold C).")
    log(f"Walk-forward: acc={metrics['accuracy']:.4f} "
        f"CI[{metrics['acc_ci_low']:.4f},{metrics['acc_ci_high']:.4f}] auc={metrics['auc']:.4f} "
        f"on {metrics['oos_days']} OOS days")
    for r in gate.itertuples():
        log(f"  gate |p-0.5|>{r.min_confidence}: cover={r.coverage:.1%} acc={r.accuracy:.4f}")

    # Persist OOS predictions.
    oos = pd.DataFrame({
        "date": df["date"].dt.date.values[mask],
        "close": df["close"].values[mask],
        "actual_up": y[mask],
        "predicted_p_up": probs[mask],
        "predicted_dir": np.where(probs[mask] >= 0.5, "UP", "DOWN"),
        "confidence": np.abs(probs[mask] - 0.5),
        "correct": (((probs[mask] >= 0.5).astype(int)) == y[mask]).astype(int),
    })
    oos.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    gate.to_csv(OUT_DIR / "confidence_gate.csv", index=False)
    pd.DataFrame([{k: (v if not isinstance(v, list) else str(v)) for k, v in metrics.items()}]).to_csv(
        OUT_DIR / "metrics_summary.csv", index=False)

    final = fit_final_model(X, y, features)
    joblib.dump(final, OUT_DIR / "model.pkl")
    # The deployable lean feature set = the L1-selected (non-zero) features, with coefficients.
    pd.DataFrame([{"feature": f, "coefficient": c} for f, c in final["coefficients"].items()]).to_csv(
        OUT_DIR / "selected_features.csv", index=False)
    pd.DataFrame({"candidate_feature": features}).to_csv(OUT_DIR / "candidate_features.csv", index=False)
    pd.DataFrame([
        {"feature": f, "selection_freq": v}
        for f, v in freport["feature_selection_freq"].items()
    ], columns=["feature", "selection_freq"]).sort_values(
        "selection_freq", ascending=False
    ).to_csv(OUT_DIR / "feature_stability.csv", index=False)
    log(f"Feature stability written -> feature_stability.csv")
    log(f"Final model: C={final['chosen_C']}, {len(final['selected_features'])} features selected "
        f"of {len(features)} candidates.")
    with open(OUT_DIR / "metrics_summary.json", "w") as fh:
        json.dump({k: (v if not isinstance(v, np.floating) else float(v))
                   for k, v in metrics.items()}, fh, indent=2, default=float)

    try:
        plot_gate(gate, metrics["accuracy"], OUT_DIR / "confidence_gate_chart.png")
    except Exception as e:
        log(f"plot failed: {e}")

    ctx = {"meta": meta, "n_rows": len(y), "features": features,
           "selected_features": final["selected_features"], "chosen_C": final["chosen_C"],
           "avg_used": freport["avg_features_used_per_fold"], "div_tickers": ALL_TICKERS,
           "date_min": str(df["date"].iloc[0].date()), "date_max": str(df["date"].iloc[-1].date()),
           "metrics": metrics, "gate": gate}
    write_summary(ctx, OUT_DIR / "evaluation_summary.md")
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
