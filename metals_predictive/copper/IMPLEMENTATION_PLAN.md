# Copper Direction Model v2 — Implementation Prompt

**Instructions:** Feed this entire file as your prompt. The task is to implement the improvements
listed below to the two model files in this directory. Read every referenced file before touching
anything. Implement changes exactly as specified. Do not add features, refactor, or restructure
beyond what is described. After all changes are made, run both models and confirm they execute
without error.

---

## Context

Working directory: `copper_direction_model_v2/`

Two model files:
- `copper_close_direction_model.py` — the night-before model (predict at prior close)
- `copper_close_morning_model.py` — the morning-of model (predict using Asian overnight returns)

A shared audit identified issues ordered below by severity. Implement all of them. Each section
states exactly what to change and where.

---

## Change 1 — Split validation window: C-selection vs calibration

**File:** `copper_close_direction_model.py`  
**Function:** `walk_forward` (line ~415)  
**Problem:** `val` is used for both C-grid selection AND isotonic calibration — two looks at the
same held-out labels, inflating calibration quality.

**What to do:**

Inside the fold loop, after computing `core`, `val`, `fc`, split `val` into two halves:
- `val_c` = first 2/3 of `val` rows → used only for C selection
- `val_cal` = last 1/3 of `val` rows → used only for calibration

Replace the current calls so `_fit_best_C` receives `val_c` and `_calibrate` receives `val_cal`.
Add a guard: if `val_cal` has fewer than 30 rows or only one class, fall back to using the
full `val` for both (to avoid crashing on early folds). Log a warning when the fallback triggers.

The split should be:
```python
split_idx = ce + (2 * len(val)) // 3
val_c   = np.arange(ce, split_idx)
val_cal = np.arange(split_idx, r)
if len(val_cal) < 30 or len(np.unique(y[val_cal])) < 2:
    val_c, val_cal = val, val   # fallback
```

Do the same inside `fit_final_model` — the function uses `val` for both C selection and
calibration on lines ~502–503. Apply the same 2/3 + 1/3 split there.

---

## Change 2 — Replace naive bootstrap with block bootstrap

**File:** `copper_close_direction_model.py`  
**Function:** `_boot_acc` (line ~446)  
**Problem:** The current bootstrap samples individual days i.i.d. Rolling-window features create
heavy autocorrelation, so the effective sample size is ~n/20. The i.i.d. bootstrap produces
confidence intervals that are too narrow.

**What to do:**

Replace the function body with a circular block bootstrap using `block_size=20`:

```python
def _boot_acc(correct, iters=3000, block_size=20):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(correct)
    if n < 40:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block_size))
    s = []
    for _ in range(iters):
        starts = rng.integers(0, n, n_blocks)
        blocks = [correct[st : st + block_size] for st in starts]
        sample = np.concatenate(blocks)[:n]
        s.append(float(sample.mean()))
    return tuple(np.percentile(s, [2.5, 97.5]))
```

The function signature and return type are unchanged. No other code needs to change.

---

## Change 3 — Fix the permutation null test to use the full pipeline

**File:** `overfitting_check.py`  
**Function:** `main` (line ~94)  
**Problem:** The permutation null runs at a fixed `C=0.05` with `calibrate=False`. The real model
selects C from a grid per fold and applies calibration. The null is slightly easier to beat than
it should be.

**What to do:**

In the permutation null section (line ~151 onwards), change the null runs to use the full
pipeline (C-grid + calibration=True), matching the real model exactly:

```python
# Real with full pipeline (matches main model):
real = walk_forward(X, y)           # no fixed cgrid, calibrate=True (defaults)
rmask = ~np.isnan(real)
real_acc = accuracy_score(y[rmask], (real[rmask] >= 0.5).astype(int))

# Null with full pipeline:
for k in range(K):
    ys = y.copy()
    rng.shuffle(ys)
    pn = walk_forward(X, ys)        # full pipeline on shuffled labels
    m2 = ~np.isnan(pn)
    null_accs.append(accuracy_score(ys[m2], (pn[m2] >= 0.5).astype(int)))
```

Note: this will make the permutation null runs slower (K × full pipeline each). K=20 is
acceptable. Do not reduce K.

Also update the local `walk_forward` function in `overfitting_check.py` to accept and honor
the `calibrate` parameter with a default of `True` (it currently defaults to `True` already —
verify and leave it).

---

## Change 4 — Morning-of model: add explicit no-shift control test

**File:** `copper_close_morning_model.py`  
**Problem:** The claim "a no-shift control stays at the 53.9% baseline" is asserted in comments
and the docstring, but never computed in code. This must be a committed, logged output.

**What to do:**

Add a new function `overnight_features_no_shift` directly below the existing
`overnight_features` function. It is identical to `overnight_features` except use
`sig = daily` (no `.shift(-1)`) — meaning it uses the *current* day's Asian return, not
tomorrow's:

```python
def overnight_features_no_shift(prices, close, dates) -> pd.DataFrame:
    """Control: same-day Asian return (NOT shift(-1)). Should NOT beat 53.9% baseline."""
    mp = prices.set_index("date").sort_index()
    mp = mp.reindex(pd.Index(dates).union(mp.index)).ffill().reindex(dates).reset_index(drop=True)
    copper_ret1 = close.pct_change(1).reset_index(drop=True)
    cols, ov = {}, []
    for tk in OVERNIGHT_TICKERS:
        name = base._safe_name(tk)
        if name not in mp.columns or mp[name].notna().sum() < 100:
            continue
        daily = mp[name].reset_index(drop=True).pct_change(1)
        sig = daily                                         # NO shift — control
        cols[f"{name}_overnight_ret"] = sig.to_numpy()
        cols[f"{name}_overnight_div"] = (sig - copper_ret1).to_numpy()
        ov.append(sig)
    if ov:
        agg = pd.concat(ov, axis=1).mean(axis=1)
        cols["asianminers_overnight_mean"] = agg.to_numpy()
        cols["asianminers_overnight_div"] = (agg - copper_ret1).to_numpy()
    return pd.DataFrame(cols, index=range(len(dates))).replace([np.inf, -np.inf], np.nan)
```

In `main()`, after computing and logging the main model's walk-forward results, add a control
run:

```python
log("--- No-shift control (should ~= night-before baseline) ---")
Xo_ctrl = overnight_features_no_shift(
    fetch_overnight(), df["close"], df["date"]
).reset_index(drop=True)
X_ctrl = pd.concat([Xc, Xd, Xo_ctrl], axis=1)
probs_ctrl, _, _ = base.walk_forward(X_ctrl, y)
mask_ctrl = ~np.isnan(probs_ctrl)
ctrl_acc = accuracy_score(y[mask_ctrl], (probs_ctrl[mask_ctrl] >= 0.5).astype(int))
ctrl_auc = roc_auc_score(y[mask_ctrl], probs_ctrl[mask_ctrl])
log(f"No-shift control: acc={ctrl_acc:.4f}  auc={ctrl_auc:.4f}  "
    f"(real model: acc={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f})")
log(f"Overnight lift: +{metrics['accuracy'] - ctrl_acc:+.4f} acc, "
    f"+{metrics['auc'] - ctrl_auc:+.4f} auc")
```

Save the control result to a CSV alongside the other outputs:
```python
pd.DataFrame([{"ctrl_acc": ctrl_acc, "ctrl_auc": ctrl_auc,
               "real_acc": metrics["accuracy"], "real_auc": metrics["auc"],
               "acc_lift": metrics["accuracy"] - ctrl_acc,
               "auc_lift": metrics["auc"] - ctrl_auc}]
             ).to_csv(OUT_DIR / "overnight_shift_validation.csv", index=False)
```

---

## Change 5 — Switch to RobustScaler

**File:** `copper_close_direction_model.py`  
**Functions:** `walk_forward` (line ~430) and `fit_final_model` (line ~499)  
**Problem:** `StandardScaler` is sensitive to fat-tail outliers common in commodity crises.
`RobustScaler` (scales by IQR) is the standard choice for financial features.

**What to do:**

Add `RobustScaler` to the sklearn imports at the top of the file:
```python
from sklearn.preprocessing import RobustScaler
```

In `walk_forward`, replace:
```python
sc = StandardScaler().fit(Xi.iloc[core])
```
with:
```python
sc = RobustScaler().fit(Xi.iloc[core])
```

In `fit_final_model`, replace the same pattern:
```python
sc = StandardScaler().fit(Xi.iloc[core])
```
with:
```python
sc = RobustScaler().fit(Xi.iloc[core])
```

Remove `StandardScaler` from the import if it is no longer used anywhere else. Check — if
`overfitting_check.py` also imports `StandardScaler` directly, update that file too.

---

## Change 6 — Add feature selection stability tracking

**File:** `copper_close_direction_model.py`  
**Function:** `walk_forward` (line ~415)  
**Problem:** `avg_features_used_per_fold` (a count) is reported, but we don't know which features
are stably selected vs. appearing in only a few folds. A feature selected in <20% of folds is
likely noise.

**What to do:**

Inside the fold loop, after fitting `base` (the L1 logistic), record which feature names have
non-zero coefficients:

```python
selected_names = [features[i] for i, c in enumerate(base.coef_.ravel()) if abs(c) > 1e-8]
fold_selections.append(selected_names)
```

Initialise `fold_selections = []` before the fold loop begins.

After the fold loop, compute selection frequency and add it to the report dict:

```python
from collections import Counter
all_counts = Counter(f for fold in fold_selections for f in fold)
n_folds = len(fold_selections)
selection_freq = {f: all_counts[f] / n_folds for f in features}
report["feature_selection_freq"] = selection_freq
```

In `main()`, after calling `walk_forward`, write the stability table to a CSV:

```python
pd.DataFrame([
    {"feature": f, "selection_freq": v}
    for f, v in freport["feature_selection_freq"].items()
], columns=["feature", "selection_freq"]).sort_values(
    "selection_freq", ascending=False
).to_csv(OUT_DIR / "feature_stability.csv", index=False)
log(f"Feature stability written -> feature_stability.csv")
```

Do not change the existing `candidate_features.csv` or `selected_features.csv` outputs.

---

## Change 7 — Add calendar features

**File:** `copper_close_direction_model.py`  
**Function:** `copper_features` (line ~242)  
**Problem:** Day-of-week and month-of-year effects are real in commodity markets and are trivially
cheap stationary features.

**What to do:**

Add a `date` parameter to `copper_features`:
```python
def copper_features(df: pd.DataFrame, external_cols: list[str]) -> pd.DataFrame:
```
becomes:
```python
def copper_features(df: pd.DataFrame, external_cols: list[str]) -> pd.DataFrame:
```
(no signature change needed — `df` already contains the `date` column)

At the end of `copper_features`, just before the final `replace([np.inf, -np.inf])` line, add:

```python
# Calendar features (stationary, no leakage)
dow = df["date"].dt.dayofweek.values.astype(float)   # 0=Mon, 4=Fri
f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
month = df["date"].dt.month.values.astype(float)
f["month_sin"] = np.sin(2 * np.pi * month / 12)
f["month_cos"] = np.cos(2 * np.pi * month / 12)
```

These four features are always non-NaN and always stationary. They will pass `filter_features`
automatically.

---

## Change 8 — Update evaluation_summary.md to report new metrics

**File:** `copper_close_direction_model.py`  
**Function:** `write_summary` (line ~525)

Add a section that includes the block-bootstrap CI caveat and the feature stability note.
After the existing confidence gate table, add:

```python
md += f"""
## Statistical Notes
- Confidence intervals use **circular block bootstrap** (block_size=20, {iters} iters) to
  account for autocorrelation in rolling-window features. Reported CIs are wider than
  a naive i.i.d. bootstrap would produce.
- Feature stability: fraction of walk-forward folds in which each feature had a non-zero L1
  coefficient is saved in `feature_stability.csv`. Features selected in <20% of folds
  should be treated as unreliable signal.
"""
```

Pass `iters=3000` as a constant or read it from the `_boot_acc` call. Keep it simple.

---

## Execution Instructions

After implementing all changes:

1. Run `python copper_close_direction_model.py` and confirm it completes without error.
   Check that the following new output files exist:
   - `outputs/copper_close_direction/feature_stability.csv`
   - Verify `metrics_summary.json` accuracy is still in the 0.52–0.56 range (expect it may
     shift slightly from Changes 1 and 2).

2. Run `python copper_close_morning_model.py` and confirm it completes without error.
   Check that the following new output file exists:
   - `outputs/copper_close_morning/overnight_shift_validation.csv`
   - Log line "No-shift control: acc=..." should appear and show acc near 0.53–0.55.
   - Log line "Overnight lift: +..." should show a meaningful positive lift (expect +0.04–+0.07).

3. Run `python overfitting_check.py` and confirm it completes without error.
   Note the updated permutation p-value — it should still be 0.000 or near zero.
   If it rises above 0.05, flag this explicitly in your response.

4. Do not re-run the models just to check outputs — trust the run logs. Only re-run if a
   model raises an exception.

5. Report back with:
   - Which changes were applied
   - The new accuracy / AUC from both models
   - The overnight shift validation result (real vs control)
   - The updated permutation p-value
   - Any unexpected issues encountered

---

## What NOT to do

- Do not add new cross-asset tickers or data sources.
- Do not change the walk-forward window sizes (`INIT_TRAIN`, `VAL_WINDOW`, `STRIDE`).
- Do not switch the base model away from Logistic Regression.
- Do not add CLI arguments, config files, or new classes.
- Do not touch `requirements.txt`, `README.md`, or any output CSV/PKL files directly.
- Do not implement Changes 5.9–5.13 from the audit report (LME spread, Kelly sizing,
  regime conditioning, rolling window) — those are future work and not in scope here.
