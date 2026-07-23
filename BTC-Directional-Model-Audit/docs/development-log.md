# Improving the Provided BTC Next-Day Direction Model

## Objective

I was given a supervised learning model created by Sohan Bendre, a previous RCM intern, that predicts whether Bitcoin will close up or down the next day. My task was to audit the existing workflow, understand how the model works, identify weaknesses, and outline concrete steps to improve its reliability. I was also given the task at extending lookahead periods from 1D to 7D and 30D predictive periods.

The current best-performing model is a standardized logistic regression classifier. It predicts the next-day BTC close direction using transformed on-chain, market, and macroeconomic features.

## Baseline Model Performance

The reported best model uses an 80/20 time-based split:

Training period: 2019-04-10 to 2024-03-07
Testing period:  2024-03-08 to 2025-05-29
Test samples:    320


Its reported performance is:

Accuracy:              55.94%
F1 score:              0.6536
Test majority baseline: 50.31%


The model beats the test-period majority baseline by about 5.6 percentage points. However, it has a strong directional bias: it predicts "up" on 77.5% of test days.

Confusion matrix [[tn, fp], [fn, tp]]:
[[46, 115], [26, 133]]

This means the model is effective at identifying many up days, but it also produces many false positives on down days.

## Current Model Features

The logistic regression model uses 14 transformed features:

Tr Miner Reserve
Tr Hashrate
Tr Mean Coin Age
Tr Open Interest
Tr S&P500
Tr Nasdaq
Tr Dow
Tr Gold
Tr Silver
Tr Copper
Tr Fed Funds Raw
Tr Fed Funds Delta
Tr Treasury Pct
Tr Treasury Delta


These features were selected from a broader dataset containing BTC price data, on-chain metrics, traditional market indicators, and macroeconomic variables.

## Key Finding: Feature-Selection Leakage

The most important issue I found is feature-selection leakage.

In the provided notebook, the full transformed dataset is built across the entire period from 2019-04-10 to 2025-05-29. Correlation analysis and Boruta-style feature selection are then run on that full dataset. Only after that does the notebook create the final train/test split.

That means the holdout period from 2024-03-08 to 2025-05-29 influenced which features were selected. As a result, the 55.94% test accuracy should be treated as exploratory rather than a clean out-of-sample result.

## Train-Only Feature Selection Check

To test the impact of this issue, I reran the shadow-feature screen using only the training period. The support for the existing feature set changed materially.

Market features:

Tr Copper             6/50
Tr Dow                1/50
Tr Gold               1/50
Tr Oil                1/50
Tr S&P500             0/50
Tr Nasdaq             0/50
Tr Silver             0/50
Tr Natural Gas        0/50
Tr Dollar Index       0/50
Tr EURUSD             0/50
Tr USDCNY             0/50


On-chain features:

Tr Mean Coin Age      39/50
Tr Hashrate           24/50
Tr Fee-Reward Ratio   7/50
Tr MVRV Ratio         6/50
Tr Miner Reserve      2/50
Tr Exchange Netflow   1/50
Tr Funding Rates      1/50
Tr Exchange Reserve   0/50
Tr Puell Multiple     0/50
Tr SOPR               0/50
Tr Active Addresses   0/50
Tr Open Interest      0/50


Economic features:

Tr Treasury Delta     5/50
Tr Treasury Pct       3/50
Tr Fed Funds Raw      0/50
Tr Fed Funds Delta    0/50
Tr Unemployment       0/50
Tr CPI                0/50
Tr M2 Raw             0/50
Tr M2 Binary          0/50
Tr GDP Raw            0/50
Tr GDP Binary         0/50


The train-only screen suggests that `Tr Mean Coin Age` and `Tr Hashrate` are the most defensible features. Several features used in the final model, including `Tr Open Interest`, most equity/metal market features, and the Fed Funds variables, look much weaker when the test period is excluded from feature selection.

## Interpretation

The provided model does show some predictive signal, but the current evaluation likely overstates confidence because feature selection was not isolated from the test set. The model also appears to earn much of its F1 score by predicting "up" frequently, so accuracy alone is not enough to judge whether the model is useful.

Before adding more complex models, the priority should be to make the validation process cleaner and more realistic.

## Mk2 Workbench Changes and Initial Findings

I created a new improvement notebook in `BTC-Price-Direction-Mk2/model_improvement_workbench.ipynb`. The goal of this notebook is not to overwrite the original model, but to create a clean workspace where I can reproduce the inherited result and then test improvements under stricter validation rules.

The Mk2 workbench changes the model development process in several important ways:

1. It treats the original model folder as read-only.

   The notebook reads the original CSV files from `Supervised-Learning-BTC-Price-Direction/Pricing Models`, but all new experiment outputs are written into the Mk2 folder.

2. It recreates the current best model as a baseline.

   The workbench reproduces the inherited logistic regression result:

   Accuracy: 55.94%
   F1 score: 0.6536
   Features: 14
   Test period: 2024-03-08 to 2025-05-29
   

   This confirms that the Mk2 notebook starts from the same performance baseline as the original model.

3. It adds a broader candidate feature set.

   In addition to the original on-chain, market, and macro features, the workbench adds BTC-native technical features:

   Tr BTC Return 1D
   Tr BTC Return 3D
   Tr BTC Return 7D
   Tr BTC Volatility 7D
   Tr BTC MA Gap 7D
   Tr BTC MA Gap 30D
   

   This matters because next-day direction may depend more on short-term BTC behavior than slow-moving macro variables.

4. It adds train-only feature selection.

   The notebook now selects features using only the training window rather than using the full dataset. This directly addresses the feature-selection leakage identified in the original workflow.

5. It adds walk-forward validation.

   Instead of judging the model only on one final 80/20 split, the workbench evaluates performance across multiple time-ordered folds. This tests whether the model signal is stable across different market regimes.

6. It adds probability-threshold testing.

   Since the current logistic model predicts "up" too often, the workbench includes infrastructure to test whether only taking higher-confidence predictions improves precision.

## Mk2 Logistic Regression Experiment Results

In the Mk2 folder, I removed references to the other original model families and focused only on improving the strongest inherited model: logistic regression.

On the original final holdout period, the logistic-regression-only experiments produced:

Current best logistic:       55.94% accuracy, 0.6536 F1
Train-selected logistic:     53.44% accuracy, 0.4808 F1
All-features L1 logistic:    52.53% accuracy, 0.5192 F1


The train-selected logistic model is more conservative than the inherited model. It predicts "up" on 40.0% of test days, compared with 77.5% for the inherited logistic model. That reduces the up-bias, but the first version also lowers accuracy from 55.94% to 53.44%.

The walk-forward results are more cautious than the single holdout result:

Current feature set, walk-forward average:
Accuracy: 52.73%
F1 score: 0.3667

Train-selected feature set, walk-forward average:
Accuracy: 50.26%
F1 score: 0.4706


This is an important finding. The original feature set looks strongest on the final holdout period, but its performance is uneven across time. The train-selected feature set does not yet improve accuracy, but it produces a more balanced prediction profile and is methodologically cleaner.

## Leakage-Free Logistic Regression Findings

I then added a section to `model_improvement_workbench.ipynb` called `Fully Leakage-Controlled Logistic Regression`. This section fixes the remaining feature-selection leakage by making sure that feature selection, scaling, and model fitting are learned only from data available before each test period.

The leakage-free comparison is:

Original logistic regression:
Accuracy: 55.94%
F1 score: 0.6536
Validation: original final holdout

Leakage-safe holdout logistic:
Accuracy: 51.90%
F1 score: 0.3667
Validation: final holdout with train-only feature selection
Average prediction-up rate: 26.27%

Leakage-safe walk-forward logistic:
Accuracy: 50.57%
F1 score: 0.3291
Validation: walk-forward with fold-local feature selection
Average prediction-up rate: 25.86%


This is the clearest evidence that leakage mattered. Once feature selection is prevented from seeing the test period, the apparent edge drops from 55.94% accuracy to roughly 51.90% on the same final holdout and 50.57% under stricter walk-forward validation.

The practical takeaway is that the inherited logistic model was directionally interesting, but its reported edge was overstated by the validation design. The leakage-free version is much closer to baseline and is too conservative in its current form, predicting "up" only about one quarter of the time.

## Mk2 Interpretation

The first Mk2 version improves the research process more than the headline model score. It confirms that the inherited model can be reproduced, then shows that the model's apparent edge weakens under walk-forward validation.

At this stage, I would not claim that the Mk2 model is already better than the original model on raw accuracy. Instead, the improvement is that the Mk2 workflow is now set up to test future changes more honestly. The next modeling work should focus on improving performance inside this cleaner framework rather than optimizing against a single holdout period.

## BTC-Mk2: First Full Rebuild Attempt

After confirming the leakage problem and building the cleaner validation process in the workbench notebook, I started a first full rebuild of the model. Where the workbench notebook stays inside the original's inherited data files and feature set, BTC-mk2 replaces the data pipeline itself and builds a genuinely new candidate model from scratch, under the corrected, leakage-safe validation rules established above.

### How this model differs from the original (non-leakage-safe) model

1. Live, self-owned data pulls instead of static provided CSVs.

   The original model reads from committed CSV snapshots (`bitcoin-price.csv`, `market-data.csv`, `on-chain-data.csv`, `economic-daily.csv`/`economic-more.csv`). BTC-mk2 pulls fresh data directly from four sources each time it runs:

   Coinbase:    10 years of daily BTC OHLCV
   CryptoQuant: full on-chain feature catalog (reserves, flows, MVRV, SOPR, funding rates, open interest, etc.)
   yfinance:    cross-asset market data (S&P 500, Nasdaq, Dow, gold, silver, copper, oil, natural gas, dollar index, EUR/USD, USD/CNY)
   FRED:        macroeconomic indicators (CPI, Fed funds rate, 10Y Treasury yield, GDP, M2, unemployment)
   

2. A much larger, broader candidate feature set.

   Instead of the original's 14 hand-picked features, BTC-mk2 engineers 111 candidate features: BTC-native short-horizon technicals (returns, momentum acceleration, realized volatility, range, volume z-scores, close-location, moving-average gaps) plus percent-change/delta/z-score/acceleration transforms of every pulled CryptoQuant, market, and economic series.

3. Feature selection is done on the training window only, with a stability filter.

   The original ran Boruta/correlation screening across the entire dataset, including the future holdout period. BTC-mk2 ranks candidates by correlation with the target computed strictly on the training split, and only keeps a feature if its correlation sign is consistent across both halves of the training period. This favors features with a persistent relationship over ones that just happen to score high once.

4. Publication-lag-aware macroeconomic data.

   FRED dates each release by the period it describes, not when it was actually published — January CPI is dated `2024-01-01` in the API but isn't public until mid-February. This is a second leakage source in the original model that isn't fixed by the workbench's train-only feature selection. BTC-mk2 shifts every series forward by its real reporting lag before use:

   CPI                   45 days
   Fed Funds Rate        35 days
   GDP                   120 days
   M2 Money Supply       45 days
   Unemployment Rate     40 days
   10Y Treasury Yield    1 day (same-day market series)
   

   It also forward-fills rather than linearly interpolating between releases, so a given day's features never depend on a future print that hadn't happened yet.

5. Trading-calendar-aware merging.

   BTC trades every day; equities and FX close on weekends and holidays. BTC-mk2 forward-fills the non-BTC sources onto BTC's daily calendar so a handful of non-trading days doesn't wipe out every feature for that date.

6. A strictly enforced chronological train/test split.

   The split point is asserted in code (`train.max(Date) < test.min(Date)`) rather than just described in prose, and the scaler and logistic regression model are fit exclusively on the training partition. NaN-handling is scoped to whichever features are actually selected, rather than a blanket drop across all 111 candidates, so a handful of sparse macro columns can't silently wipe out most of the dataset.

### Results

Training samples: 703 (2017-09-11 to 2024-03-25)
Test samples:     208 (2024-07-12 to 2026-04-23)
Features selected: 15, a mix of BTC technicals (return/momentum/close-location) and
                    macro features (Unemployment Rate, M2 Money Supply, GDP,
                    S&P 500 / Nasdaq deltas)

BTC-mk2 (leakage-safe):
Accuracy:          52.40%
F1 score:          0.5520
Precision:         0.5214
Recall:            0.5865
ROC AUC:           0.5055
Majority baseline: 50.00%
Beat baseline by:  +2.40 points

For comparison:

Original, as claimed (leakage present):        55.94% accuracy, 0.6536 F1
Original, leakage-safe holdout (from above):    51.90% accuracy, 0.3667 F1
Original, leakage-safe walk-forward:            50.57% accuracy, 0.3291 F1

### Interpretation

On raw headline accuracy, BTC-mk2 (52.40%) looks lower than the original's reported 55.94%. That comparison isn't fair to either model, though, since the original's number includes feature-selection leakage. Compared on equal, leakage-safe footing, BTC-mk2 modestly outperforms the corrected original: +2.40 points over baseline versus +1.6 points (holdout) or roughly 0 points (walk-forward) for the leakage-safe original.

This is a small edge, and it should be read as a first working rebuild rather than a finished model. The current feature set also has some redundancy — several of the top BTC-native features (1-day return, its negation, intraday return, short-horizon momentum acceleration) are close restatements of the same single-day move, which uses up selection slots that could go to more independent signal. The next round of work should de-duplicate correlated candidates during selection and test whether a larger or differently tuned candidate pool widens the margin further.

## Conclusion

My main improvement to the provided model is not immediately replacing the algorithm. The first step is correcting the validation design. Once feature selection is moved inside a walk-forward process, the model can be re-evaluated on genuinely unseen data. After that, additional BTC-native features, cleaner lags, and probability-based trading thresholds can be tested with more confidence.
