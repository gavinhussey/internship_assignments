# Internship Assignments

This is my internship project folder — a collection of the research and analysis
projects I worked on, spanning quantitative crypto research, a trading-comps
valuation exercise, a futures hedging assignment, and an operations data-matching
tool. Each project lives in its own subfolder (or standalone file) and is
self-contained.

## Projects

### [`magnitude_of_move/`](magnitude_of_move) — BTC Next-Day Move Magnitude
Predicts the *size* of tomorrow's BTC move (not direction) as a bullish/bearish
price range. Builds a volatility-regime and feature-engineering pipeline, tries a
regime classifier and hurdle model before settling on quantile regression
(`GradientBoostingRegressor`) as the deliverable, then stress-tests it: benchmarked
against a GARCH(1,1) baseline, scored with pinball loss, given a block-bootstrap
confidence interval, and back-tested through the FTX collapse (Nov 2022) as a
known regime-shift event. Honest finding: the engineered feature set doesn't
convincingly beat GARCH once uncertainty is accounted for.

### [`BTC-Directional-Model-Audit/`](BTC-Directional-Model-Audit) — Auditing a Prior Directional Model
Audits a previous intern's claimed 55.94%-accuracy BTC next-day *direction* model,
finds the result was inflated by feature-selection leakage, rebuilds it
leakage-free, and then runs five independent modeling attempts (a much larger
feature pool, longer horizons, per-fold feature selection, regime-switching, and
trend-following) to search harder for a real edge. None of the ten total attempts
produce a statistically defensible edge over "always predict up" — see the
folder's README for the full walkthrough and results tables.

### [`dtc_matcher_deliverable/`](dtc_matcher_deliverable) — DTC Ticket Matcher
An operations tool (not a research project) that matches drop-ship/pass-through
orders to their outbound and inbound yard tickets by weight, date window, and
supplier cross-check, producing a confidence-graded match report.

### [`hedge_assignment.html`](hedge_assignment.html) — Futures Hedging Assignment
A self-contained report computing an optimal futures hedge (formula, variables,
and results) for a given position, dated 2026-06-30.

### [`Space_X_Trading_Comp.xlsx`](Space_X_Trading_Comp.xlsx) — SpaceX Trading Comps
A guided comparable-companies ("comps") valuation of SpaceX: selecting and
screening peer companies, pulling market data, choosing and applying trading
multiples, and concluding a valuation range.
