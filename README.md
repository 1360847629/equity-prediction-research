# Equity direction and volatility research

## Trading simulation

`results/trading/TRADING_REPORT.html` reports a historical delayed-execution simulation of the saved validation-selected context model. Rules are fixed in `TRADING_PLAN.json`: signal after close, next-session open entry and same-session close exit; at most 20 qualifying stocks (up probability at least .55), at most 5% of opening capital per stock including entry costs, uninvested cash earns zero. No leverage, shorts or overnight positions. There are no live broker orders.

```powershell
.\.venv\Scripts\python trading_backtest.py
.\.venv\Scripts\python verify_trading.py
```

Base assumed execution drag is 6 bps per side (1 commission, 2 half-spread, 3 slippage), with 0 and 15 bps sensitivity. These assumptions are not broker fee quotes or measured market impact. Entry cost is included in the allocation; both legs are charged. Portfolio statistics include returns, drawdowns, zero-risk-free Sharpe, exposure, turnover, net trade win rate/profit factor and total modeled costs. Full round-trip records, daily equity paths and annual metrics accompany the report. Close-marked drawdown does not measure intraday worst loss. The model's close-to-close prediction target differs from the executable open-to-close holding period; overnight returns are not captured.

At the 6 bps per-side scenario, the primary threshold strategy lost approximately 15.43% over the full sample, with a 16.46% maximum daily-close drawdown. The unfiltered top-20 strategy gained about 40.20% before modeled costs but lost 56.33% after the base costs. This demonstrates cost sensitivity, not a reason to optimize assumptions until returns look favorable. The primary threshold strategy also lost about 1.74% at zero modeled cost. It remains an exploratory retrospective simulation; no profitable live-trading claim is supported.

Equal-weight stock intraday and SPY intraday baselines provide related holding-window comparisons; SPY buy-and-hold includes overnight exposure and is contextual, not exposure matched. Missing year-boundary prediction sessions are retained as cash for stock strategies, with dates listed in metadata. Fractional notionals and adjusted-price proxy fills are assumed; integer shares, market capacity, actual auction execution, account-specific fees, taxes and intraday stop fills are not modeled. Historical outcomes cannot establish future loss limits.

## Latest experiment: market and sector ETF context with XGBoost

See `results/context/CONTEXT_REPORT.html` for the latest comparison. `CONTEXT_PLAN.json` fixes the features, model settings and validation rule. It adds 14 market/sector ETFs and longer stock momentum for 67 features total. Original-feature logistic regression is retrained on exactly the same rows as all richer models, so changes in evaluation coverage do not create a misleading comparison.

```powershell
.\.venv\Scripts\python -m pip install -r requirements-context.txt
.\.venv\Scripts\python context_research.py
.\.venv\Scripts\python verify_context.py
```

The first run downloads and caches context ETF bars. XGBoost uses its native Python training API, CPU histogram trees, binary logistic loss and fixed regularization. Its boosting-round count is selected from 50 and 150 using prior-year validation log loss; the best single context model is also selected by validation log loss. Models refit on all pre-test labels. No test-based threshold tuning or retries. Native-library availability is explicitly recorded; unavailable models are skipped with an error record, never silently replaced. API reference: https://xgboost.readthedocs.io/en/stable/python/python_intro.html

The report and CSV files contain aggregate and per-year metrics, validation choices, full predictions, run times, feature/data fingerprints and paired block-bootstrap uncertainty estimates. These are exploratory results from previously inspected data. All features are available only after the day's close. Nothing here executes trades or establishes profitability.

Measured result: historical-frequency baseline accuracy 52.00%, original-feature logistic control 51.81%, context logistic 50.44%, context neural 51.35%, XGBoost 51.22%. XGBoost was selected on validation in all four folds, but did not outperform the historical-frequency baseline on aggregate test log loss (0.693928 versus 0.692380; lower is better). The richer features and mature tree model did not establish an advantage. Do not replace these results with the earlier volatility gains.

## Current project: next-day direction prediction

The current deliverable is `results/direction/DIRECTION_REPORT.html` (Chinese). The target is the probability that the next observed adjusted close-to-close return is strictly positive. Zero returns are grouped with non-up observations. Features are available after the current close. Original volatility studies remain below as earlier experiments with a different target; their gains do not carry over to direction prediction.

Run after downloading the data and installing the locked requirements:

```powershell
.\.venv\Scripts\python direction_research.py
.\.venv\Scripts\python verify_direction.py
```

Models: L2 logistic regression, sigmoid-output neural classifier trained with binary cross entropy, and a convex ensemble selected on prior-year validation log loss. Four expanding-window test periods cover 2023–partial 2026, with 99,144 evaluated observations. The protocol is in `DIRECTION_PLAN.json`.

Measured accuracy: always-up/historical-rate baseline 52.00%, logistic 51.59%, neural classifier 51.56%, ensemble 51.65%. Historical-rate baseline log loss is 0.692411 versus ensemble 0.692826 and neural 0.693350 (lower is better). These models did not outperform the simple baseline on the primary metric. No trading advantage is established. Settings were not retuned after seeing these results.

Companion CSVs include annual/monthly results, ROC AUC, balanced accuracy, Brier scores, calibration, paired bootstrap intervals, runtime, ensemble choices and signed returns/labels/probabilities. Tests check target alignment, causal features, synthetic learning, probability ranges and saved-result consistency. This remains exploratory retrospective research on previously inspected data, not live trading.

## Expanded research study

The earlier volatility deliverable is `results/advanced/RESEARCH_REPORT.html`. It contains four expanding-window evaluations (2023, 2024, 2025 and partial 2026), ridge, boosted stumps, a ReLU neural network, validation-selected ensemble weights, a volatility-feature ablation, data audits and paired moving-block bootstrap intervals with three block-length settings. Full predictions, selection decisions, runtime records and metrics are saved alongside the report. The neural model reduced aggregate MAE by approximately 1.21% versus rolling-20; it did not beat that baseline in every year.

Reproduce the expanded study after the initial run:

```powershell
.\.venv\Scripts\python advanced_research.py
.\.venv\Scripts\python verify_advanced.py
```

The protocol is saved in `EXPERIMENT_PLAN.json`. This extension is exploratory: the dataset, including 2025–2026 outcomes, was already inspected during the starter project. Results do not establish a fresh confirmatory holdout or a sustained track record. `results/advanced/APPLICATION_WORDING.txt` provides a factual project description to use only after reproducing and understanding the work. The original single-period study below remains as a documented earlier experiment; these are the earlier volatility results.

A reproducible starter research project comparing constant, rolling-volatility, ridge, and gradient-boosted predictions. The included run uses 176,664 daily records across 102 selected US stocks, from January 2, 2020 through November 20, 2026. Read results/REPORT.md for findings and limitations.

## Run

Use Python 3.11 or later. In this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python research.py
.\.venv\Scripts\python verify.py
```

The first run downloads daily prices from Yahoo Finance using download_recent.py. It uses a fixed cutoff of November 23, 2026 (exclusive), excluding incomplete prices from that day. Subsequent runs reuse the cached CSV and check its hash against the manifest. Vendor revisions may mean a later download differs from this run, even with identical dates. The snapshot fingerprint and retrieval details are included in results/data_manifest.json.

To refresh later, explicitly choose a new cutoff:

```powershell
.\.venv\Scripts\python download_recent.py --as-of YYYY-MM-DD
.\.venv\Scripts\python research.py
.\.venv\Scripts\python verify.py
```

Keep the as-of date fixed to reproduce an experiment. Set evaluation periods for new experiments before viewing their held-out results.

## Methods and evaluation

Features use current and trailing price/volume information available after the close. The target is the absolute next-observation close-to-close return, a proxy for daily volatility, not return direction. Train: 2020-2023. Validation: 2024. Final test: 2025 through the latest observed label in November 2026. Labels crossing split boundaries are excluded. After configuration selection, models are refit on all pre-2025 observations.

The estimators in models.py use NumPy: ridge regression with standardized features, and squared-error boosting with quantile-binned decision stumps. Boosting is a limited reference implementation, not a full production tree library. No scikit-learn or SciPy is required.

## Included run

- 174,522 usable observations; 48,246 test observations.
- Ridge test MAE: 0.009810.
- Boosting test MAE: 0.009840.
- Rolling-20 baseline test MAE: 0.009966.
- Constant baseline test MAE: 0.010484.

Lower MAE is better. Errors are in decimal return units: 0.01 means one percentage point. Both fitted models modestly improve over the rolling baseline on this historical sample. These results do not establish statistical significance or trading profitability.

## Files

- download_recent.py: explicit ticker list, cached downloads and manifest.
- research.py: features, time splits, comparisons and report generation.
- models.py: NumPy estimators.
- verify.py: future-feature invariance and target alignment checks.
- results/REPORT.md: measured results and limitations.
- results/metrics.csv: validation/test errors and training durations.
- results/monthly_metrics.csv: performance variation by month.
- results/test_predictions.csv: auditable held-out predictions.
- results/run_metadata.json: counts, fingerprint, versions and configuration.
- results/data_manifest.json: coverage and source adjustments.
- results/verification.txt: verification outcome.

## Limitations and ownership

The convenience sample is explicitly listed in the download script. It is not a random sample or historical index membership; selection and survivorship bias apply. Yahoo adjusted-close ratios are applied to OHLC, and provider volume is unchanged. This is a retrospective vendor snapshot, not independently verified point-in-time data. Missing and invalid bars are removed; a next observation can follow a gap. Tail events are retained. Confirm usage terms before redistributing data; raw vendor data is excluded from the ZIP.

This is a modest educational dataset and does not demonstrate distributed computing or institutional-scale research. No tradable strategy, execution assumptions, transaction costs, or statistical significance tests are provided. The verification script checks feature causality and target alignment; the main script asserts temporal separation, duplicate keys and the input fingerprint. Additional data checks and walk-forward evaluation remain useful next steps.

Prepared with AI assistance. Reproduce the run, understand the code and extend it with your own experiments before presenting it in applications. Describe completed work accurately as a project; it does not establish professional experience or a proven research track record. The GitHub repository contains source code, summary results and documented limitations.


## Repository package

This repository package includes source code, experiment plans, reports, summary metrics and verification records. Large row-level prediction/trade files, raw vendor data and saved tree models are intentionally omitted. Run the corresponding scripts to regenerate them. Use the documented data fingerprints to identify differences from the original snapshot; a fresh download can differ after vendor revisions. Earlier report references to companion full predictions refer to files generated locally, not files committed here. The published reports document AI assistance and the unfavorable direction/trading results.

