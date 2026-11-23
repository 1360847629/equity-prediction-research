# Equity volatility research: measured results

Source: https://finance.yahoo.com/

Dataset: 176,664 original rows; 174,522 usable rows across 102 stocks.
Raw prices: 2020-01-02 through 2026-11-20. Usable feature dates: 2020-01-31 through 2026-11-19; final labels use the next observed close.

## Research question
Can price and volume information available after a daily close predict the magnitude of the next observed close-to-close return? This predicts volatility, not direction or profit.

## Evaluation
Train: 2020-2023. Validation: 2024. Test: 2025 onward. Labels crossing split boundaries are excluded. Two quantile-bin configurations for a custom NumPy gradient-boosted decision-stump model are compared on validation MAE; the selected configuration and ridge model are refit on pre-2025 data. Test data is reserved for the final comparison. Features include the current close, so predictions are only available after that close.

| Model | Test MAE (return units) | Test RMSE | Fit seconds |
|---|---:|---:|---:|
| ridge | 0.009810 | 0.015125 | 0.04 |
| gradient_boosting | 0.009840 | 0.015211 | 0.83 |
| rolling_20 | 0.009966 | 0.015505 | 0.00 |
| constant | 0.010484 | 0.016183 | 0.00 |

Gradient boosting improvement over rolling-20 baseline in test MAE: 1.26%. Negative means worse.

## Scale and limitations
Feature table memory: 12.0 MiB (not peak process memory). This is a modest-sized educational dataset, not evidence of distributed computing or institutional-scale research.
The convenience sample contains selected currently traded companies, introducing selection and survivorship bias; it is not a point-in-time index universe. OHLC values use Yahoo adjusted-close ratios; volume is unchanged. Adjustments are retrospective and vendor data is not independently verified. Tail events are retained; vendor errors may create spurious extremes. No transaction costs, execution assumptions, or tradable strategy are tested. No claim of profitability is supported. A single historical holdout cannot establish robustness across regimes.

## Next experiments
Investigate tail observations; obtain verified adjusted point-in-time data; add rolling walk-forward folds, daily block-bootstrap uncertainty, and feature ablations. Measure runtime and peak memory at increasing sample sizes. Do this before making stronger research claims.

## Data snapshot
Provider: Yahoo Finance. The download manifest records the as-of cutoff, retrieval time, ticker coverage, exclusions and adjustment convention. Latest raw close is reported in run_metadata.json; the last feature date is one observation earlier because evaluation needs a known next close. Raw data is not bundled for redistribution.

## Ownership
Prepared with AI assistance. Review, reproduce, explain, and extend the work before presenting it as your own independent research. Do not describe this as professional experience or a proven track record.