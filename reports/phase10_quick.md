# Phase 10: Quick check (4-ratio model)

Fit on train, reported on validation. The test split was not read. Protocol and amendment: docs/decisions.md, Phase 10.

## 1. Population

Train: 4082 of 4094 firms have all four ratios (285 bankrupt). Validation: 872 of 878 (60 bankrupt). Blank counts, train: {'current_ratio': 12, 'book_equity_to_liabilities': 10, 'net_profit_to_sales': 0, 'net_profit_plus_dep_to_liabilities': 10}; validation: {'current_ratio': 6, 'book_equity_to_liabilities': 5, 'net_profit_to_sales': 0, 'net_profit_plus_dep_to_liabilities': 5}.

## 2. Model choice (train 5-fold, pre-registered)

PR AUC: logistic regression 0.299, monotone LightGBM 0.351, difference +0.052 (the challenger needs +0.02). Challenger chosen: yes.
Logistic coefficients on standardized ratios (for reference): current_ratio -0.001, book_equity_to_liabilities -0.297, net_profit_to_sales -0.547, net_profit_plus_dep_to_liabilities -0.651. All negative as expected: yes.

Platt scaling on train out-of-fold raw scores: slope 0.853, intercept -0.305.

Mean absolute contribution per ratio on train firms (calibrated log-odds):

| Ratio | Mean absolute contribution |
|---|---|
| Cash earnings ratio (proxy) | 0.389 |
| Debt-to-equity ratio | 0.362 |
| Net profit margin | 0.174 |
| Current ratio | 0.165 |

## 3. Validation: 4-ratio model against the frozen 64-ratio model (same firms)

| Model | PR AUC (95% CI) | ROC AUC (95% CI) | Brier | ECE | Calibration slope |
|---|---|---|---|---|---|
| 4-ratio, monotone LightGBM + Platt | 0.315 (0.235 to 0.435) | 0.837 (0.782 to 0.888) | 0.0549 | 0.0184 | 1.01 |
| 64-ratio, frozen LightGBM + Platt | 0.782 (0.698 to 0.867) | 0.957 (0.932 to 0.978) | 0.0300 | 0.0134 | 0.98 |

No-skill PR AUC (validation bankrupt rate): 0.069. Paired difference, 4-ratio minus 64-ratio: PR AUC -0.467 (-0.546 to -0.360), ROC AUC -0.119 (-0.178 to -0.073).
Train out-of-fold, 4-ratio: PR AUC 0.337, ROC AUC 0.802, ECE 0.0082.

## 4. The 1 to 10 scale

Cut points are the 10th to 90th percentiles of the train out-of-fold calibrated probability: 0.0122, 0.0174, 0.0223, 0.0303, 0.0366, 0.0460, 0.0611, 0.0876, 0.1543. The scale is a rank within the Polish training population, not an absolute risk.

| Score | Band | Train OOF firms | Bankrupt | Rate (95% Wilson) | Val firms | Bankrupt | Rate (95% Wilson) |
|---|---|---|---|---|---|---|---|
| 1 | Low | 387 | 9 | 0.023 (0.012 to 0.044) | 30 | 1 | 0.033 (0.006 to 0.167) |
| 2 | Low | 397 | 6 | 0.015 (0.007 to 0.033) | 155 | 1 | 0.006 (0.001 to 0.036) |
| 3 | Low | 404 | 3 | 0.007 (0.003 to 0.022) | 52 | 0 | 0.000 (0.000 to 0.069) |
| 4 | Watch | 434 | 8 | 0.018 (0.009 to 0.036) | 136 | 2 | 0.015 (0.004 to 0.052) |
| 5 | Watch | 397 | 16 | 0.040 (0.025 to 0.064) | 105 | 2 | 0.019 (0.005 to 0.067) |
| 6 | Watch | 430 | 14 | 0.033 (0.019 to 0.054) | 53 | 2 | 0.038 (0.010 to 0.128) |
| 7 | Elevated | 407 | 20 | 0.049 (0.032 to 0.075) | 92 | 6 | 0.065 (0.030 to 0.135) |
| 8 | Elevated | 409 | 31 | 0.076 (0.054 to 0.106) | 78 | 5 | 0.064 (0.028 to 0.141) |
| 9 | High | 408 | 42 | 0.103 (0.077 to 0.136) | 79 | 13 | 0.165 (0.099 to 0.261) |
| 10 | High | 409 | 136 | 0.333 (0.289 to 0.380) | 92 | 28 | 0.304 (0.220 to 0.405) |

## Flags

1. Validation was not used for any choice, but it has few bankrupt firms, so the intervals are wide.
2. The cash earnings ratio is a proxy: (net profit + depreciation) / total liabilities, not operating cash flow / current liabilities.
3. Debt-to-equity uses total liabilities as debt (UNVERIFIED that this matches what is meant by debt).
4. The range shown in the app is the observed bankruptcy rate of train out-of-fold firms with the same score. It does not cover another country, another year or a different ratio definition.
5. The test split is spent, so nothing here is checked on untouched data.

