# Phase 10: Quick check (4-ratio model)

Fit on train, reported on validation. The test split was not read. Protocol: docs/decisions.md, Phase 10.

## 1. Population

Train: 4082 of 4094 firms have all four ratios (285 bankrupt). Validation: 872 of 878 (60 bankrupt). Blank counts, train: {'current_ratio': 12, 'book_equity_to_liabilities': 10, 'net_profit_to_sales': 0, 'net_profit_plus_dep_to_liabilities': 10}; validation: {'current_ratio': 6, 'book_equity_to_liabilities': 5, 'net_profit_to_sales': 0, 'net_profit_plus_dep_to_liabilities': 5}.

## 2. Coefficients and checks

| Feature (standardized) | Coefficient | Expected sign |
|---|---|---|
| current_ratio | -0.001 | negative |
| book_equity_to_liabilities | -0.297 | negative |
| net_profit_to_sales | -0.547 | negative |
| net_profit_plus_dep_to_liabilities | -0.651 | negative |

Intercept -3.012. All signs as pre-registered: yes.

Train 5-fold PR AUC: logistic regression 0.299, monotone LightGBM challenger 0.351, difference +0.052 (challenger needs +0.02). Challenger wins: yes.

## STOPPED

The monotone challenger beat logistic regression by the pre-registered margin. The user decides.

