# Data report: Polish companies bankruptcy, 5year

Primary dataset for solvency-signal. Produced in Phase 1 by `src/preprocess.py`.

## Source

- UCI Machine Learning Repository, dataset 365, file `5year.arff` inside `data/raw/polish+companies+bankruptcy+data.zip`.
- Ratios come from the 5th (last) year of the forecast period; the label is bankruptcy within the following year (1-year horizon, per UCI).
- Licence: CC BY 4.0. Cite: Tomczak, S. (2016). Polish Companies Bankruptcy. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600

## Pipeline result

| Stage | Rows | Bankrupt | Rate |
|---|---|---|---|
| Raw | 5,910 | 410 | 6.94% |
| After dropping exact duplicates | 5,850 | 408 | 6.97% |
| Train (70%) | 4,094 | 286 | 6.99% |
| Validation (15%) | 878 | 61 | 6.95% |
| Test (15%) | 878 | 61 | 6.95% |

Stratified random split, seed 42. No company id or year exists, so a time-based split is not possible.
The 60 dropped duplicates had no conflicting labels (58 alive, 2 bankrupt rows removed).

Clipping (1st and 99th percentile), median imputation with missing indicators and standard scaling are defined in
`build_preprocessor()` and are fitted on the training split only. They are not applied to the saved CSVs.

## Findings and flags

1. **Column order verified.** `assets_to_liabilities` (Attr17) equals 1 / `liabilities_to_assets` (Attr2) in 5,889 of 5,891 rows,
   and quick ratio sits between cash ratio and current ratio in 99.6% of rows.
2. **Probable data error, raw row 1992:** Attr17 = 0.0064 while 1 / Attr2 = 1.76. Left in place; flagged.
3. **"Gross profit" is probably profit before tax.** Attr19 (gross profit / sales) has Spearman 0.986 with net margin (Attr23)
   but differs from (sales - COGS) / sales (Attr56) in 96% of rows. Inference from data and the Polish term "zysk brutto"; not confirmed by UCI.
4. **Attr48 and Attr49 are internally inconsistent:** labelled EBITDA but the formula subtracts depreciation.
5. **Near duplicates:** Attr9 and Attr36 are equal in 58% of rows. Attr17 is an exact reciprocal of Attr2.
6. **No true operating cash flow.** Closest proxies: Attr26 (net profit + depreciation) / total liabilities and Attr16.
   These are also the strongest single features (train AUC about 0.79, direction ignored).
7. **Attr37 missing (42.65%) is structural.** `long_term_liabilities_to_equity` is exactly 0 in 42.6% of rows, so the missing
   value means "no long-term liabilities" (division by zero), not unknown.
8. **Missingness is predictive.** When Attr27 is missing, 30.0% of firms are bankrupt vs 5.4% when present; Attr37 missing: 8.0% vs 6.2%.
   Missing indicators help here, but a model that leans on them may not transfer to real company input where those values are known. Flag for Phase 2.
9. **Possible zeros coded for missing:** operating profit ratios (Attr22, Attr42) are exactly 0 in 9.1% of rows; retained earnings (Attr6) in 38.5%. Unverified.
10. **Extreme tails.** Several ratios have max values thousands of times their 99th percentile (for example `inventory_turnover` max 4.8 million vs p99 606),
    caused by near-zero denominators. Handled by train-fitted clipping.
11. **Attr55 is an absolute amount** (working capital), not a ratio; currency and unit not stated.

## Feature dictionary

| Attr | Name | UCI formula | Train missing % | Flag |
|---|---|---|---|---|
| Attr1 | `net_profit_to_assets` | net profit / total assets | 0.05 |  |
| Attr2 | `liabilities_to_assets` | total liabilities / total assets | 0.05 |  |
| Attr3 | `working_capital_to_assets` | working capital / total assets | 0.05 |  |
| Attr4 | `current_ratio` | current assets / short-term liabilities | 0.29 |  |
| Attr5 | `cash_interval_days` | [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365 | 0.22 |  |
| Attr6 | `retained_earnings_to_assets` | retained earnings / total assets | 0.05 |  |
| Attr7 | `ebit_to_assets` | EBIT / total assets | 0.05 |  |
| Attr8 | `book_equity_to_liabilities` | book value of equity / total liabilities | 0.24 |  |
| Attr9 | `sales_to_assets` | sales / total assets | 0.00 | near duplicate of Attr36 (equal in 58% of rows) |
| Attr10 | `equity_to_assets` | equity / total assets | 0.05 |  |
| Attr11 | `gp_extra_finexp_to_assets` | (gross profit + extraordinary items + financial expenses) / total assets | 0.05 | 'gross profit' likely means profit before tax (translation) |
| Attr12 | `gp_to_short_term_liabilities` | gross profit / short-term liabilities | 0.29 | 'gross profit' likely means profit before tax (translation) |
| Attr13 | `gp_plus_dep_to_sales` | (gross profit + depreciation) / sales | 0.00 | 'gross profit' likely means profit before tax (translation) |
| Attr14 | `gp_plus_interest_to_assets` | (gross profit + interest) / total assets | 0.05 | 'gross profit' likely means profit before tax (translation) |
| Attr15 | `liabilities_days_of_gp_dep` | (total liabilities * 365) / (gross profit + depreciation) | 0.07 | 'gross profit' likely means profit before tax (translation); sign flips when denominator is negative |
| Attr16 | `gp_plus_dep_to_liabilities` | (gross profit + depreciation) / total liabilities | 0.24 | 'gross profit' likely means profit before tax (translation); cash flow proxy, not true operating cash flow |
| Attr17 | `assets_to_liabilities` | total assets / total liabilities | 0.24 | exact reciprocal of Attr2 |
| Attr18 | `gp_to_assets` | gross profit / total assets | 0.05 | 'gross profit' likely means profit before tax (translation) |
| Attr19 | `gp_to_sales` | gross profit / sales | 0.00 | 'gross profit' likely means profit before tax (translation) |
| Attr20 | `inventory_days` | (inventory * 365) / sales | 0.00 |  |
| Attr21 | `sales_growth` | sales (n) / sales (n-1) | 1.78 | needs prior year; often missing |
| Attr22 | `operating_profit_to_assets` | profit on operating activities / total assets | 0.05 |  |
| Attr23 | `net_profit_to_sales` | net profit / sales | 0.00 |  |
| Attr24 | `gp_3y_to_assets` | gross profit (in 3 years) / total assets | 2.22 | 'gross profit' likely means profit before tax (translation); '(in 3 years)' unclear, possibly 3-year cumulative |
| Attr25 | `equity_less_share_capital_to_assets` | (equity - share capital) / total assets | 0.05 |  |
| Attr26 | `net_profit_plus_dep_to_liabilities` | (net profit + depreciation) / total liabilities | 0.24 | cash flow proxy, not true operating cash flow |
| Attr27 | `operating_profit_to_fin_expenses` | profit on operating activities / financial expenses | 6.35 | interest coverage style; undefined when financial expenses are zero |
| Attr28 | `working_capital_to_fixed_assets` | working capital / fixed assets | 1.81 |  |
| Attr29 | `log_total_assets` | logarithm of total assets | 0.05 | size measure, log base not stated |
| Attr30 | `liabilities_less_cash_to_sales` | (total liabilities - cash) / sales | 0.00 |  |
| Attr31 | `gp_plus_interest_to_sales` | (gross profit + interest) / sales | 0.00 | 'gross profit' likely means profit before tax (translation) |
| Attr32 | `current_liabilities_days_of_cogs` | (current liabilities * 365) / cost of products sold | 0.83 |  |
| Attr33 | `opex_to_short_term_liabilities` | operating expenses / short-term liabilities | 0.29 |  |
| Attr34 | `opex_to_liabilities` | operating expenses / total liabilities | 0.24 |  |
| Attr35 | `profit_on_sales_to_assets` | profit on sales / total assets | 0.05 |  |
| Attr36 | `total_sales_to_assets` | total sales / total assets | 0.05 | near duplicate of Attr9 (equal in 58% of rows) |
| Attr37 | `quick_assets_to_long_term_liabilities` | (current assets - inventories) / long-term liabilities | 42.65 | about 43% missing; undefined when no long-term liabilities |
| Attr38 | `constant_capital_to_assets` | constant capital / total assets | 0.05 | 'constant capital' likely equity plus long-term liabilities (translation) |
| Attr39 | `profit_on_sales_to_sales` | profit on sales / sales | 0.00 |  |
| Attr40 | `cash_ratio` | (current assets - inventory - receivables) / short-term liabilities | 0.29 |  |
| Attr41 | `liabilities_to_annualised_op_profit_dep` | total liabilities / ((profit on operating activities + depreciation) * (12/365)) | 1.39 | 12/365 scaling looks odd; sign flips when denominator is negative |
| Attr42 | `operating_profit_to_sales` | profit on operating activities / sales | 0.00 |  |
| Attr43 | `receivables_plus_inventory_days` | rotation receivables + inventory turnover in days | 0.00 |  |
| Attr44 | `receivables_days` | (receivables * 365) / sales | 0.00 |  |
| Attr45 | `net_profit_to_inventory` | net profit / inventory | 4.40 | undefined when inventory is zero |
| Attr46 | `quick_ratio` | (current assets - inventory) / short-term liabilities | 0.29 |  |
| Attr47 | `inventory_days_of_cogs` | (inventory * 365) / cost of products sold | 0.61 |  |
| Attr48 | `ebitda_to_assets` | EBITDA (profit on operating activities - depreciation) / total assets | 0.05 | inconsistent: EBITDA adds depreciation, formula subtracts it |
| Attr49 | `ebitda_to_sales` | EBITDA (profit on operating activities - depreciation) / sales | 0.00 | inconsistent: EBITDA adds depreciation, formula subtracts it |
| Attr50 | `current_assets_to_liabilities` | current assets / total liabilities | 0.24 |  |
| Attr51 | `short_term_liabilities_to_assets` | short-term liabilities / total assets | 0.05 |  |
| Attr52 | `short_term_liabilities_days_of_cogs` | (short-term liabilities * 365) / cost of products sold | 0.64 |  |
| Attr53 | `equity_to_fixed_assets` | equity / fixed assets | 1.81 |  |
| Attr54 | `constant_capital_to_fixed_assets` | constant capital / fixed assets | 1.81 | 'constant capital' likely equity plus long-term liabilities (translation) |
| Attr55 | `working_capital` | working capital | 0.00 | absolute amount, not a ratio; currency and unit not stated |
| Attr56 | `gross_margin` | (sales - cost of products sold) / sales | 0.00 |  |
| Attr57 | `net_quick_assets_to_cash_costs` | (current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation) | 0.05 | 'gross profit' likely means profit before tax (translation) |
| Attr58 | `costs_to_sales` | total costs / total sales | 0.00 |  |
| Attr59 | `long_term_liabilities_to_equity` | long-term liabilities / equity | 0.05 |  |
| Attr60 | `inventory_turnover` | sales / inventory | 4.40 | undefined when inventory is zero |
| Attr61 | `receivables_turnover` | sales / receivables | 0.22 |  |
| Attr62 | `short_term_liabilities_days_of_sales` | (short-term liabilities * 365) / sales | 0.00 |  |
| Attr63 | `sales_to_short_term_liabilities` | sales / short-term liabilities | 0.29 |  |
| Attr64 | `sales_to_fixed_assets` | sales / fixed assets | 1.81 |  |
