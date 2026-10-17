"""Feature names for the Polish companies bankruptcy dataset (UCI id 365).

The ARFF files label columns Attr1 to Attr64 only. Formulas are taken from
the UCI description (X1 to X64). Column order was checked against the data
with accounting identities: X17 equals 1 / X2 in every row, and
X40 <= X46 <= X4 holds in 99.6% of rows.

Each entry is (arff_column, short_name, uci_formula, flag). A non-empty flag
marks a definition that is ambiguous, inconsistent or behaves unlike a
plain ratio. Translation note: in the UCI text "gross profit" most likely
means profit before tax (Polish "zysk brutto"), not sales minus cost of
goods sold. Evidence: X19 "gross profit / sales" has Spearman 0.986 with
X23 "net profit / sales" but differs from X56 "(sales - cost of products
sold) / sales" in 96% of rows. This is an inference, not confirmed by UCI.
"""

GP_FLAG = "'gross profit' likely means profit before tax (translation)"

FEATURES = [
    ("Attr1", "net_profit_to_assets", "net profit / total assets", ""),
    ("Attr2", "liabilities_to_assets", "total liabilities / total assets", ""),
    ("Attr3", "working_capital_to_assets", "working capital / total assets", ""),
    ("Attr4", "current_ratio", "current assets / short-term liabilities", ""),
    ("Attr5", "cash_interval_days",
     "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365", ""),
    ("Attr6", "retained_earnings_to_assets", "retained earnings / total assets", ""),
    ("Attr7", "ebit_to_assets", "EBIT / total assets", ""),
    ("Attr8", "book_equity_to_liabilities", "book value of equity / total liabilities", ""),
    ("Attr9", "sales_to_assets", "sales / total assets",
     "near duplicate of Attr36 (equal in 58% of rows)"),
    ("Attr10", "equity_to_assets", "equity / total assets", ""),
    ("Attr11", "gp_extra_finexp_to_assets",
     "(gross profit + extraordinary items + financial expenses) / total assets", GP_FLAG),
    ("Attr12", "gp_to_short_term_liabilities", "gross profit / short-term liabilities", GP_FLAG),
    ("Attr13", "gp_plus_dep_to_sales", "(gross profit + depreciation) / sales", GP_FLAG),
    ("Attr14", "gp_plus_interest_to_assets", "(gross profit + interest) / total assets", GP_FLAG),
    ("Attr15", "liabilities_days_of_gp_dep",
     "(total liabilities * 365) / (gross profit + depreciation)",
     GP_FLAG + "; sign flips when denominator is negative"),
    ("Attr16", "gp_plus_dep_to_liabilities", "(gross profit + depreciation) / total liabilities",
     GP_FLAG + "; cash flow proxy, not true operating cash flow"),
    ("Attr17", "assets_to_liabilities", "total assets / total liabilities", "exact reciprocal of Attr2"),
    ("Attr18", "gp_to_assets", "gross profit / total assets", GP_FLAG),
    ("Attr19", "gp_to_sales", "gross profit / sales", GP_FLAG),
    ("Attr20", "inventory_days", "(inventory * 365) / sales", ""),
    ("Attr21", "sales_growth", "sales (n) / sales (n-1)", "needs prior year; often missing"),
    ("Attr22", "operating_profit_to_assets", "profit on operating activities / total assets", ""),
    ("Attr23", "net_profit_to_sales", "net profit / sales", ""),
    ("Attr24", "gp_3y_to_assets", "gross profit (in 3 years) / total assets",
     GP_FLAG + "; '(in 3 years)' unclear, possibly 3-year cumulative"),
    ("Attr25", "equity_less_share_capital_to_assets", "(equity - share capital) / total assets", ""),
    ("Attr26", "net_profit_plus_dep_to_liabilities", "(net profit + depreciation) / total liabilities",
     "cash flow proxy, not true operating cash flow"),
    ("Attr27", "operating_profit_to_fin_expenses", "profit on operating activities / financial expenses",
     "interest coverage style; undefined when financial expenses are zero"),
    ("Attr28", "working_capital_to_fixed_assets", "working capital / fixed assets", ""),
    ("Attr29", "log_total_assets", "logarithm of total assets", "size measure, log base not stated"),
    ("Attr30", "liabilities_less_cash_to_sales", "(total liabilities - cash) / sales", ""),
    ("Attr31", "gp_plus_interest_to_sales", "(gross profit + interest) / sales", GP_FLAG),
    ("Attr32", "current_liabilities_days_of_cogs", "(current liabilities * 365) / cost of products sold", ""),
    ("Attr33", "opex_to_short_term_liabilities", "operating expenses / short-term liabilities", ""),
    ("Attr34", "opex_to_liabilities", "operating expenses / total liabilities", ""),
    ("Attr35", "profit_on_sales_to_assets", "profit on sales / total assets", ""),
    ("Attr36", "total_sales_to_assets", "total sales / total assets",
     "near duplicate of Attr9 (equal in 58% of rows)"),
    ("Attr37", "quick_assets_to_long_term_liabilities", "(current assets - inventories) / long-term liabilities",
     "about 43% missing; undefined when no long-term liabilities"),
    ("Attr38", "constant_capital_to_assets", "constant capital / total assets",
     "'constant capital' likely equity plus long-term liabilities (translation)"),
    ("Attr39", "profit_on_sales_to_sales", "profit on sales / sales", ""),
    ("Attr40", "cash_ratio", "(current assets - inventory - receivables) / short-term liabilities", ""),
    ("Attr41", "liabilities_to_annualised_op_profit_dep",
     "total liabilities / ((profit on operating activities + depreciation) * (12/365))",
     "12/365 scaling looks odd; sign flips when denominator is negative"),
    ("Attr42", "operating_profit_to_sales", "profit on operating activities / sales", ""),
    ("Attr43", "receivables_plus_inventory_days", "rotation receivables + inventory turnover in days", ""),
    ("Attr44", "receivables_days", "(receivables * 365) / sales", ""),
    ("Attr45", "net_profit_to_inventory", "net profit / inventory", "undefined when inventory is zero"),
    ("Attr46", "quick_ratio", "(current assets - inventory) / short-term liabilities", ""),
    ("Attr47", "inventory_days_of_cogs", "(inventory * 365) / cost of products sold", ""),
    ("Attr48", "ebitda_to_assets", "EBITDA (profit on operating activities - depreciation) / total assets",
     "inconsistent: EBITDA adds depreciation, formula subtracts it"),
    ("Attr49", "ebitda_to_sales", "EBITDA (profit on operating activities - depreciation) / sales",
     "inconsistent: EBITDA adds depreciation, formula subtracts it"),
    ("Attr50", "current_assets_to_liabilities", "current assets / total liabilities", ""),
    ("Attr51", "short_term_liabilities_to_assets", "short-term liabilities / total assets", ""),
    ("Attr52", "short_term_liabilities_days_of_cogs", "(short-term liabilities * 365) / cost of products sold", ""),
    ("Attr53", "equity_to_fixed_assets", "equity / fixed assets", ""),
    ("Attr54", "constant_capital_to_fixed_assets", "constant capital / fixed assets",
     "'constant capital' likely equity plus long-term liabilities (translation)"),
    ("Attr55", "working_capital", "working capital",
     "absolute amount, not a ratio; currency and unit not stated"),
    ("Attr56", "gross_margin", "(sales - cost of products sold) / sales", ""),
    ("Attr57", "net_quick_assets_to_cash_costs",
     "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)", GP_FLAG),
    ("Attr58", "costs_to_sales", "total costs / total sales", ""),
    ("Attr59", "long_term_liabilities_to_equity", "long-term liabilities / equity", ""),
    ("Attr60", "inventory_turnover", "sales / inventory", "undefined when inventory is zero"),
    ("Attr61", "receivables_turnover", "sales / receivables", ""),
    ("Attr62", "short_term_liabilities_days_of_sales", "(short-term liabilities * 365) / sales", ""),
    ("Attr63", "sales_to_short_term_liabilities", "sales / short-term liabilities", ""),
    ("Attr64", "sales_to_fixed_assets", "sales / fixed assets", ""),
]

RENAME = {attr: name for attr, name, _, _ in FEATURES}
FEATURE_NAMES = [name for _, name, _, _ in FEATURES]
TARGET = "bankrupt"
