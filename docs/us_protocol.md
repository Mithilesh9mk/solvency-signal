# US model protocol (written before any model is fitted)

Separate model on the American NYSE and NASDAQ data (`data/raw/bankruptcy_dataset-main.zip`, 1999 to 2018, 18 raw accounting
variables X1 to X18). Not pooled with the Polish data. No change to the Polish model, calibrator, cuts or app. Seed 42 everywhere.
No figure in any report is typed by hand; every number is written by `scripts/run_us_model.py`.

## 1. Facts found while inspecting the data (before fitting)

- The provided `status_label` is a company-level label. Every row of a failed company is labelled failed (checked: no company has
  mixed labels). It does not say "fails within a year".
- The 18 variables have no data dictionary in the repo. Their meanings are not verified here, so they are used as raw columns only
  and no ratio is built from them. Magnitudes look mixed in scale within a column. Neither the zip nor the repo documents the units.
- Failures cluster in 2002 to 2009, so the three periods differ in base rate and in economic regime.

## 2. Label

Primary label, derived and stated here before results: **one-year-ahead failure**. A firm-year is positive only if the company is
labelled failed and the row is that company's last observed fiscal year. The dataset README says the fiscal year before the filing
is the labelled one, so the last row is the year before the filing. Earlier rows of a failed company are negative. This matches the
Polish one-year horizon and removes the company-identity shortcut that the provided label allows.

Diagnostic only, never a headline: the same models scored against the provided company-level label on validation, to show how much
that label would flatter them.

## 3. Split (by fiscal year, as specified)

- Train: 1999 to 2011. Validation: 2012 to 2014. Test: 2015 to 2018.
- Firms appear in several periods. That is intended, because a forecaster scores firms it has seen before. Firm and period overlap
  counts are reported.
- The derived label needs each company's last observed year, so the loader reads the company, year and status columns of all
  years to build labels. It uses no X value and no label from 2015 onward for fitting, scoring or reporting, and it returns no
  row from those years. A failure in 2015 or later therefore makes the company's earlier rows negative, which is the intended
  one-year-ahead meaning.
- The test years are read once, by `scripts/us_final_test.py`, only after the user approves, with a lock file written first.
  `scripts/run_us_model.py` never loads rows from 2015 onward.

## 4. Features and models (fixed in advance, no tuning)

- Features: X1 to X18 exactly as given. No imputation (the data has no missing values). Industry columns are not used.
- LightGBM: the same settings as the Polish headline (`src/train.make_lgbm`), plus nothing else.
- Logistic regression baseline: signed log transform `sign(x) * log1p(|x|)`, standardised on train, L2 with C = 1.
- No hyperparameter is changed after seeing validation. If LightGBM is worse than logistic regression, that is reported as is.

## 5. Calibration and operating points

- Platt scaling fitted on train out-of-fold raw scores: 5 folds, grouped by company so a company never sits on both sides of a
  fold (`StratifiedGroupKFold`, shuffle, seed 42).
- Validation is never used to fit, recalibrate or cut. The calibrator is not refitted if validation shows drift. Drift is reported.
- No Low, Watch, High bands. The Polish band rule (80% recall, 60% precision) assumes a base rate that this label does not have, and a
  new rule chosen after seeing results would not be pre-registered. Instead report precision and recall when the top 5% and the top
  10% of firm-years by score are flagged, with the score cuts taken from train out-of-fold percentiles and applied to validation. The
  flagged share actually seen on validation is reported next to them.

## 6. What is reported on validation

- PR AUC and ROC AUC for both models, with 95% intervals from a bootstrap that resamples companies (1000 draws, seed 42), and the
  no-skill PR AUC (the positive rate). Paired difference LightGBM minus logistic regression with its interval.
- Calibration: mean predicted against observed rate, Brier score and ECE (10 equal-frequency bins), overall and for each validation
  year.
- Precision and recall at the two flagged shares in section 5.
- Contrast with the provided company-level label (section 2), and performance for firms seen in train against firms not seen in train.
- Counts of rows, companies and positives per split, and per-year positive rates, all computed.

## 7. How the result will be read

- Useful means the lower end of the validation PR AUC interval is above the no-skill level. Anything else is reported as no skill.
- The headline model is the user's decision. The report presents both models and a recommendation, and nothing is deployed.
- Limits stated in the report: unverified variable meanings and units, a different regime in each period, few positives (validation and
  test counts come from code), US listed firms only, and the derived label rests on the README statement about the filing year.

## 8. Out of scope

Pooling with Polish or Taiwanese data (the Taiwanese file is pre-scaled to 0 to 1 and cannot supply real-unit ratios), the Indian
data (no file in the repo), an app tab, and any change to frozen Polish artifacts.
