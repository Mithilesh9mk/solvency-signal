# US model, validation report

Protocol: docs/us_protocol.md. Fitted on 1999 to 2011, reported on 2012 to 2014. The test years were not loaded. Every number below is written by scripts/run_us_model.py.

## Data

| Split | Firm-years | Companies | Positives | Positive rate | Positives under the provided company label |
|---|---|---|---|---|---|
| Train 1999 to 2011 | 55927 | 7719 | 403 | 0.72% | 4442 |
| Validation 2012 to 2014 | 10473 | 4074 | 87 | 0.83% | 491 |

Companies present in both train and validation: 3290.

Positives by year (derived one-year-ahead label):

| Year | Firm-years | Positives | Rate |
|---|---|---|---|
| 1999 | 5308 | 3 | 0.06% |
| 2000 | 5226 | 7 | 0.13% |
| 2001 | 4897 | 10 | 0.20% |
| 2002 | 4651 | 17 | 0.37% |
| 2003 | 4417 | 29 | 0.66% |
| 2004 | 4348 | 46 | 1.06% |
| 2005 | 4205 | 40 | 0.95% |
| 2006 | 4128 | 51 | 1.24% |
| 2007 | 4009 | 59 | 1.47% |
| 2008 | 3857 | 58 | 1.50% |
| 2009 | 3743 | 23 | 0.61% |
| 2010 | 3625 | 35 | 0.97% |
| 2011 | 3513 | 25 | 0.71% |
| 2012 | 3485 | 26 | 0.75% |
| 2013 | 3484 | 28 | 0.80% |
| 2014 | 3504 | 33 | 0.94% |

## Discrimination on validation (no-skill PR AUC 0.0083, 87 positives in 10473 firm-years)

Intervals resample companies (1000 draws, seed 42).

| Model | PR AUC (95% CI) | ROC AUC (95% CI) | PR AUC on provided company label (diagnostic) |
|---|---|---|---|
| lightgbm | 0.107 (0.064 to 0.183) | 0.893 (0.862 to 0.919) | 0.108 (no skill 0.0469) |
| logistic | 0.082 (0.048 to 0.146) | 0.850 (0.815 to 0.881) | 0.120 (no skill 0.0469) |

Paired difference, lightgbm minus logistic: PR AUC 0.025 (-0.035 to 0.094), ROC AUC 0.043 (0.009 to 0.078).

## Calibration (Platt fitted on train out-of-fold, not refitted)

| Model | Mean predicted | Observed rate | Brier | ECE |
|---|---|---|---|---|
| lightgbm | 0.73% | 0.83% | 0.00789 | 0.00438 |
| logistic | 0.67% | 0.83% | 0.00805 | 0.00262 |

Per validation year, mean predicted against observed:

| Model | Year | Positives | Mean predicted | Observed |
|---|---|---|---|---|
| lightgbm | 2012 | 26 | 0.77% | 0.75% |
| lightgbm | 2013 | 28 | 0.69% | 0.80% |
| lightgbm | 2014 | 33 | 0.73% | 0.94% |
| logistic | 2012 | 26 | 0.69% | 0.75% |
| logistic | 2013 | 28 | 0.64% | 0.80% |
| logistic | 2014 | 33 | 0.67% | 0.94% |

## Flagging the top of the ranking (cuts from train out-of-fold percentiles)

| Model | Flag | Cut | Share flagged on validation | Precision | Recall |
|---|---|---|---|---|---|
| lightgbm | top5 | 2.29% | 4.8% | 7.6% | 43.7% |
| lightgbm | top10 | 1.50% | 10.3% | 4.9% | 60.9% |
| logistic | top5 | 2.43% | 4.1% | 7.4% | 36.8% |
| logistic | top10 | 1.80% | 8.7% | 4.6% | 48.3% |

## Firms seen in train against firms not seen

| Model | Group | Firm-years | Positives | PR AUC |
|---|---|---|---|---|
| lightgbm | seen | 9008 | 83 | 0.118 |
| lightgbm | unseen | 1465 | 4 | n/a |
| logistic | seen | 9008 | 83 | 0.091 |
| logistic | unseen | 1465 | 4 | n/a |

PR AUC is shown only where a group has at least 5 positives.

## Train out-of-fold (company-grouped)

| Model | PR AUC | Mean predicted | Observed |
|---|---|---|---|
| lightgbm | 0.035 | 0.72% | 0.72% |
| logistic | 0.031 | 0.72% | 0.72% |

## Limits

- The 18 variables have no dictionary in the repo and their units look mixed, so they are used raw and not interpreted.
- The label is derived from the README statement that the last fiscal year is the one before the filing.
- Failures cluster in 2002 to 2009, so each period has a different regime and base rate.
- US listed firms only. Not comparable to the Polish figures, because the label, base rate and features differ.
- Nothing here is deployed and no headline model is chosen. That is the user's decision.
