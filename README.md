# Early Tremor

An early-warning neural network for corporate financial distress.

> Status: Phase 2 (baselines). Scored on validation only; the test split is untouched.

## Problem

TODO: Predict which companies are likely to go bankrupt or enter financial
distress in the following period, early enough for the warning to be useful.

## Data

| Role | Dataset | Notes |
|---|---|---|
| Primary | Polish companies bankruptcy, `5year.arff` (UCI 365, CC BY 4.0) | 5,910 firms, 64 financial ratios, label is bankruptcy within 1 year |
| Candidate | American companies, NYSE and NASDAQ 1999 to 2018 (CC BY 4.0) | 18 raw accounting variables, no operating cash flow; labels need checking |
| Optional benchmark | Taiwanese bankruptcy prediction (UCI) | Most features pre-scaled to 0 to 1, some contain values near 1e10 |

Feature names, formulas, data quality flags and split sizes: [reports/data_report.md](reports/data_report.md).

Citation: Tomczak, S. (2016). Polish Companies Bankruptcy. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600

## Method

TODO: Preprocessing, baselines, neural network architecture, handling of
class imbalance, validation strategy. Fixed random seed 42 throughout.

## Results

Validation split, Polish 5year, 878 firms of which 61 went bankrupt within a year (no-skill PR AUC 0.069).

| Model | PR AUC | ROC AUC | Recall at 5% FPR |
|---|---|---|---|
| Logistic regression | 0.610 | 0.894 | 0.689 |
| LightGBM | 0.783 | 0.957 | 0.787 |

Confidence intervals, ablations and flags: [reports/baselines_val.md](reports/baselines_val.md).
Neural network results will be added in a later phase.

## Limitations

TODO

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Prepare the data (reads the raw zip, writes splits to `data/processed`):

```bash
python -m src.preprocess
```

Run the tests:

```bash
python -m pytest tests
```

Train the baselines and score them on validation, then draw the curves:

```bash
python -m src.train
python -m src.evaluate
```

TODO: Commands for the neural network and the app.

## Live app

TODO: Link to the deployed Streamlit app.

## Author

Mithilesh Kale ([LinkedIn](https://www.linkedin.com/in/mithileshkalemk9))

## Disclaimer

This project is for research and educational purposes only. It is not
financial, investment or credit advice, and its predictions should not be
used as the sole basis for any financial decision.
