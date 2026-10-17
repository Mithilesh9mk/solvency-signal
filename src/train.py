"""Model training for solvency-signal.

Trains every model in FACTORIES on the Polish 5year splits:
- Logistic regression on the shared preprocessor (train-fitted clipping,
  median imputation with missing indicators, standard scaling).
- LightGBM on the raw ratios. Trees are insensitive to scale and handle
  NaN natively, so no clipping or imputation is applied.
- TremorNet, the neural network (see src/nn.py), added in Phase 3.

Choices kept deliberately plain:
- No class reweighting, so predicted probabilities stay comparable to the
  true bankruptcy rate (matters for the Brier score). Ranking metrics are
  largely unaffected.
- Fixed hyperparameters, no tuning. Validation is never used for early
  stopping, which would leak it into training and bias its scores. The
  neural network early-stops on an inner holdout carved from train.
- 5-fold stratified cross-validation on train only, to show how much the
  scores move between folds. Validation is scored once. Test is not read.

Run: python -m src.train

All randomness uses SEED = 42.
"""

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluate import bootstrap_ci, score
from src.nn import TremorNet
from src.polish_features import FEATURE_NAMES, TARGET
from src.preprocess import PROCESSED, ROOT, build_preprocessor

SEED = 42
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def make_logreg() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)),
    ])


def make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )


FACTORIES = {"logistic_regression": make_logreg, "lightgbm": make_lgbm, "neural_net": TremorNet}


def load_split(name: str, allow_test: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed split. The test split is refused unless explicitly allowed."""
    if name == "test" and not allow_test:
        raise PermissionError("test split is locked until the final model is approved")
    df = pd.read_csv(PROCESSED / f"polish_5year_{name}.csv")
    return df[FEATURE_NAMES], df[TARGET]


def cross_validate(make, X, y, folds: int = 5) -> dict:
    """Stratified k-fold on train: mean and std of each metric."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for tr, te in skf.split(X, y):
        model = make().fit(X.iloc[tr], y.iloc[tr])
        rows.append(score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std())} for k in df.columns}


def main() -> None:
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    MODELS.mkdir(exist_ok=True)
    results, preds = {}, {"bankrupt": y_va.to_numpy()}
    for name, make in FACTORIES.items():
        cv = cross_validate(make, X_tr, y_tr)
        model = make().fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        joblib.dump(model, MODELS / f"{name}.joblib")
        preds[name] = p
        results[name] = {"val": score(y_va, p), "val_ci": bootstrap_ci(y_va, p), "train_cv": cv}
        v = results[name]["val"]
        extra = f"  best epoch {model.best_epoch_}" if hasattr(model, "best_epoch_") else ""
        print(f"{name}: val PR AUC {v['pr_auc']:.3f}  ROC AUC {v['roc_auc']:.3f}  "
              f"Brier {v['brier']:.4f}  recall@5%FPR {v['recall_at_fpr5']:.3f}{extra}")
    (REPORTS / "models_val.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(preds).to_csv(MODELS / "val_predictions.csv", index=False)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
