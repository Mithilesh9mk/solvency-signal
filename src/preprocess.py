"""Data preprocessing for Early Tremor.

Primary dataset: Polish companies bankruptcy data (UCI id 365), file
5year.arff. Ratios come from the last year of the forecast period and the
label is bankruptcy within the following year.

Steps:
1. Read 5year.arff straight from the raw zip in memory. data/raw is never
   modified or extracted into.
2. Rename Attr1..Attr64 to readable names (see polish_features.py) and cast
   the label to int.
3. Drop exact duplicate rows before splitting, so copies cannot land in
   both train and test.
4. Stratified train / validation / test split (70 / 15 / 15, seed 42).
   The data has no company id or year, so a time-based split is impossible.
5. Write the splits to data/processed with missing values left in place.

Anything learned from data (clipping bounds, imputation medians, scaling)
lives in build_preprocessor() and must be fitted on the training split only.

Run: python -m src.preprocess

All randomness uses SEED = 42.
"""

import io
import zipfile
from pathlib import Path

import arff
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.polish_features import FEATURE_NAMES, RENAME, TARGET

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "polish+companies+bankruptcy+data.zip"
PROCESSED = ROOT / "data" / "processed"
SPLIT_SIZES = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_polish(year: int = 5, zip_path: Path = RAW_ZIP) -> pd.DataFrame:
    """Load one Polish ARFF file from the raw zip with readable column names."""
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(f"{year}year.arff").decode("utf-8")
    data = arff.loads(text)
    cols = [a[0] for a in data["attributes"]]
    df = pd.DataFrame(data["data"], columns=cols)
    df = df.rename(columns=RENAME).rename(columns={"class": TARGET})
    df[TARGET] = df[TARGET].astype(int)
    df[FEATURE_NAMES] = df[FEATURE_NAMES].astype(float)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and replace infinities with NaN."""
    df = df.drop_duplicates().reset_index(drop=True)
    df[FEATURE_NAMES] = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    return df


def split(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Stratified 70 / 15 / 15 split on the target."""
    train, rest = train_test_split(
        df, test_size=1 - SPLIT_SIZES["train"], stratify=df[TARGET], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest[TARGET], random_state=seed
    )
    return {"train": train, "val": val, "test": test}


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data.

    The Polish ratios have extreme tails (divisions by near-zero values), so
    clipping keeps a few rows from dominating a neural network. NaN passes
    through untouched.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def build_preprocessor() -> Pipeline:
    """Unfitted transform: clip tails, median impute with missing flags, scale."""
    return Pipeline([
        ("clip", QuantileClipper(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])


def main() -> None:
    raw = load_polish(5)
    df = clean(raw)
    parts = split(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_csv(PROCESSED / f"polish_5year_{name}.csv", index=False)
    print(f"raw rows {len(raw)}, after dedup {len(df)}")
    for name, part in parts.items():
        print(f"{name}: {len(part)} rows, {int(part[TARGET].sum())} bankrupt "
              f"({part[TARGET].mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
