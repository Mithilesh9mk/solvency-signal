"""American bankruptcy data (1999 to 2018): loading, the one-year-ahead label and the time split.

Protocol: docs/us_protocol.md. The label is derived: a firm-year is positive only if the company is labelled failed and the row is
its last observed fiscal year. Labels need each company's last year, so the loader reads company, year and status for every year,
but `load_period` returns rows only for the requested period and refuses the test years. Only `final_read` returns them, and only
scripts/us_final_test.py calls it.
"""

import zipfile

import numpy as np
import pandas as pd

from src.preprocess import ROOT

RAW_ZIP = ROOT / "data" / "raw" / "bankruptcy_dataset-main.zip"
MEMBER = "bankruptcy_dataset-main/american_bankruptcy_dataset.csv"
FEATURES = [f"X{i}" for i in range(1, 19)]
PERIODS = {"train": (1999, 2011), "val": (2012, 2014), "test": (2015, 2018)}
SEED = 42


def load_labelled() -> pd.DataFrame:
    """Every row with the company id, integer year, provided company-level label (y_company) and derived label (y)."""
    with zipfile.ZipFile(RAW_ZIP) as z:
        df = pd.read_csv(z.open(MEMBER), usecols=["company_name", "fyear", "status_label"] + FEATURES)
    return derive_labels(df)


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds year, y_company (provided, company level) and y (positive only on a failed company's last observed year)."""
    df = df.copy()
    df["year"] = df["fyear"].astype(int)
    df["y_company"] = (df["status_label"] == "failed").astype(int)
    last = df.groupby("company_name")["year"].transform("max")
    df["y"] = ((df["y_company"] == 1) & (df["year"] == last)).astype(int)
    return df.drop(columns=["fyear", "status_label"]).sort_values(["company_name", "year"]).reset_index(drop=True)


def _slice(df: pd.DataFrame, name: str) -> pd.DataFrame:
    lo, hi = PERIODS[name]
    return df[(df["year"] >= lo) & (df["year"] <= hi)].reset_index(drop=True)


def load_period(name: str, df: pd.DataFrame = None) -> pd.DataFrame:
    """Rows of the train or validation years. The test years are not available here."""
    if name not in ("train", "val"):
        raise PermissionError("the test years are read once, by scripts/us_final_test.py")
    return _slice(load_labelled() if df is None else df, name)


def final_read() -> pd.DataFrame:
    """The test years. Called only by scripts/us_final_test.py after it has written its lock file."""
    return _slice(load_labelled(), "test")


def counts(df: pd.DataFrame) -> dict:
    return {"rows": int(len(df)), "companies": int(df["company_name"].nunique()), "positives": int(df["y"].sum()),
            "positive_rate": float(df["y"].mean()), "positives_company_label": int(df["y_company"].sum())}


def per_year(df: pd.DataFrame) -> list:
    g = df.groupby("year")["y"].agg(["size", "sum", "mean"]).reset_index()
    return [{"year": int(r["year"]), "rows": int(r["size"]), "positives": int(r["sum"]), "positive_rate": float(r["mean"])}
            for _, r in g.iterrows()]


def seen_mask(train: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """True for rows of `other` whose company also has rows in train."""
    return other["company_name"].isin(set(train["company_name"])).to_numpy()
