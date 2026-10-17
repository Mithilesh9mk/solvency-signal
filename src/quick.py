"""Quick check: a 4-ratio model for solvency-signal (Phase 10).

A small monotone LightGBM on four ratios, Platt-calibrated, with a 1 to 10 score. The fitted model lives in
artifacts/quick/quick_model.json as a LightGBM text dump plus plain numbers (bounds, Platt coefficients, cut points and the
observed bankruptcy rate per score). No pickle.

The four inputs, as the user typed them: current ratio, debt-to-equity, net profit margin and a cash earnings ratio.
The model sees them as dataset attributes Attr4, Attr8 (the reciprocal of debt-to-equity), Attr23 and Attr26.
"""

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUICK_PATH = "artifacts/quick/quick_model.json"

# Model-space features, in the order the model uses. Source attribute in the second slot.
FEATURES = [
    ("current_ratio", "Attr4"),
    ("book_equity_to_liabilities", "Attr8"),
    ("net_profit_to_sales", "Attr23"),
    ("net_profit_plus_dep_to_liabilities", "Attr26"),
]
FEATURE_COLS = [f[0] for f in FEATURES]
# What the user types. key, label, one-line definition, proxy flag, example start value.
INPUTS = [
    {"key": "current_ratio", "label": "Current ratio", "formula": "current assets / current liabilities",
     "higher": "safer", "proxy": False, "default": 1.5, "feature": "current_ratio"},
    {"key": "debt_to_equity", "label": "Debt-to-equity ratio", "formula": "total liabilities / shareholders' equity",
     "higher": "riskier", "proxy": False, "default": 1.0, "feature": "book_equity_to_liabilities"},
    {"key": "net_profit_margin", "label": "Net profit margin", "formula": "net profit / revenue",
     "higher": "safer", "proxy": False, "default": 0.05, "feature": "net_profit_to_sales"},
    {"key": "cash_earnings_ratio", "label": "Cash earnings ratio (proxy)",
     "formula": "(net profit + depreciation) / total liabilities", "higher": "safer", "proxy": True, "default": 0.15,
     "feature": "net_profit_plus_dep_to_liabilities"},
]
INPUT_KEYS = [i["key"] for i in INPUTS]
LABEL_BY_FEATURE = {i["feature"]: i["label"] for i in INPUTS}

BANDS = [(1, 3, "Low"), (4, 6, "Watch"), (7, 8, "Elevated"), (9, 10, "High")]
CUT_PERCENTILES = list(range(10, 100, 10))


class QuickModelError(RuntimeError):
    """The quick-check file is missing or inconsistent, so the page does not score."""


def band_for(score: int) -> str:
    for lo, hi, name in BANDS:
        if lo <= score <= hi:
            return name
    raise ValueError(f"score out of range: {score}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------ input handling
def to_model_space(inputs: dict) -> np.ndarray:
    """User ratios to the model's four features. Debt-to-equity becomes equity / liabilities (its reciprocal).

    Zero debt-to-equity means no debt, so equity / liabilities is infinite and is later clipped to the train maximum.
    A negative debt-to-equity is kept (negative equity) and gives a negative equity / liabilities."""
    vals = {}
    for k in INPUT_KEYS:
        try:
            v = float(inputs[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{k} is missing or not a number")
        if not math.isfinite(v):
            raise ValueError(f"{k} is not a finite number")
        vals[k] = v
    de = vals["debt_to_equity"]
    eq_to_liab = math.inf if de == 0 else 1.0 / de
    return np.array([vals["current_ratio"], eq_to_liab, vals["net_profit_margin"], vals["cash_earnings_ratio"]], dtype=float)


def clip(X, p: dict) -> np.ndarray:
    return np.clip(np.asarray(X, dtype=float), p["vmin"], p["vmax"])


def raw_logodds(X, p: dict) -> np.ndarray:
    """Raw boosted log-odds before Platt scaling."""
    return p["_booster"].predict(clip(np.atleast_2d(X), p), raw_score=True)


def predict_proba(X, p: dict) -> np.ndarray:
    return sigmoid(p["platt"]["a"] * raw_logodds(X, p) + p["platt"]["b"])


def score_for(prob, cuts) -> np.ndarray:
    """1 + the number of cut points the probability reaches (a probability equal to a cut counts as above it)."""
    return 1 + np.searchsorted(np.asarray(cuts), np.asarray(prob), side="right")


# --------------------------------------------------------------------------------------------- one firm
def assess(inputs: dict, p: dict) -> dict:
    """Score one firm: calibrated probability, 1 to 10 score, band, the observed rate for that score, per-ratio
    contributions in calibrated log-odds, and which inputs sit outside the 1st to 99th percentile training range."""
    x = to_model_space(inputs)
    xc = clip(x, p)
    contrib = p["_booster"].predict(xc.reshape(1, -1), pred_contrib=True)[0]
    a, b = p["platt"]["a"], p["platt"]["b"]
    prob = float(sigmoid(a * contrib.sum() + b))
    score = int(score_for(prob, p["cuts"]))
    row = p["score_rates"][score - 1]
    flags = [name for j, (name, _a) in enumerate(FEATURES) if x[j] < p["lo"][j] or x[j] > p["hi"][j]]
    return {"probability": prob, "score": score, "band": band_for(score),
            "observed": {"n": row["n"], "bankrupt": row["bankrupt"], "rate": row["rate"], "range": row["wilson"]},
            "contributions": {name: float(a * contrib[j]) for j, (name, _a) in enumerate(FEATURES)},
            "baseline_logodds": float(a * contrib[-1] + b), "outside_range": flags}


def input_bounds(p: dict) -> dict:
    """Training 1st to 99th percentile of each input in the units the user types (for hints)."""
    lo, hi = p["lo"], p["hi"]
    return {"current_ratio": [lo[0], hi[0]], "net_profit_margin": [lo[2], hi[2]], "cash_earnings_ratio": [lo[3], hi[3]],
            "debt_to_equity": [1.0 / hi[1] if hi[1] > 0 else None, 1.0 / lo[1] if lo[1] > 0 else None]}


# ---------------------------------------------------------------------------------------------- loading
def validate(p: dict) -> None:
    need = ("features", "lo", "hi", "vmin", "vmax", "model_str", "platt", "cuts", "score_rates", "n_train")
    miss = [k for k in need if k not in p]
    if miss:
        raise QuickModelError(f"quick model file is missing: {miss}")
    if list(p["features"]) != FEATURE_COLS:
        raise QuickModelError("quick model features do not match the code")
    if len(p["cuts"]) != 9 or any(b <= a for a, b in zip(p["cuts"], p["cuts"][1:])):
        raise QuickModelError("cut points must be 9 increasing numbers")
    if p["platt"]["a"] <= 0:
        raise QuickModelError("the Platt slope must be positive")
    if len(p["score_rates"]) != 10 or any(r["n"] <= 0 for r in p["score_rates"]):
        raise QuickModelError("every score from 1 to 10 needs observed firms")
    for k in ("lo", "hi", "vmin", "vmax"):
        if len(p[k]) != 4:
            raise QuickModelError(f"{k} must have four values")


def build(p: dict) -> dict:
    """Attach the booster and arrays to a parameter dict."""
    p = dict(p)
    for k in ("lo", "hi", "vmin", "vmax"):
        p[k] = np.asarray(p[k], dtype=float)
    try:
        p["_booster"] = lgb.Booster(model_str=p["model_str"])
    except lgb.basic.LightGBMError as e:
        raise QuickModelError("the quick model text could not be loaded") from e
    return p


def load(root: Path = ROOT) -> dict:
    path = Path(root) / QUICK_PATH
    try:
        p = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QuickModelError(f"quick model file is not readable: {path.name}") from e
    validate(p)
    return build(p)
