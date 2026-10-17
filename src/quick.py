"""Quick check: a 4-ratio model for solvency-signal (Phase 10).

Pure numbers, no pickle. The fitted model lives in artifacts/quick/quick_model.json: winsor bounds, standardization,
logistic coefficients, 1000 bootstrap coefficient sets (for the range) and the cut points of the 1 to 10 scale.

The four inputs, as the user typed them: current ratio, debt-to-equity, net profit margin and a cash earnings ratio.
The model sees them as dataset attributes Attr4, Attr8 (the reciprocal of debt-to-equity), Attr23 and Attr26.
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUICK_PATH = "artifacts/quick/quick_model.json"

# Model-space features, in the order the coefficients use. Source attribute in the second slot.
FEATURES = [
    ("current_ratio", "Attr4"),
    ("book_equity_to_liabilities", "Attr8"),
    ("net_profit_to_sales", "Attr23"),
    ("net_profit_plus_dep_to_liabilities", "Attr26"),
]
FEATURE_COLS = [f[0] for f in FEATURES]

# What the user types. key, label, one-line definition, whether it is a proxy.
INPUTS = [
    {"key": "current_ratio", "label": "Current ratio", "formula": "current assets / current liabilities",
     "higher": "safer", "proxy": False, "default": 1.5},
    {"key": "debt_to_equity", "label": "Debt-to-equity ratio", "formula": "total liabilities / shareholders' equity",
     "higher": "riskier", "proxy": False, "default": 1.0},
    {"key": "net_profit_margin", "label": "Net profit margin", "formula": "net profit / revenue",
     "higher": "safer", "proxy": False, "default": 0.05},
    {"key": "cash_earnings_ratio", "label": "Cash earnings ratio (proxy for operating cash flow ratio)",
     "formula": "(net profit + depreciation) / total liabilities", "higher": "safer", "proxy": True, "default": 0.15},
]
INPUT_KEYS = [i["key"] for i in INPUTS]

BANDS = [(1, 3, "Low"), (4, 6, "Watch"), (7, 8, "Elevated"), (9, 10, "High")]
CUT_PERCENTILES = list(range(10, 100, 10))
N_BOOT = 1000


class QuickModelError(RuntimeError):
    """The quick-check file is missing or inconsistent, so the page does not score."""


def band_for(score: int) -> str:
    for lo, hi, name in BANDS:
        if lo <= score <= hi:
            return name
    raise ValueError(f"score out of range: {score}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ------------------------------------------------------------------------------------------- transforms
def to_model_space(inputs: dict) -> np.ndarray:
    """User ratios to the model's four features. Debt-to-equity becomes equity / liabilities (its reciprocal).

    Zero debt-to-equity means no debt, so equity / liabilities is infinite and is later clipped to the upper bound.
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


def transform(X, p: dict) -> np.ndarray:
    """Clip to the train 1st to 99th percentile bounds, then standardize with the train mean and sd."""
    X = np.asarray(X, dtype=float)
    return (np.clip(X, p["lo"], p["hi"]) - p["mean"]) / p["sd"]


def predict_proba(X, p: dict) -> np.ndarray:
    return sigmoid(transform(X, p) @ np.asarray(p["coef"]) + p["intercept"])


def score_for(prob, cuts) -> np.ndarray:
    """1 + the number of cut points the probability reaches (a probability equal to a cut counts as above it)."""
    return 1 + np.searchsorted(np.asarray(cuts), np.asarray(prob), side="right")


# --------------------------------------------------------------------------------------------- one firm
def assess(inputs: dict, p: dict) -> dict:
    """Score one firm. Returns probability, its bootstrap range, the 1 to 10 score, the band, per-ratio contributions
    in log-odds and any flags for inputs outside the range the model was fitted on."""
    x = to_model_space(inputs)
    lo, hi = np.asarray(p["lo"]), np.asarray(p["hi"])
    z = transform(x, p)
    coef = np.asarray(p["coef"])
    prob = float(sigmoid(z @ coef + p["intercept"]))
    boot = np.asarray(p["boot"])
    bp = sigmoid(z @ boot[:, :-1].T + boot[:, -1])
    rng = [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))]
    score = int(score_for(prob, p["cuts"]))
    flags = []
    for j, (name, _attr) in enumerate(FEATURES):
        if x[j] < lo[j] or x[j] > hi[j]:
            flags.append(name)
    contrib = z * coef
    return {"probability": prob, "range": rng, "score": score, "band": band_for(score),
            "contributions": {name: float(contrib[j]) for j, (name, _a) in enumerate(FEATURES)},
            "outside_range": flags, "baseline_logodds": float(p["intercept"])}


def input_bounds(p: dict) -> dict:
    """Training range of each input in the units the user types (for hints)."""
    lo, hi = p["lo"], p["hi"]
    out = {"current_ratio": [lo[0], hi[0]], "net_profit_margin": [lo[2], hi[2]], "cash_earnings_ratio": [lo[3], hi[3]]}
    # equity / liabilities in [lo[1], hi[1]]; debt-to-equity is its reciprocal where equity is positive
    out["debt_to_equity"] = [1.0 / hi[1] if hi[1] > 0 else None, 1.0 / lo[1] if lo[1] > 0 else None]
    out["equity_to_liabilities"] = [lo[1], hi[1]]
    return out


# ---------------------------------------------------------------------------------------------- loading
def validate(p: dict) -> None:
    need = ("features", "lo", "hi", "mean", "sd", "coef", "intercept", "boot", "cuts", "n_train")
    miss = [k for k in need if k not in p]
    if miss:
        raise QuickModelError(f"quick model file is missing: {miss}")
    if list(p["features"]) != FEATURE_COLS:
        raise QuickModelError("quick model features do not match the code")
    if len(p["cuts"]) != 9 or any(b <= a for a, b in zip(p["cuts"], p["cuts"][1:])):
        raise QuickModelError("cut points must be 9 increasing numbers")
    if any(c >= 0 for c in p["coef"]):
        raise QuickModelError("a coefficient has the wrong sign (every ratio is expected to lower risk as it rises)")
    if any(s <= 0 for s in p["sd"]):
        raise QuickModelError("a standard deviation is not positive")
    if len(p["boot"]) < 100 or any(len(r) != len(p["coef"]) + 1 for r in p["boot"]):
        raise QuickModelError("bootstrap coefficients are the wrong shape")


def load(root: Path = ROOT) -> dict:
    path = Path(root) / QUICK_PATH
    try:
        p = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QuickModelError(f"quick model file is not readable: {path.name}") from e
    validate(p)
    for k in ("lo", "hi", "mean", "sd", "coef"):
        p[k] = np.asarray(p[k], dtype=float)
    return p
