"""Probability calibration, risk bands and cost thresholds for solvency-signal.

Design (rules fixed before looking at results, see docs/decisions.md):
- Calibrators are fitted on 5-fold out-of-fold (OOF) predictions on the
  TRAIN split, so every positive (286) is used and no firm is scored by a model
  that saw it. Validation is only used to report, never to fit a calibrator or
  choose a threshold.
- Methods: none, Platt (sigmoid on the logit of the score), isotonic.
- Method per model: lowest cross-validated log loss on the OOF scores, ties
  within one standard error resolved toward the simpler method.
- Bands on the calibrated probability, rule fixed in docs/decisions.md and
  fitted on validation by user instruction: Watch is the highest threshold with
  recall >= 0.80, High is the lowest threshold with precision >= 0.60. If the
  cuts collide there are two bands, Low and High.
- ECE uses 10 equal-frequency bins.

Calibrators are plain parameter sets (JSON), so they load without refitting.
The test split is never read here. All randomness uses SEED = 42.
"""

import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

SEED = 42
FOLDS = 5
ECE_BINS = 10
LL_EPS = 1e-4  # log loss clips probabilities to [LL_EPS, 1 - LL_EPS]
SCORE_EPS = 1e-6  # clip before the logit in Platt
METHODS = ("none", "platt", "isotonic")  # ordered simplest first
BAND_RULE = {"high_min_precision": 0.60, "watch_min_recall": 0.80}  # fitted on validation, see docs/decisions.md
PRECISION_FLOORS = (0.25, 0.35, 0.50, 0.60, 0.70)
COST_RATIOS = (5, 10, 20)  # assumptions: missed distress cost : false alarm cost
BANDS = ("Low", "Watch", "High")


# ---------------------------------------------------------------- calibrators
class Calibrator:
    method = "base"

    def fit(self, scores, y):
        raise NotImplementedError

    def transform(self, scores):
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError


class NoCalibration(Calibrator):
    method = "none"

    def fit(self, scores, y):
        return self

    def transform(self, scores):
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)

    def to_dict(self):
        return {"method": self.method}


class PlattCalibrator(Calibrator):
    """p = sigmoid(a * logit(score) + b), fitted by (almost) unpenalised logistic regression."""

    method = "platt"

    def __init__(self, a=1.0, b=0.0):
        self.a, self.b = a, b

    @staticmethod
    def _logit(scores):
        s = np.clip(np.asarray(scores, dtype=float), SCORE_EPS, 1 - SCORE_EPS)
        return np.log(s / (1 - s))

    def fit(self, scores, y):
        lr = LogisticRegression(C=1e6, max_iter=1000, random_state=SEED)
        lr.fit(self._logit(scores).reshape(-1, 1), np.asarray(y))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, scores):
        return 1.0 / (1.0 + np.exp(-(self.a * self._logit(scores) + self.b)))

    def to_dict(self):
        return {"method": self.method, "a": self.a, "b": self.b}


class IsotonicCalibrator(Calibrator):
    """Non-decreasing step-like map stored as knots; np.interp matches sklearn's transform."""

    method = "isotonic"

    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.y = None if y is None else np.asarray(y, dtype=float)

    def fit(self, scores, y):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        iso.fit(np.asarray(scores, dtype=float), np.asarray(y))
        self.x, self.y = iso.X_thresholds_.copy(), iso.y_thresholds_.copy()
        return self

    def transform(self, scores):
        return np.interp(np.asarray(scores, dtype=float), self.x, self.y)

    def to_dict(self):
        return {"method": self.method, "x": self.x.tolist(), "y": self.y.tolist()}


def make_calibrator(method: str) -> Calibrator:
    return {"none": NoCalibration, "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}[method]()


def calibrator_from_dict(d: dict) -> Calibrator:
    if d["method"] == "none":
        return NoCalibration()
    if d["method"] == "platt":
        return PlattCalibrator(d["a"], d["b"])
    if d["method"] == "isotonic":
        return IsotonicCalibrator(d["x"], d["y"])
    raise ValueError(f"unknown calibration method: {d['method']}")


# --------------------------------------------------------------- OOF and selection
def oof_predictions(make, X, y, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold probabilities. Same folds as train.cross_validate."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        model = make().fit(X.iloc[tr], y.iloc[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def log_loss_clipped(y, p) -> float:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), LL_EPS, 1 - LL_EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error with equal-frequency bins (ties broken by row order)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    total = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            total += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(total)


def reliability_points(y, p, n_bins: int = ECE_BINS) -> list:
    """(mean predicted, observed rate, count) per equal-frequency bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    return [(float(p[i].mean()), float(y[i].mean()), int(len(i))) for i in np.array_split(order, n_bins) if len(i)]


def metrics(y, p) -> dict:
    y = np.asarray(y)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": log_loss_clipped(y, p),
            "ece": ece(y, p), "pr_auc": float(average_precision_score(y, p)),
            "mean_pred": float(np.mean(p)), "observed_rate": float(np.mean(y))}


def select_method(oof, y, folds: int = FOLDS, seed: int = SEED) -> dict:
    """Cross-validated log loss of each method on the OOF scores (train only)."""
    oof, y = np.asarray(oof), np.asarray(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    per = {m: {"log_loss": [], "brier": []} for m in METHODS}
    for tr, te in skf.split(oof.reshape(-1, 1), y):
        for m in METHODS:
            p = make_calibrator(m).fit(oof[tr], y[tr]).transform(oof[te])
            per[m]["log_loss"].append(log_loss_clipped(y[te], p))
            per[m]["brier"].append(float(brier_score_loss(y[te], p)))
    table = {m: {"log_loss_mean": float(np.mean(v["log_loss"])),
                 "log_loss_se": float(np.std(v["log_loss"], ddof=1) / np.sqrt(folds)),
                 "brier_mean": float(np.mean(v["brier"]))} for m, v in per.items()}
    best = min(METHODS, key=lambda m: table[m]["log_loss_mean"])
    limit = table[best]["log_loss_mean"] + table[best]["log_loss_se"]
    chosen = next(m for m in METHODS if table[m]["log_loss_mean"] <= limit)  # simplest within 1 SE
    return {"table": table, "best_by_mean": best, "selected": chosen, "one_se_limit": float(limit)}


# --------------------------------------------------------------------- bootstrap
def bootstrap_indices(y, n: int = 1000, seed: int = SEED) -> list:
    """Stratified resamples, shared across models and methods so differences are paired."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)]


def bootstrap_metrics(y, preds: dict, indices: list, alpha: float = 0.05) -> dict:
    """95% percentile CIs per method for each metric, plus paired differences vs 'none'."""
    y = np.asarray(y)
    keys = ("brier", "log_loss", "ece", "pr_auc")
    draws = {m: {k: [] for k in keys} for m in preds}
    for idx in indices:
        for m, p in preds.items():
            r = metrics(y[idx], p[idx])
            for k in keys:
                draws[m][k].append(r[k])
    q = [alpha / 2 * 100, (1 - alpha / 2) * 100]
    out = {}
    for m in preds:
        out[m] = {"ci": {k: np.percentile(draws[m][k], q).tolist() for k in keys}}
        if m != "none" and "none" in preds:
            out[m]["delta_vs_none"] = {
                k: {"ci": np.percentile(np.array(draws[m][k]) - np.array(draws["none"][k]), q).tolist()}
                for k in keys}
    return out


# ------------------------------------------------------------- thresholds and bands
def pr_points(y, s) -> dict:
    """Precision, recall and flagged count at every distinct threshold (flag score >= t)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy)
    n = np.arange(1, len(s) + 1)
    ends = np.append(ss[:-1] != ss[1:], True)  # last row of each tie group
    return {"threshold": ss[ends], "flagged": n[ends], "tp": tp[ends],
            "precision": tp[ends] / n[ends], "recall": tp[ends] / max(int(y.sum()), 1)}


def band_thresholds(y, s, high_min_precision: float = BAND_RULE["high_min_precision"],
                    watch_min_recall: float = BAND_RULE["watch_min_recall"]) -> dict:
    """Apply the pre-stated band rule to (y, calibrated scores), here the validation split."""
    pts = pr_points(y, s)
    notes = []
    ok_high = pts["precision"] >= high_min_precision
    if ok_high.any():
        t_high = float(pts["threshold"][ok_high].min())  # lowest threshold meeting the precision floor
    else:
        t_high = float(pts["threshold"].max()) + 1e-12
        notes.append("no threshold reaches the precision floor; High band is empty")
    ok_watch = pts["recall"] >= watch_min_recall
    t_watch = float(pts["threshold"][ok_watch].max())  # highest threshold still reaching the recall floor
    if t_watch >= t_high:
        notes.append("the 80% recall cut is not below the precision cut, so there are only two bands, Low and High; "
                     "the single cut is the 80% recall cut and no Watch band is forced")
        t_high = t_watch
    return {"t_watch": t_watch, "t_high": t_high, "rule": dict(BAND_RULE), "notes": notes}


def precision_floor_diagnostic(y, s, floors=PRECISION_FLOORS) -> list:
    """For each precision floor: the lowest threshold meeting it, and the OOF recall and share flagged there."""
    pts = pr_points(y, s)
    rows = []
    for f in floors:
        ok = pts["precision"] >= f
        if not ok.any():
            rows.append({"precision_floor": f, "threshold": None, "recall": None, "share_flagged": None})
            continue
        i = int(np.flatnonzero(ok)[-1])  # arrays run from high to low threshold; last ok = lowest threshold
        rows.append({"precision_floor": f, "threshold": float(pts["threshold"][i]),
                     "recall": float(pts["recall"][i]), "share_flagged": float(pts["flagged"][i] / len(s))})
    return rows


def assign_bands(p, t_watch: float, t_high: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p >= t_high, "High", np.where(p >= t_watch, "Watch", "Low"))


def band_table(y, p, t_watch: float, t_high: float) -> list:
    y = np.asarray(y)
    band = assign_bands(p, t_watch, t_high)
    rows = []
    for b in BANDS:
        m = band == b
        pos = int(y[m].sum())
        rows.append({"band": b, "firms": int(m.sum()), "share_of_firms": float(m.mean()),
                     "bankrupt": pos, "bankrupt_rate": pos / m.sum() if m.sum() else None,
                     "share_of_all_bankrupt": pos / max(int(y.sum()), 1)})
    return rows


def tradeoff_rows(y, p, grid=None) -> list:
    """Precision-recall table on a fixed grid of calibrated-probability cutoffs."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    grid = grid if grid is not None else [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
                                         0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []
    for t in grid:
        f = p >= t
        tp = int((f & (y == 1)).sum())
        rows.append({"threshold": t, "flagged": int(f.sum()), "share_flagged": float(f.mean()), "tp": tp,
                     "precision": tp / f.sum() if f.sum() else None, "recall": tp / max(int(y.sum()), 1)})
    return rows


def cost_thresholds(y_fit, s_fit, y_eval, s_eval, ratio: float) -> dict:
    """Cost = ratio * missed distress + 1 * false alarm. Threshold chosen on the fit data
    (OOF train); evaluated on the eval data. Also the theoretical cut 1 / (1 + ratio),
    optimal if the probabilities were perfectly calibrated."""
    pts = pr_points(y_fit, s_fit)
    pos = int(np.sum(y_fit))
    fn = pos - pts["tp"]
    fp = pts["flagged"] - pts["tp"]
    cost = ratio * fn + fp
    flag_none = ratio * pos
    best = int(np.flatnonzero(cost == cost.min())[-1])  # ties: highest threshold
    t_emp = float(pts["threshold"][best]) if cost[best] < flag_none else float(pts["threshold"].max()) + 1e-12
    t_theory = 1.0 / (1.0 + ratio)
    out = {"ratio": ratio, "t_theory": t_theory, "t_empirical": t_emp}
    ye, se = np.asarray(y_eval), np.asarray(s_eval, dtype=float)
    for name, t in (("theory", t_theory), ("empirical", t_emp)):
        f = se >= t
        tp = int((f & (ye == 1)).sum())
        miss = int(ye.sum()) - tp
        fa = int(f.sum()) - tp
        out[name] = {"threshold": t, "flagged": int(f.sum()), "recall": tp / max(int(ye.sum()), 1),
                     "precision": tp / f.sum() if f.sum() else None,
                     "cost_per_1000": 1000 * (ratio * miss + fa) / len(ye)}
    out["flag_nobody_cost_per_1000"] = 1000 * ratio * int(ye.sum()) / len(ye)
    out["flag_everybody_cost_per_1000"] = 1000 * (len(ye) - int(ye.sum())) / len(ye)
    return out


# ------------------------------------------------------------------- artifacts
def save_artifacts(out_dir, calibrators: dict, bands: dict, config: dict) -> None:
    """calibrators: {model: {method: Calibrator}}, bands: {model: {...}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibrators.json").write_text(json.dumps(
        {m: {meth: c.to_dict() for meth, c in d.items()} for m, d in calibrators.items()}))
    (out_dir / "bands.json").write_text(json.dumps(bands, indent=1))
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))


def load_artifacts(out_dir) -> dict:
    """Load saved calibrators and bands. Nothing is refitted."""
    cal = json.loads((out_dir / "calibrators.json").read_text())
    return {"calibrators": {m: {meth: calibrator_from_dict(d) for meth, d in dd.items()} for m, dd in cal.items()},
            "bands": json.loads((out_dir / "bands.json").read_text()),
            "config": json.loads((out_dir / "config.json").read_text())}


def risk_band(model: str, raw_scores, artifacts: dict) -> tuple:
    """Calibrated probability and band for raw model scores, from saved artifacts."""
    b = artifacts["bands"][model]
    p = artifacts["calibrators"][model][b["selected_method"]].transform(raw_scores)
    return p, assign_bands(p, b["t_watch"], b["t_high"])
