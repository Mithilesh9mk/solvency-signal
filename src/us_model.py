"""US models, calibration and company-clustered evaluation (protocol: docs/us_protocol.md).

Two models with settings fixed in advance: LightGBM (the Polish headline settings) and a logistic baseline on a signed-log
transform. Platt scaling is fitted on train out-of-fold raw scores with folds grouped by company. Nothing here reads the test years.
"""

import json

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src import calibrate as C
from src.train import make_lgbm
from src.us_data import FEATURES

SEED = 42
FOLDS = 5
N_BOOT = 1000
FLAG_SHARES = (0.05, 0.10)


def slog(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


class LogitModel:
    """Signed log, standardise on the fitting data, L2 logistic regression with C = 1."""

    def fit(self, X, y):
        Z = slog(X)
        self.mean, self.sd = Z.mean(axis=0), Z.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit((Z - self.mean) / self.sd, y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba((slog(X) - self.mean) / self.sd)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "sd": self.sd.tolist(), "coef": self.lr.coef_[0].tolist(),
                "intercept": float(self.lr.intercept_[0])}

    @classmethod
    def from_dict(cls, d: dict):
        m = cls()
        m.mean, m.sd = np.array(d["mean"]), np.array(d["sd"])
        m.lr = LogisticRegression(C=1.0)
        m.lr.classes_ = np.array([0, 1])
        m.lr.coef_, m.lr.intercept_ = np.array([d["coef"]]), np.array([d["intercept"]])
        return m


MAKERS = {"lightgbm": make_lgbm, "logistic": LogitModel}


def oof_scores(name: str, X: np.ndarray, y: np.ndarray, groups) -> np.ndarray:
    """Raw out-of-fold probabilities, folds grouped by company and stratified on the label."""
    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        oof[te] = MAKERS[name]().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def fit_all(name: str, X: np.ndarray, y: np.ndarray, groups) -> dict:
    """Final model on all train rows, Platt on train out-of-fold scores, flag cuts from train out-of-fold percentiles."""
    oof = oof_scores(name, X, y, groups)
    platt = C.PlattCalibrator().fit(oof, y)
    p_oof = platt.transform(oof)
    cuts = {f"top{int(s * 100)}": float(np.quantile(p_oof, 1 - s)) for s in FLAG_SHARES}
    return {"model": MAKERS[name]().fit(X, y), "platt": platt, "cuts": cuts, "oof_raw": oof, "oof_cal": p_oof}


def predict(fit: dict, X: np.ndarray) -> tuple:
    raw = fit["model"].predict_proba(X)[:, 1]
    return raw, fit["platt"].transform(raw)


# ------------------------------------------------------------------------------- evaluation
def cluster_indices(groups, n: int = N_BOOT, seed: int = SEED) -> list:
    """Bootstrap resamples that draw whole companies, shared by every model so differences are paired."""
    groups = np.asarray(groups)
    codes, uniq = np.unique(groups, return_inverse=True)[1], np.unique(groups)
    rows_of = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([rows_of[i] for i in pick]))
    return out


def _safe(fn, y, p):
    return float(fn(y, p)) if 0 < y.sum() < len(y) else np.nan


def boot_ci(y, scores: dict, idx: list) -> dict:
    """95% intervals for PR AUC and ROC AUC per score vector, plus the paired difference of the first two."""
    y = np.asarray(y)
    names = list(scores)
    draws = {n: {"pr_auc": [], "roc_auc": []} for n in names}
    for ix in idx:
        for n in names:
            draws[n]["pr_auc"].append(_safe(average_precision_score, y[ix], scores[n][ix]))
            draws[n]["roc_auc"].append(_safe(roc_auc_score, y[ix], scores[n][ix]))
    q = [2.5, 97.5]
    out = {n: {k: np.nanpercentile(v, q).tolist() for k, v in draws[n].items()} for n in names}
    if len(names) >= 2:
        a, b = names[0], names[1]
        out["diff"] = {k: np.nanpercentile(np.array(draws[a][k]) - np.array(draws[b][k]), q).tolist() for k in ("pr_auc", "roc_auc")}
    return out


def calibration(y, p) -> dict:
    return {"mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y)), "brier": C.metrics(y, p)["brier"],
            "ece": C.ece(y, p)}


def flagged(y, p, cut: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    m = p >= cut
    tp = int(y[m].sum())
    return {"cut": float(cut), "share_flagged": float(m.mean()), "flagged": int(m.sum()), "true_positives": tp,
            "precision": tp / m.sum() if m.sum() else None, "recall": tp / max(int(y.sum()), 1)}


def evaluate(y, groups, years, raw: dict, cal: dict, cuts: dict, idx: list, y_company=None) -> dict:
    """Everything the protocol lists for one evaluation set. raw and cal map model name to score vectors."""
    y, years = np.asarray(y), np.asarray(years)
    res = {"no_skill_pr_auc": float(y.mean()), "positives": int(y.sum()), "rows": int(len(y)), "models": {}}
    ci = boot_ci(y, raw, idx)
    for n in raw:
        r = {"pr_auc": float(average_precision_score(y, raw[n])), "roc_auc": float(roc_auc_score(y, raw[n])),
             "pr_auc_ci": ci[n]["pr_auc"], "roc_auc_ci": ci[n]["roc_auc"],
             "calibration": calibration(y, cal[n]),
             "by_year": [{"year": int(t), "positives": int(y[years == t].sum()), **{k: v for k, v in calibration(y[years == t], cal[n][years == t]).items()
                                                                                  if k in ("mean_predicted", "observed_rate")}}
                         for t in sorted(set(years))],
             "flag": {k: flagged(y, cal[n], cuts[n][k]) for k in cuts[n]}}
        if y_company is not None:
            r["pr_auc_company_label"] = float(average_precision_score(np.asarray(y_company), raw[n]))
            r["no_skill_company_label"] = float(np.mean(y_company))
        res["models"][n] = r
    if "diff" in ci:
        a, b = list(raw)[:2]
        res["paired_difference"] = {"a": a, "b": b, "pr_auc": float(res["models"][a]["pr_auc"] - res["models"][b]["pr_auc"]),
                                    "pr_auc_ci": ci["diff"]["pr_auc"], "roc_auc": float(res["models"][a]["roc_auc"] - res["models"][b]["roc_auc"]),
                                    "roc_auc_ci": ci["diff"]["roc_auc"]}
    return res


def subgroup_pr(y, score, mask, min_pos: int = 5) -> dict:
    y, score, mask = np.asarray(y), np.asarray(score), np.asarray(mask)
    yy = y[mask]
    return {"rows": int(mask.sum()), "positives": int(yy.sum()),
            "pr_auc": float(average_precision_score(yy, score[mask])) if yy.sum() >= min_pos else None}


# ------------------------------------------------------------------------------- artifact
def to_artifact(fits: dict, meta: dict) -> dict:
    out = {"features": FEATURES, "meta": meta, "models": {}}
    for n, f in fits.items():
        m = {"platt": {"a": f["platt"].a, "b": f["platt"].b}, "cuts": f["cuts"]}
        if n == "lightgbm":
            m["model_str"] = f["model"].booster_.model_to_string()
        else:
            m["logit"] = f["model"].to_dict()
        out["models"][n] = m
    return out


def predict_from_artifact(art: dict, name: str, X: np.ndarray) -> tuple:
    m = art["models"][name]
    if name == "lightgbm":
        booster = lgb.Booster(model_str=m["model_str"])
        raw = booster.predict(np.asarray(X, dtype=float))
    else:
        raw = LogitModel.from_dict(m["logit"]).predict_proba(X)[:, 1]
    return raw, C.PlattCalibrator(m["platt"]["a"], m["platt"]["b"]).transform(raw)


def save_artifact(art: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art), encoding="utf-8")
