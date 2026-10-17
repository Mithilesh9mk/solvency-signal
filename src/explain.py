"""SHAP explanations for the frozen LightGBM + Platt model (train and validation only).

SHAP is computed in the model's raw log-odds. Platt is sigmoid(a * logit(score) + b), so when the
calibrator was fitted on raw log-odds, calibrated log-odds are a * raw + b and SHAP values convert
by multiplying by a. `check_slope_conversion` verifies that; if it fails the caller must report the
raw scale. Nothing here reads the test split. All randomness uses SEED = 42.
"""

import re
import warnings

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src import calibrate as C
from src.polish_features import CATEGORIES, CATEGORY, FEATURES

SEED = 42
TOL = 1e-6
# A dictionary flag marks a definition UNVERIFIED if it says the definition is a translation guess,
# unclear, odd, inconsistent or not stated. Quality notes (missing, duplicate, undefined) are not.
UNVERIFIED_RE = re.compile(r"translation|unclear|\bodd\b|inconsistent|not stated")
SHORT = 46
# Features whose category assignment is a judgement call (see src/polish_features.py).
BORDERLINE = ("Attr27", "Attr30", "Attr33", "Attr34", "Attr37", "Attr58")


def feature_dictionary() -> list:
    rows = []
    for attr, name, formula, flag in FEATURES:
        unverified = bool(UNVERIFIED_RE.search(flag))
        short = formula if len(formula) <= SHORT else formula[:SHORT - 3].rstrip() + "..."
        rows.append({"attr": attr, "name": name, "formula": formula, "category": CATEGORY[attr],
                     "unverified": unverified, "note": flag,
                     "label": f"{attr}: {formula}" + (f" [UNVERIFIED {attr}: {flag}]" if unverified else ""),
                     "short_label": f"{attr} {short}" + (" [U]" if unverified else "")})
    return rows


# ---------------------------------------------------------------------------------- SHAP
def shap_values(model, X) -> tuple:
    """TreeSHAP in raw log-odds. Returns (phi, base) with phi of shape (rows, features)."""
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        phi = explainer.shap_values(X)
    if isinstance(phi, list):
        phi = phi[1]
    base = np.ravel(explainer.expected_value)[-1]
    return np.asarray(phi, dtype=float), float(base)


def native_contrib(model, X) -> tuple:
    """LightGBM's own exact TreeSHAP, used to cross-check the shap library."""
    out = model.predict(X, pred_contrib=True)
    return out[:, :-1], float(out[0, -1])


def check_slope_conversion(model, X, calibrator, tol: float = TOL) -> dict:
    """Does a * raw_log_odds + b reproduce the saved calibrator's log-odds? ok is False if not."""
    if not isinstance(calibrator, C.PlattCalibrator):
        return {"ok": False, "reason": "calibrator is not Platt", "max_abs_error": None, "clipped": None,
                "tol": tol, "n": int(len(X))}
    raw_p = model.predict_proba(X)[:, 1]
    clipped = int(((raw_p < C.SCORE_EPS) | (raw_p > 1 - C.SCORE_EPS)).sum())
    margin = model.predict(X, raw_score=True)
    p_cal = calibrator.transform(raw_p)
    z_cal = np.log(p_cal / (1 - p_cal))
    err = float(np.max(np.abs(calibrator.a * margin + calibrator.b - z_cal)))
    ok = bool(clipped == 0 and err <= tol)
    return {"ok": ok, "reason": "" if ok else "converted values do not reproduce calibrated log-odds",
            "max_abs_error": err, "clipped": clipped, "tol": tol, "n": int(len(X))}


def explain_dataset(model, calibrator, X) -> dict:
    """SHAP on the calibrated log-odds scale when the slope check passes, else on the raw scale."""
    phi, base = shap_values(model, X)
    check = check_slope_conversion(model, X, calibrator)
    use_cal = check["ok"]
    a, b = (calibrator.a, calibrator.b) if use_cal else (1.0, 0.0)
    nat_phi, nat_base = native_contrib(model, X)
    margin = model.predict(X, raw_score=True)
    return {"phi": phi * a, "base": base * a + b, "scale": "calibrated_log_odds" if use_cal else "raw_log_odds",
            "slope": a, "check": check,
            "additivity_error_raw": float(np.max(np.abs(base + phi.sum(1) - margin))),
            "max_diff_vs_native": float(max(np.max(np.abs(phi - nat_phi)), abs(base - nat_base)))}


# ------------------------------------------------------------------------- importance
def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def category_index() -> dict:
    """Feature column positions per category, in dictionary order."""
    return {cat: [i for i, f in enumerate(FEATURES) if CATEGORY[f[0]] == cat] for cat in CATEGORIES}


def importance(phi_va, y_va, phi_tr, gain, n_boot: int = 1000, seed: int = SEED) -> dict:
    """Mean |SHAP| per feature and category with stratified bootstrap CIs over validation firms."""
    a_va, a_tr = np.abs(phi_va), np.abs(phi_tr)
    mean_va, mean_tr = a_va.mean(0), a_tr.mean(0)
    cats = category_index()
    idx = C.bootstrap_indices(y_va, n_boot, seed)
    feat_d = np.array([a_va[i].mean(0) for i in idx])
    rank_d = np.array([np.argsort(np.argsort(-d)) + 1 for d in feat_d])
    cat_d = {c: feat_d[:, ix].sum(1) for c, ix in cats.items()}
    q = [2.5, 97.5]
    top10_va, top10_tr = set(np.argsort(-mean_va)[:10]), set(np.argsort(-mean_tr)[:10])
    return {"mean_abs_val": mean_va.tolist(), "mean_abs_train": mean_tr.tolist(),
            "ci_val": np.percentile(feat_d, q, axis=0).T.tolist(),
            "rank_val": (np.argsort(np.argsort(-mean_va)) + 1).tolist(),
            "rank_ci_val": np.percentile(rank_d, q, axis=0).T.tolist(),
            "category": {c: {"mean_abs_sum_val": float(mean_va[ix].sum()), "mean_abs_sum_train": float(mean_tr[ix].sum()),
                             "share_val": float(mean_va[ix].sum() / mean_va.sum()), "n_features": len(ix),
                             "ci_val": np.percentile(cat_d[c], q).tolist()} for c, ix in cats.items()},
            "spearman_train_vs_val": float(spearmanr(mean_tr, mean_va)[0]),
            "top10_overlap_train_val": len(top10_va & top10_tr),
            "spearman_gain_vs_val": float(spearmanr(gain, mean_va)[0]),
            "total_mean_abs_val": float(mean_va.sum())}


def missingness_table(X_tr, y_tr, X_va, y_va, col: str = "sales_growth") -> dict:
    """Bankruptcy rate with the column missing vs present, per split. Pattern only."""
    out = {}
    for split, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
        miss = X[col].isna().to_numpy()
        y = np.asarray(y)
        out[split] = {}
        for tag, m in (("missing", miss), ("present", ~miss)):
            n, k = int(m.sum()), int(y[m].sum())
            out[split][tag] = {"firms": n, "bankrupt": k, "rate": k / n if n else None, "wilson_95": wilson(k, n)}
    return out


def shap_by_missing(phi, X, col: str = "sales_growth") -> dict:
    j = [f[1] for f in FEATURES].index(col)
    miss = X[col].isna().to_numpy()
    res = {}
    for tag, m in (("missing", miss), ("present", ~miss)):
        res[tag] = {"firms": int(m.sum()), "mean_shap": float(phi[m, j].mean()) if m.any() else None,
                    "mean_abs_shap": float(np.abs(phi[m, j]).mean()) if m.any() else None}
    return res


# ----------------------------------------------------------------------------- one firm
def explain_firm(model, calibrator, row: pd.DataFrame, bands: dict, top_k: int = 5, explainer=None) -> dict:
    """Probability, band and top drivers for one firm (a one-row frame).

    Contributions are in calibrated log-odds when the calibrator is Platt (the slope check is run
    for the whole data set in scripts/run_phase6.py), otherwise in raw log-odds."""
    d = feature_dictionary()
    ex = explainer or shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        phi = ex.shap_values(row)
    if isinstance(phi, list):
        phi = phi[1]
    phi = np.asarray(phi, dtype=float).reshape(-1)
    base = float(np.ravel(ex.expected_value)[-1])
    p = float(calibrator.transform(model.predict_proba(row)[:, 1])[0])
    platt = isinstance(calibrator, C.PlattCalibrator)
    a, b = (calibrator.a, calibrator.b) if platt else (1.0, 0.0)
    contrib = phi * a
    drivers = []
    for j in np.argsort(-np.abs(contrib))[:top_k]:
        val = row.iloc[0, j]
        missing = bool(pd.isna(val))
        shown = "missing" if missing else f"{val:.4g}"
        verb = "raises" if contrib[j] > 0 else "lowers"
        note = f" [UNVERIFIED definition, {d[j]['attr']}]" if d[j]["unverified"] else ""
        drivers.append({"attr": d[j]["attr"], "label": d[j]["label"], "unverified": d[j]["unverified"],
                        "value": None if missing else float(val), "contribution": float(contrib[j]),
                        "direction": verb,
                        "sentence": f"{d[j]['attr']} ({d[j]['formula']}) is {shown} and {verb} the risk score "
                                    f"by {abs(contrib[j]):.2f}{note}"})
    return {"probability": p, "band": str(C.assign_bands([p], bands["t_watch"], bands["t_high"])[0]),
            "base_probability": float(1 / (1 + np.exp(-(base * a + b)))),
            "scale": "calibrated_log_odds" if platt else "raw_log_odds", "drivers": drivers}
