"""Phase 10: fit the 4-ratio Quick check model (train fit, validation report, test never read).

The protocol is in docs/decisions.md ("Phase 10 protocol"). Writes artifacts/quick/quick_model.json and
reports/phase10_quick.{md,json}. If a coefficient has the wrong sign, or the monotone challenger wins the train
cross-validation by the pre-registered margin, the run stops and no artifact is written.

Run: python -m scripts.run_phase10_quick
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src import app_logic as A
from src import calibrate as C
from src import evaluate as EV
from src import explain as E
from src import quick as Q
from src.preprocess import ROOT
from src.train import load_split

SEED = 42
N_BOOT = Q.N_BOOT
CHALLENGER_MARGIN = 0.02
REPORTS = ROOT / "reports"
OUT = ROOT / Q.QUICK_PATH


def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(v, nd=3):
    return f"{f3(v[0], nd)} to {f3(v[1], nd)}"


def fit_params(X: np.ndarray, y: np.ndarray) -> dict:
    """Winsor bounds and standardization from X, then L2 logistic regression (C = 1, no class weights)."""
    lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    Xc = np.clip(X, lo, hi)
    mean, sd = Xc.mean(axis=0), Xc.std(axis=0)
    p = {"lo": lo, "hi": hi, "mean": mean, "sd": sd}
    lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xc - mean) / sd, y)
    p["coef"], p["intercept"] = lr.coef_[0], float(lr.intercept_[0])
    return p


def make_challenger() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=4, min_child_samples=30,
                              monotone_constraints=[-1, -1, -1, -1], random_state=SEED, verbose=-1)


def cross_val(X: np.ndarray, y: np.ndarray, folds: int = 5) -> dict:
    """Train-only 5-fold: logistic regression (refit with its own bounds per fold) against the monotone challenger."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    pr_lr, pr_ch = [], []
    for tr, te in skf.split(X, y):
        p = fit_params(X[tr], y[tr])
        oof[te] = Q.predict_proba(X[te], p)
        pr_lr.append(average_precision_score(y[te], oof[te]))
        ch = make_challenger().fit(X[tr], y[tr])
        pr_ch.append(average_precision_score(y[te], ch.predict_proba(X[te])[:, 1]))
    d = float(np.mean(pr_ch) - np.mean(pr_lr))
    return {"oof": oof, "pr_auc_lr": float(np.mean(pr_lr)), "pr_auc_challenger": float(np.mean(pr_ch)),
            "pr_auc_lr_folds": [float(v) for v in pr_lr], "pr_auc_challenger_folds": [float(v) for v in pr_ch],
            "difference": d, "challenger_wins": bool(d >= CHALLENGER_MARGIN)}


def boot_coefs(Z: np.ndarray, y: np.ndarray, n: int = N_BOOT) -> np.ndarray:
    """Refit on stratified train resamples, with the full-train bounds and standardization held fixed."""
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = np.zeros((n, Z.shape[1] + 1))
    for b in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[idx], y[idx])
        out[b, :-1], out[b, -1] = lr.coef_[0], lr.intercept_[0]
    return out


def score_table(y: np.ndarray, scores: np.ndarray) -> list:
    rows = []
    for s in range(1, 11):
        m = scores == s
        n, k = int(m.sum()), int(y[m].sum())
        rows.append({"score": s, "band": Q.band_for(s), "n": n, "bankrupt": k,
                     "rate": k / n if n else None, "wilson": E.wilson(k, n)})
    return rows


def calib_line(y, p) -> dict:
    """Calibration intercept and slope (logistic fit of the label on logit of the prediction) and ECE."""
    lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]), "ece": float(C.ece(y, p))}


def prepare(split: str):
    X, y = load_split(split)
    cols = Q.FEATURE_COLS
    miss = X[cols].isna()
    keep = ~miss.any(axis=1)
    return X, y, keep, {c: int(miss[c].sum()) for c in cols}


def run(reports=REPORTS, out=OUT, n_boot: int = N_BOOT) -> dict:
    X_tr_all, y_tr_all, keep_tr, miss_tr = prepare("train")
    X_va_all, y_va_all, keep_va, miss_va = prepare("val")
    X_tr, y_tr = X_tr_all.loc[keep_tr, Q.FEATURE_COLS].to_numpy(float), y_tr_all[keep_tr].to_numpy(int)
    X_va, y_va = X_va_all.loc[keep_va, Q.FEATURE_COLS].to_numpy(float), y_va_all[keep_va].to_numpy(int)

    p = fit_params(X_tr, y_tr)
    signs_ok = bool(np.all(p["coef"] < 0))
    cv = cross_val(X_tr, y_tr)
    res = {"seed": SEED, "protocol_margin": CHALLENGER_MARGIN, "signs_ok": signs_ok,
           "coef": [float(c) for c in p["coef"]], "intercept": p["intercept"],
           "population": {"train_firms": int(len(y_tr_all)), "train_used": int(len(y_tr)), "train_bankrupt": int(y_tr.sum()),
                          "train_blank_by_ratio": miss_tr, "val_firms": int(len(y_va_all)), "val_used": int(len(y_va)),
                          "val_bankrupt": int(y_va.sum()), "val_blank_by_ratio": miss_va},
           "cv": {k: v for k, v in cv.items() if k != "oof"}, "stopped": ""}
    if not signs_ok:
        res["stopped"] = "A coefficient has the wrong sign. Nothing was written to the app."
    elif cv["challenger_wins"]:
        res["stopped"] = "The monotone challenger beat logistic regression by the pre-registered margin. The user decides."
    if res["stopped"]:
        write_report(res, reports)
        return res

    Z = Q.transform(X_tr, p)
    boot = boot_coefs(Z, y_tr, n_boot)
    cuts = [float(c) for c in np.percentile(cv["oof"], Q.CUT_PERCENTILES)]
    params = {"features": Q.FEATURE_COLS, "lo": p["lo"].tolist(), "hi": p["hi"].tolist(), "mean": p["mean"].tolist(),
              "sd": p["sd"].tolist(), "coef": p["coef"].tolist(), "intercept": p["intercept"],
              "boot": boot.tolist(), "cuts": cuts, "n_train": int(len(y_tr)), "n_train_bankrupt": int(y_tr.sum()),
              "seed": SEED, "bands": [list(b) for b in Q.BANDS]}
    Q.validate(params)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=1) + "\n", encoding="utf-8")

    p_va = Q.predict_proba(X_va, p)
    s_tr, s_va = Q.score_for(cv["oof"], cuts), Q.score_for(p_va, cuts)
    # frozen 64-ratio model on the same validation firms (model and calibrator unchanged)
    frozen = A.load_frozen()
    Xf = X_va_all.loc[keep_va].astype(float)
    p64 = frozen["calibrator"].transform(frozen["model"].predict_proba(Xf)[:, 1])
    m4, m64 = EV.score(y_va, p_va), EV.score(y_va, p64)
    res.update({
        "cuts": cuts, "stopped": "",
        "val": {"quick": {**m4, **EV.bootstrap_ci(y_va, p_va, n_boot), **calib_line(y_va, p_va)},
                "full_64": {**m64, **EV.bootstrap_ci(y_va, p64, n_boot), **calib_line(y_va, p64)},
                "paired_quick_minus_full": EV.paired_bootstrap(y_va, p_va, p64, n=2000)},
        "oof": {**EV.score(y_tr, cv["oof"]), **calib_line(y_tr, cv["oof"])},
        "table_train_oof": score_table(y_tr, s_tr), "table_val": score_table(y_va, s_va),
        "bootstrap_boot_n": int(len(boot)),
        "inputs_bounds": Q.input_bounds(p),
    })
    write_report(res, reports)
    return res


def write_report(res: dict, reports=REPORTS) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase10_quick.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    (reports / "phase10_quick.md").write_text(render(res), encoding="utf-8")


def render(r: dict) -> str:
    pop, cv = r["population"], r["cv"]
    L = ["# Phase 10: Quick check (4-ratio model)", "",
         "Fit on train, reported on validation. The test split was not read. Protocol: docs/decisions.md, Phase 10.", "",
         "## 1. Population", "",
         f"Train: {pop['train_used']} of {pop['train_firms']} firms have all four ratios ({pop['train_bankrupt']} bankrupt). "
         f"Validation: {pop['val_used']} of {pop['val_firms']} ({pop['val_bankrupt']} bankrupt). "
         f"Blank counts, train: {pop['train_blank_by_ratio']}; validation: {pop['val_blank_by_ratio']}.", "",
         "## 2. Coefficients and checks", "",
         "| Feature (standardized) | Coefficient | Expected sign |", "|---|---|---|"]
    from src.quick import FEATURE_COLS
    for name, c in zip(FEATURE_COLS, r["coef"]):
        L.append(f"| {name} | {c:.3f} | negative |")
    L += ["", f"Intercept {r['intercept']:.3f}. All signs as pre-registered: {'yes' if r['signs_ok'] else 'NO'}.", "",
          f"Train 5-fold PR AUC: logistic regression {cv['pr_auc_lr']:.3f}, monotone LightGBM challenger "
          f"{cv['pr_auc_challenger']:.3f}, difference {cv['difference']:+.3f} (challenger needs +{r['protocol_margin']:.2f}). "
          f"Challenger wins: {'yes' if cv['challenger_wins'] else 'no'}.", ""]
    if r["stopped"]:
        L += ["## STOPPED", "", r["stopped"], ""]
        return "\n".join(L) + "\n"
    v, oof = r["val"], r["oof"]
    L += ["## 3. Validation: 4-ratio model against the frozen 64-ratio model (same firms)", "",
          "| Model | PR AUC (95% CI) | ROC AUC (95% CI) | Brier | ECE | Calibration slope |", "|---|---|---|---|---|---|"]
    for key, nm in (("quick", "4-ratio, logistic"), ("full_64", "64-ratio, frozen LightGBM + Platt")):
        m = v[key]
        L.append(f"| {nm} | {m['pr_auc']:.3f} ({ci(m['pr_auc_ci'])}) | {m['roc_auc']:.3f} ({ci(m['roc_auc_ci'])}) | "
                 f"{m['brier']:.4f} | {m['ece']:.4f} | {m['slope']:.2f} |")
    pb = v["paired_quick_minus_full"]
    L += ["", f"No-skill PR AUC (validation bankrupt rate): {v['quick']['positive_rate']:.3f}. "
          f"Paired difference, 4-ratio minus 64-ratio: PR AUC {pb['pr_auc']['diff']:+.3f} ({ci(pb['pr_auc']['ci'])}), "
          f"ROC AUC {pb['roc_auc']['diff']:+.3f} ({ci(pb['roc_auc']['ci'])}).",
          f"Train out-of-fold, 4-ratio: PR AUC {oof['pr_auc']:.3f}, ROC AUC {oof['roc_auc']:.3f}, ECE {oof['ece']:.4f}.", "",
          "## 4. The 1 to 10 scale", "",
          "Cut points are the 10th to 90th percentiles of the train out-of-fold probability: "
          + ", ".join(f"{c:.4f}" for c in r["cuts"]) + ". The scale is a rank within the Polish training population.", "",
          "| Score | Band | Train OOF firms | Bankrupt | Rate (95% Wilson) | Val firms | Bankrupt | Rate (95% Wilson) |",
          "|---|---|---|---|---|---|---|---|"]
    for a, b in zip(r["table_train_oof"], r["table_val"]):
        L.append(f"| {a['score']} | {a['band']} | {a['n']} | {a['bankrupt']} | {f3(a['rate'])} ({ci(a['wilson'])}) | "
                 f"{b['n']} | {b['bankrupt']} | {f3(b['rate'])} ({ci(b['wilson'])}) |")
    L += ["", "## Flags", "",
          "1. Validation was not used for any choice, but it has few bankrupt firms, so the intervals are wide.",
          "2. The cash earnings ratio is a proxy: (net profit + depreciation) / total liabilities, not operating cash flow / current liabilities.",
          "3. Debt-to-equity uses total liabilities as debt (UNVERIFIED that this matches what is meant by debt).",
          "4. The range is bootstrap estimation noise only. It does not cover another country, another year or a different ratio definition.",
          "5. The test split is spent, so nothing here is checked on untouched data.", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    res = run()
    if res["stopped"]:
        print("STOPPED:", res["stopped"])
        return
    print(f"wrote {Q.QUICK_PATH}, reports/phase10_quick.md and .json")


if __name__ == "__main__":
    main()
