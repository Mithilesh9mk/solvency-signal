"""Phase 10: fit the 4-ratio Quick check model (train fit, validation report, test never read).

The protocol and its amendment are in docs/decisions.md ("Phase 10"). Model: monotone LightGBM on four ratios, Platt
scaling fitted on train out-of-fold raw scores, 1 to 10 scale from train out-of-fold percentiles. Logistic regression is
fitted only for the pre-registered train cross-validation comparison. Writes artifacts/quick/quick_model.json and
reports/phase10_quick.{md,json}.

Run: python -m scripts.run_phase10_quick
"""

import json

import lightgbm as lgb
import numpy as np
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
CHALLENGER_MARGIN = 0.02
REPORTS = ROOT / "reports"
OUT = ROOT / Q.QUICK_PATH


def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(v, nd=3):
    return f"{f3(v[0], nd)} to {f3(v[1], nd)}"


def make_challenger() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=4, min_child_samples=30,
                              monotone_constraints=[-1, -1, -1, -1], random_state=SEED, verbose=-1)


def lr_fit_predict(Xtr, ytr, Xte):
    """Logistic regression for the pre-registered comparison: winsorize at train 1/99, standardize, L2 with C = 1."""
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    Xc = np.clip(Xtr, lo, hi)
    mean, sd = Xc.mean(axis=0), Xc.std(axis=0)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xc - mean) / sd, ytr)
    return lr.predict_proba((np.clip(Xte, lo, hi) - mean) / sd)[:, 1], lr.coef_[0]


def cross_val(X: np.ndarray, y: np.ndarray, folds: int = 5) -> dict:
    """Train-only 5-fold: logistic regression against the monotone challenger. Keeps the challenger's raw OOF scores."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    raw_oof = np.zeros(len(y))
    pr_lr, pr_ch = [], []
    for tr, te in skf.split(X, y):
        p_lr, _ = lr_fit_predict(X[tr], y[tr], X[te])
        pr_lr.append(average_precision_score(y[te], p_lr))
        ch = make_challenger().fit(X[tr], y[tr])
        raw_oof[te] = ch.predict(X[te], raw_score=True)
        pr_ch.append(average_precision_score(y[te], raw_oof[te]))
    d = float(np.mean(pr_ch) - np.mean(pr_lr))
    _, coef = lr_fit_predict(X, y, X[:2])
    return {"raw_oof": raw_oof, "pr_auc_lr": float(np.mean(pr_lr)), "pr_auc_challenger": float(np.mean(pr_ch)),
            "difference": d, "challenger_wins": bool(d >= CHALLENGER_MARGIN),
            "lr_coef_standardized": [float(c) for c in coef], "lr_signs_ok": bool(np.all(coef < 0))}


def score_table(y: np.ndarray, scores: np.ndarray) -> list:
    rows = []
    for s in range(1, 11):
        m = scores == s
        n, k = int(m.sum()), int(y[m].sum())
        rows.append({"score": s, "band": Q.band_for(s), "n": n, "bankrupt": k,
                     "rate": k / n if n else None, "wilson": E.wilson(k, n)})
    return rows


def calib_line(y, p) -> dict:
    """Calibration intercept and slope (logistic fit of the label on the log-odds of the prediction) and ECE."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(Q.logit(p).reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]), "ece": float(C.ece(y, p))}


def prepare(X, y):
    miss = X[Q.FEATURE_COLS].isna()
    keep = ~miss.any(axis=1)
    return X, y, keep, {c: int(miss[c].sum()) for c in Q.FEATURE_COLS}


def run(reports=REPORTS, out=OUT) -> dict:
    X_tr_all, y_tr_all, keep_tr, miss_tr = prepare(*load_split("train"))
    X_va_all, y_va_all, keep_va, miss_va = prepare(*load_split("val"))
    X_tr, y_tr = X_tr_all.loc[keep_tr, Q.FEATURE_COLS].to_numpy(float), y_tr_all[keep_tr].to_numpy(int)
    X_va, y_va = X_va_all.loc[keep_va, Q.FEATURE_COLS].to_numpy(float), y_va_all[keep_va].to_numpy(int)

    cv = cross_val(X_tr, y_tr)
    res = {"seed": SEED, "protocol_margin": CHALLENGER_MARGIN, "stopped": "",
           "population": {"train_firms": int(len(y_tr_all)), "train_used": int(len(y_tr)), "train_bankrupt": int(y_tr.sum()),
                          "train_blank_by_ratio": miss_tr, "val_firms": int(len(y_va_all)), "val_used": int(len(y_va)),
                          "val_bankrupt": int(y_va.sum()), "val_blank_by_ratio": miss_va},
           "cv": {k: v for k, v in cv.items() if k != "raw_oof"}}
    if not cv["challenger_wins"]:
        res["stopped"] = "Logistic regression was chosen by the protocol, and the app code supports the boosted model only."
        write_report(res, reports)
        return res

    platt = LogisticRegression(C=1e6, max_iter=1000).fit(cv["raw_oof"].reshape(-1, 1), y_tr)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    oof_p = Q.sigmoid(a * cv["raw_oof"] + b)
    cuts = [float(c) for c in np.percentile(oof_p, Q.CUT_PERCENTILES)]
    s_tr = Q.score_for(oof_p, cuts)
    model = make_challenger().fit(X_tr, y_tr)
    params = {"features": Q.FEATURE_COLS, "lo": np.percentile(X_tr, 1, axis=0).tolist(),
              "hi": np.percentile(X_tr, 99, axis=0).tolist(), "vmin": X_tr.min(axis=0).tolist(),
              "vmax": X_tr.max(axis=0).tolist(), "model_str": model.booster_.model_to_string(),
              "platt": {"a": a, "b": b}, "cuts": cuts, "score_rates": score_table(y_tr, s_tr),
              "n_train": int(len(y_tr)), "n_train_bankrupt": int(y_tr.sum()), "seed": SEED,
              "bands": [list(x) for x in Q.BANDS]}
    Q.validate(params)
    built = Q.build(params)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=1) + "\n", encoding="utf-8")

    p_va = Q.predict_proba(X_va, built)
    s_va = Q.score_for(p_va, cuts)
    contrib = built["_booster"].predict(Q.clip(X_tr, built), pred_contrib=True)
    mean_abs = {n: float(a * np.abs(contrib[:, j]).mean()) for j, n in enumerate(Q.FEATURE_COLS)}
    frozen = A.load_frozen()
    Xf = X_va_all.loc[keep_va].astype(float)
    p64 = frozen["calibrator"].transform(frozen["model"].predict_proba(Xf)[:, 1])
    m4, m64 = EV.score(y_va, p_va), EV.score(y_va, p64)
    res.update({
        "platt": {"a": a, "b": b}, "cuts": cuts, "mean_abs_contribution_train": mean_abs,
        "val": {"quick": {**m4, **EV.bootstrap_ci(y_va, p_va), **calib_line(y_va, p_va)},
                "full_64": {**m64, **EV.bootstrap_ci(y_va, p64), **calib_line(y_va, p64)},
                "paired_quick_minus_full": EV.paired_bootstrap(y_va, p_va, p64, n=2000)},
        "oof": {**EV.score(y_tr, oof_p), **calib_line(y_tr, oof_p)},
        "table_train_oof": params["score_rates"], "table_val": score_table(y_va, s_va),
        "input_bounds": Q.input_bounds(built)})
    write_report(res, reports)
    return res


def write_report(res: dict, reports=REPORTS) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase10_quick.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    (reports / "phase10_quick.md").write_text(render(res), encoding="utf-8")


def render(r: dict) -> str:
    pop, cv = r["population"], r["cv"]
    L = ["# Phase 10: Quick check (4-ratio model)", "",
         "Fit on train, reported on validation. The test split was not read. Protocol and amendment: docs/decisions.md, Phase 10.", "",
         "## 1. Population", "",
         f"Train: {pop['train_used']} of {pop['train_firms']} firms have all four ratios ({pop['train_bankrupt']} bankrupt). "
         f"Validation: {pop['val_used']} of {pop['val_firms']} ({pop['val_bankrupt']} bankrupt). "
         f"Blank counts, train: {pop['train_blank_by_ratio']}; validation: {pop['val_blank_by_ratio']}.", "",
         "## 2. Model choice (train 5-fold, pre-registered)", "",
         f"PR AUC: logistic regression {cv['pr_auc_lr']:.3f}, monotone LightGBM {cv['pr_auc_challenger']:.3f}, difference "
         f"{cv['difference']:+.3f} (the challenger needs +{r['protocol_margin']:.2f}). Challenger chosen: "
         f"{'yes' if cv['challenger_wins'] else 'no'}.",
         "Logistic coefficients on standardized ratios (for reference): "
         + ", ".join(f"{n} {c:+.3f}" for n, c in zip(Q.FEATURE_COLS, cv["lr_coef_standardized"]))
         + f". All negative as expected: {'yes' if cv['lr_signs_ok'] else 'NO'}.", ""]
    if r["stopped"]:
        return "\n".join(L + ["## STOPPED", "", r["stopped"], ""]) + "\n"
    v, oof = r["val"], r["oof"]
    L += [f"Platt scaling on train out-of-fold raw scores: slope {r['platt']['a']:.3f}, intercept {r['platt']['b']:.3f}.", "",
          "Mean absolute contribution per ratio on train firms (calibrated log-odds):", "",
          "| Ratio | Mean absolute contribution |", "|---|---|"]
    for n, m in sorted(r["mean_abs_contribution_train"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {Q.LABEL_BY_FEATURE[n]} | {m:.3f} |")
    L += ["", "## 3. Validation: 4-ratio model against the frozen 64-ratio model (same firms)", "",
          "| Model | PR AUC (95% CI) | ROC AUC (95% CI) | Brier | ECE | Calibration slope |", "|---|---|---|---|---|---|"]
    for key, nm in (("quick", "4-ratio, monotone LightGBM + Platt"), ("full_64", "64-ratio, frozen LightGBM + Platt")):
        m = v[key]
        L.append(f"| {nm} | {m['pr_auc']:.3f} ({ci(m['pr_auc_ci'])}) | {m['roc_auc']:.3f} ({ci(m['roc_auc_ci'])}) | "
                 f"{m['brier']:.4f} | {m['ece']:.4f} | {m['slope']:.2f} |")
    pb = v["paired_quick_minus_full"]
    L += ["", f"No-skill PR AUC (validation bankrupt rate): {v['quick']['positive_rate']:.3f}. Paired difference, "
          f"4-ratio minus 64-ratio: PR AUC {pb['pr_auc']['diff']:+.3f} ({ci(pb['pr_auc']['ci'])}), "
          f"ROC AUC {pb['roc_auc']['diff']:+.3f} ({ci(pb['roc_auc']['ci'])}).",
          f"Train out-of-fold, 4-ratio: PR AUC {oof['pr_auc']:.3f}, ROC AUC {oof['roc_auc']:.3f}, ECE {oof['ece']:.4f}.", "",
          "## 4. The 1 to 10 scale", "",
          "Cut points are the 10th to 90th percentiles of the train out-of-fold calibrated probability: "
          + ", ".join(f"{c:.4f}" for c in r["cuts"]) + ". The scale is a rank within the Polish training population, not an absolute risk.", "",
          "| Score | Band | Train OOF firms | Bankrupt | Rate (95% Wilson) | Val firms | Bankrupt | Rate (95% Wilson) |",
          "|---|---|---|---|---|---|---|---|"]
    for a, b in zip(r["table_train_oof"], r["table_val"]):
        L.append(f"| {a['score']} | {a['band']} | {a['n']} | {a['bankrupt']} | {f3(a['rate'])} ({ci(a['wilson'])}) | "
                 f"{b['n']} | {b['bankrupt']} | {f3(b['rate'])} ({ci(b['wilson'])}) |")
    L += ["", "## Flags", "",
          "1. Validation was not used for any choice, but it has few bankrupt firms, so the intervals are wide.",
          "2. The cash earnings ratio is a proxy: (net profit + depreciation) / total liabilities, not operating cash flow / current liabilities.",
          "3. Debt-to-equity uses total liabilities as debt (UNVERIFIED that this matches what is meant by debt).",
          "4. The range shown in the app is the observed bankruptcy rate of train out-of-fold firms with the same score. It does not cover another country, another year or a different ratio definition.",
          "5. The test split is spent, so nothing here is checked on untouched data.", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    res = run()
    print("STOPPED: " + res["stopped"] if res["stopped"] else f"wrote {Q.QUICK_PATH}, reports/phase10_quick.md and .json")


if __name__ == "__main__":
    main()
