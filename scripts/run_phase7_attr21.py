"""Phase 7: what is in the data when Attr21 (sales growth) is missing? Train and validation only.

Run: python -m scripts.run_phase7_attr21

Descriptive analysis of the frozen LightGBM + Platt model (docs/decisions.md, Phase 7 protocol). Nothing is
retrained, recalibrated or re-cut, and the test split is never loaded. All randomness uses SEED = 42.
Writes reports/phase7_attr21.{json,md} and reports/figures/phase7_attr21.png.
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_phase6 import verify_frozen
from src import calibrate as C
from src import explain as E
from src.plotting import BLUE, GREEN, INK2, VERMILLION, apply_style, style_axes
from src.polish_features import FEATURES
from src.preprocess import ROOT
from src.train import load_split, make_lgbm

SEED = 42
N_BOOT = 1000
NL = "\n"
COL = "sales_growth"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
BANDS = ("Low", "Watch", "High")
MIN_CLASS = 5  # AUC and its CI are not reported with fewer firms than this in either class


# ------------------------------------------------------------------------------ pieces
def counts(X_tr, y_tr, X_va, y_va) -> dict:
    out = E.missingness_table(X_tr, y_tr, X_va, y_va, COL)
    for split, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
        miss = X[COL].isna().to_numpy()
        pos = int(np.sum(y))
        out[split]["missing_share_of_all_bankrupt"] = float(np.asarray(y)[miss].sum() / pos)
        out[split]["missing_share_of_firms"] = float(miss.mean())
    return out


def comissing(X, top: int = 10) -> dict:
    miss = X[COL].isna().to_numpy()
    n_missing = X.isna().sum(axis=1).to_numpy()
    rate_m, rate_p = X[miss].isna().mean(), X[~miss].isna().mean()
    diff = (rate_m - rate_p).drop(COL).sort_values(ascending=False)
    attr = {f[1]: f[0] for f in FEATURES}
    return {"mean_missing_attributes": {"missing": float(n_missing[miss].mean()), "present": float(n_missing[~miss].mean())},
            "median_missing_attributes": {"missing": float(np.median(n_missing[miss])), "present": float(np.median(n_missing[~miss]))},
            "count_distribution_missing_group": {int(k): int(v) for k, v in pd.Series(n_missing[miss]).value_counts().sort_index().items()},
            "top_attributes": [{"attr": attr[c], "name": c, "rate_when_attr21_missing": float(rate_m[c]),
                                "rate_when_attr21_present": float(rate_p[c]), "difference": float(d)}
                               for c, d in diff.head(top).items()]}


def sales_check(X) -> dict:
    """Do current sales exist? Attr9 and Attr36 are sales / assets ratios."""
    miss = X[COL].isna()
    out = {"firms_missing": int(miss.sum())}
    for c in ("sales_to_assets", "total_sales_to_assets"):
        v = X.loc[miss, c]
        out[c] = {"equal_to_zero": int((v == 0).sum()), "missing": int(v.isna().sum())}
    return out


def profile(X, y, attrs: list) -> dict:
    names = {f[0]: f[1] for f in FEATURES}
    yy = np.asarray(y)
    miss = X[COL].isna().to_numpy()
    groups = {"bankrupt, Attr21 missing": (yy == 1) & miss, "bankrupt, Attr21 present": (yy == 1) & ~miss,
              "healthy, Attr21 present": (yy == 0) & ~miss, "healthy, Attr21 missing": (yy == 0) & miss}
    return {g: {"firms": int(m.sum()), **{a: (float(X.loc[m, names[a]].median()) if m.sum() else None) for a in attrs}}
            for g, m in groups.items()}


def auc_block(y, p, n_boot: int) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pos = int(y.sum())
    out = {"firms": int(len(y)), "bankrupt": pos, "prevalence": float(y.mean()) if len(y) else None,
           "pr_auc": None, "roc_auc": None}
    if pos < MIN_CLASS or len(y) - pos < MIN_CLASS:
        out["note"] = f"fewer than {MIN_CLASS} firms in one class, AUC not reported"
        return out
    out["pr_auc"], out["roc_auc"] = float(average_precision_score(y, p)), float(roc_auc_score(y, p))
    idx = C.bootstrap_indices(y, n_boot, SEED)
    out["pr_auc_ci"] = np.percentile([average_precision_score(y[i], p[i]) for i in idx], [2.5, 97.5]).tolist()
    out["roc_auc_ci"] = np.percentile([roc_auc_score(y[i], p[i]) for i in idx], [2.5, 97.5]).tolist()
    return out


def band_block(y, p, t_watch: float, t_high: float) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pos = int(y.sum())
    out = {}
    for tag, t in (("high", t_high), ("watch_plus_high", t_watch)):
        f = p >= t
        tp = int((f & (y == 1)).sum())
        out[tag] = {"flagged": int(f.sum()), "bankrupt": tp, "precision": tp / f.sum() if f.sum() else None,
                    "recall": tp / pos if pos else None}
    return out


def performance(X, y, p, bands, n_boot: int) -> dict:
    miss = X[COL].isna().to_numpy()
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    out = {}
    for tag, m in (("all", np.ones(len(y), bool)), ("present", ~miss), ("missing", miss)):
        out[tag] = {**auc_block(y[m], p[m], n_boot), "bands": band_block(y[m], p[m], bands["t_watch"], bands["t_high"])}
    high = p >= bands["t_high"]
    out["high_band_composition"] = {"high_firms": int(high.sum()), "high_with_attr21_missing": int((high & miss).sum()),
                                    "bankrupt_in_high_with_missing": int((high & miss & (y == 1)).sum()),
                                    "high_with_attr21_present": int((high & ~miss).sum()),
                                    "bankrupt_in_high_with_present": int((high & ~miss & (y == 1)).sum())}
    return out


def what_if(model, calibrator, bands, X_va, X_tr) -> dict:
    """Frozen model behaviour when Attr21 is blanked or filled. Descriptive, not a recommendation."""
    a = calibrator.a if isinstance(calibrator, C.PlattCalibrator) else 1.0

    def score(X):
        p = calibrator.transform(model.predict_proba(X)[:, 1])
        return p, C.assign_bands(p, bands["t_watch"], bands["t_high"])

    base_p, base_b = score(X_va)
    miss = X_va[COL].isna().to_numpy()
    blank = X_va.copy()
    blank.loc[~miss, COL] = np.nan
    fill = X_va.copy()
    median = float(X_tr[COL].median())
    fill.loc[miss, COL] = median
    out = {"train_median_attr21": median}
    for tag, X_new, sel in (("present_then_blanked", blank, ~miss), ("missing_then_filled_with_train_median", fill, miss)):
        p_new, b_new = score(X_new)
        logit = lambda q: np.log(q / (1 - q))  # noqa: E731
        trans = {f"{o}->{n}": int(((base_b[sel] == o) & (b_new[sel] == n)).sum()) for o in BANDS for n in BANDS}
        out[tag] = {"firms": int(sel.sum()), "transitions": trans,
                    "band_counts_before": {b: int((base_b[sel] == b).sum()) for b in BANDS},
                    "band_counts_after": {b: int((b_new[sel] == b).sum()) for b in BANDS},
                    "mean_change_calibrated_log_odds": float(np.mean(logit(p_new[sel]) - logit(base_p[sel]))),
                    "slope_a": a}
    return out


def phase2_ablation(path: Path = REPORTS / "models_val.md") -> dict:
    """Quote the Phase 2 ablation rows (model retrained without Attr21). Not rerun."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    full = re.search(r"\| All features \| ([\d.]+) \|", text)
    drop = re.search(r"\| Attr21 dropped \| ([\d.]+) \|", text)
    return {"pr_auc_all_features": float(full.group(1)) if full else None,
            "pr_auc_attr21_dropped": float(drop.group(1)) if drop else None,
            "source": "reports/models_val.md (Phase 2, not rerun)"}


# ------------------------------------------------------------------------------ compute
def compute(X_tr, y_tr, X_va, y_va, model, calibrator, bands, top_attrs: list, n_boot: int = N_BOOT) -> dict:
    oof = C.oof_predictions(make_lgbm, X_tr, y_tr)
    oof_p = calibrator.transform(oof)
    va_p = calibrator.transform(model.predict_proba(X_va)[:, 1])
    return {"counts": counts(X_tr, y_tr, X_va, y_va),
            "comissing": {"train": comissing(X_tr), "val": comissing(X_va)},
            "sales_check": {"train": sales_check(X_tr), "val": sales_check(X_va)},
            "profile_train": profile(X_tr, y_tr, top_attrs), "profile_attrs": top_attrs,
            "performance": {"val": performance(X_va, y_va, va_p, bands, n_boot),
                            "train_oof": performance(X_tr, y_tr, oof_p, bands, n_boot)},
            "what_if_val": what_if(model, calibrator, bands, X_va, X_tr), "phase2_ablation": phase2_ablation(),
            "bands": {"t_watch": bands["t_watch"], "t_high": bands["t_high"]}}


# --------------------------------------------------------------------------------- plot
def plot(res: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.5]})
    c = res["counts"]
    xs = np.arange(2)
    for k, (split, color, marker) in enumerate((("train", BLUE, "o"), ("val", VERMILLION, "s"))):
        rates = [c[split][t]["rate"] for t in ("missing", "present")]
        err = [[c[split][t]["rate"] - c[split][t]["wilson_95"][0] for t in ("missing", "present")],
               [c[split][t]["wilson_95"][1] - c[split][t]["rate"] for t in ("missing", "present")]]
        a.errorbar(xs + (k - 0.5) * 0.14, rates, yerr=err, fmt=marker, color=color, capsize=3,
                   label=f"{split} (missing n={c[split]['missing']['firms']}, present n={c[split]['present']['firms']})")
    a.set_xticks(xs, ["Attr21 missing", "Attr21 present"])
    a.set_xlim(-0.5, 1.5)
    a.set_ylim(0, 1.05)
    a.set_ylabel("Bankruptcy rate (95% Wilson interval)")
    a.set_title("Bankruptcy rate by Attr21 status", fontsize=10)
    a.legend(frameon=False, fontsize=7.5, loc="center right")
    style_axes(a)
    w = res["what_if_val"]
    rows = [("present, as is", w["present_then_blanked"]["band_counts_before"]),
            ("present, then blanked", w["present_then_blanked"]["band_counts_after"]),
            ("missing, as is", w["missing_then_filled_with_train_median"]["band_counts_before"]),
            ("missing, then filled with train median", w["missing_then_filled_with_train_median"]["band_counts_after"])]
    colors, hatches = {"Low": BLUE, "Watch": GREEN, "High": VERMILLION}, {"Low": "", "Watch": "//", "High": "xx"}
    for i, (label, cnt) in enumerate(rows[::-1]):
        total = sum(cnt.values())
        left = 0.0
        for band in BANDS:
            share = cnt[band] / total
            b.barh(i, share, left=left, color=colors[band], hatch=hatches[band], edgecolor="white", lw=0.6,
                   label=band if i == 0 else None)
            if share > 0.04:
                b.text(left + share / 2, i, str(cnt[band]), ha="center", va="center", color="black", fontsize=8,
                       fontweight="bold", bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9})
            left += share
    b.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8)
    b.set_xlim(0, 1)
    b.set_xlabel("Share of validation firms in each band (counts inside bars)")
    b.set_title("Frozen model band when Attr21 is changed (what-if, not a recommendation)", fontsize=9)
    b.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    style_axes(b)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------- report
def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(v, c, nd=3):
    return f3(v, nd) if c is None else f"{f3(v, nd)} ({f3(c[0], nd)} to {f3(c[1], nd)})"


def make_flags(res: dict) -> list:
    c, pv, po = res["counts"], res["performance"]["val"], res["performance"]["train_oof"]
    w = res["what_if_val"]
    cm = res["comissing"]
    flags = [
        f"Attr21 is missing for {c['train']['missing']['firms']} train and {c['val']['missing']['firms']} validation firms, and "
        f"{c['train']['missing']['bankrupt']} and {c['val']['missing']['bankrupt']} of them are bankrupt. Those firms are "
        f"{c['train']['missing_share_of_all_bankrupt']:.1%} (train) and {c['val']['missing_share_of_all_bankrupt']:.1%} (validation) of all bankrupt firms. "
        "In this data, a missing Attr21 almost always goes with a bankrupt label. The cause is not known.",
        f"The validation group is only {c['val']['missing']['firms']} firms ({c['val']['missing']['firms'] - c['val']['missing']['bankrupt']} healthy), so anything measured on it is very noisy. "
        "Train has the larger group but the model was fit on it, so out-of-fold scores are used for the train performance numbers.",
        f"High band composition on validation: {pv['high_band_composition']['high_with_attr21_missing']} of {pv['high_band_composition']['high_firms']} High firms have Attr21 missing; "
        f"on train out-of-fold: {po['high_band_composition']['high_with_attr21_missing']} of {po['high_band_composition']['high_firms']}.",
        f"On firms with Attr21 present, validation PR AUC is {ci(pv['present']['pr_auc'], pv['present'].get('pr_auc_ci'))} against "
        f"{ci(pv['all']['pr_auc'], pv['all'].get('pr_auc_ci'))} for all firms, and train out-of-fold PR AUC is {f3(po['present']['pr_auc'])} against {f3(po['all']['pr_auc'])}. "
        "The Phase 2 retrain without Attr21 scored "
        f"{f3(res['phase2_ablation']['pr_auc_attr21_dropped'])} against {f3(res['phase2_ablation']['pr_auc_all_features'])} with it.",
        f"What-if on the frozen model (validation): blanking Attr21 for the {w['present_then_blanked']['firms']} firms that have it moves "
        f"{sum(v for k, v in w['present_then_blanked']['transitions'].items() if k.split('->')[0] != k.split('->')[1])} firms to a different band "
        f"(mean change {w['present_then_blanked']['mean_change_calibrated_log_odds']:+.2f} calibrated log-odds). "
        f"Filling the {w['missing_then_filled_with_train_median']['firms']} missing firms with the train median moves "
        f"{sum(v for k, v in w['missing_then_filled_with_train_median']['transitions'].items() if k.split('->')[0] != k.split('->')[1])} "
        f"firms (mean change {w['missing_then_filled_with_train_median']['mean_change_calibrated_log_odds']:+.2f}). "
        "A user who leaves Attr21 blank gets the model's missing-value behaviour, not a neutral score.",
        f"Co-missingness (train): Attr21-missing firms have a median of {cm['train']['median_missing_attributes']['missing']:.0f} attributes missing "
        f"(mean {cm['train']['mean_missing_attributes']['missing']:.2f}, pulled up by a few firms with many missing), against a median of "
        f"{cm['train']['median_missing_attributes']['present']:.0f} (mean {cm['train']['mean_missing_attributes']['present']:.2f}) for the others. The distribution is in the table.",
        f"For firms with Attr21 present, High-band precision is {f3(pv['present']['bands']['high']['precision'])} on validation and "
        f"{f3(po['present']['bands']['high']['precision'])} on train out-of-fold, against {f3(pv['all']['bands']['high']['precision'])} and "
        f"{f3(po['all']['bands']['high']['precision'])} for all firms. The headline numbers are flattered by the small group where a missing value goes with the bankrupt label.",
        "The dataset has no company id and no year, and no field says why a value is missing. This report states counts and patterns only.",
        "The test split cannot be broken down by Attr21 status without reading it again, so no such breakdown exists. The Phase 5 test results are unchanged and already include whatever reliance on Attr21 the model has.",
        "Out-of-fold scores use the saved Platt calibrator that was fitted on those same out-of-fold scores, so the calibrated train numbers are mildly optimistic.",
        "Test split untouched in this phase. No model, calibrator or cut was changed.",
    ]
    return flags


def perf_rows(name: str, perf: dict) -> list:
    rows = []
    for tag in ("all", "present", "missing"):
        b = perf[tag]
        rows.append(f"| {name} | Attr21 {tag} | {b['firms']} | {b['bankrupt']} | {ci(b['pr_auc'], b.get('pr_auc_ci'))} | "
                    f"{ci(b['roc_auc'], b.get('roc_auc_ci'))} | {b['bands']['high']['flagged']} / {f3(b['bands']['high']['precision'])} / "
                    f"{f3(b['bands']['high']['recall'])} | {b['bands']['watch_plus_high']['flagged']} / "
                    f"{f3(b['bands']['watch_plus_high']['precision'])} / {f3(b['bands']['watch_plus_high']['recall'])} |")
    return rows


def render(res: dict) -> str:
    c, cm, sc = res["counts"], res["comissing"], res["sales_check"]
    names = {f[0]: f for f in FEATURES}
    L = ["# Phase 7: Attr21 (sales growth) missingness", "",
         "Generated by scripts/run_phase7_attr21.py. Train and validation only. The frozen model, calibrator and cuts are unchanged, "
         "and the test split was not read. Patterns and counts only: the dataset has no company id, year or reason for missingness.", "",
         "Attr21 is defined as sales (n) / sales (n-1), so it cannot be computed without last year's sales (a property of the definition).", "",
         "## 1. How many firms, and how many are bankrupt", "",
         "| Split | Attr21 | Firms | Bankrupt | Rate (95% Wilson) | Share of all firms | Share of all bankrupt |", "|---|---|---|---|---|---|---|"]
    for s in ("train", "val"):
        for tag in ("missing", "present"):
            r = c[s][tag]
            sh = c[s]["missing_share_of_firms"] if tag == "missing" else 1 - c[s]["missing_share_of_firms"]
            sb = c[s]["missing_share_of_all_bankrupt"] if tag == "missing" else 1 - c[s]["missing_share_of_all_bankrupt"]
            L.append(f"| {s} | {tag} | {r['firms']} | {r['bankrupt']} | {f3(r['rate'])} ({f3(r['wilson_95'][0])} to {f3(r['wilson_95'][1])}) | {sh:.1%} | {sb:.1%} |")
    L += ["", "![Attr21](figures/phase7_attr21.png)", "", "## 2. What else is missing for those firms", "",
          "| Split | Attributes missing per firm, Attr21 missing (median, mean) | Attributes missing per firm, Attr21 present (median, mean) | Count distribution in the missing group (attributes missing: firms) |",
          "|---|---|---|---|"]
    for s in ("train", "val"):
        d = cm[s]
        dist = ", ".join(f"{k}: {v}" for k, v in d["count_distribution_missing_group"].items())
        L.append(f"| {s} | {d['median_missing_attributes']['missing']:.0f}, {d['mean_missing_attributes']['missing']:.2f} | "
                 f"{d['median_missing_attributes']['present']:.0f}, {d['mean_missing_attributes']['present']:.2f} | {dist} |")
    L += ["", "Ten other attributes whose missing rate differs most between the two groups (train):", "",
          "| Attribute | Missing rate when Attr21 missing | Missing rate when Attr21 present | Difference |", "|---|---|---|---|"]
    for t in cm["train"]["top_attributes"]:
        L.append(f"| {t['attr']}: {names[t['attr']][2]} | {t['rate_when_attr21_missing']:.3f} | {t['rate_when_attr21_present']:.3f} | {t['difference']:+.3f} |")
    L += ["", "## 3. Are current-year sales ratios available?", "",
          "Attr9 and Attr36 are sales / total assets. Among the Attr21-missing firms, counts of zero or missing (pattern only):", "",
          "| Split | Firms | Attr9 equal to 0 | Attr9 missing | Attr36 equal to 0 | Attr36 missing |", "|---|---|---|---|---|---|"]
    for s in ("train", "val"):
        d = sc[s]
        L.append(f"| {s} | {d['firms_missing']} | {d['sales_to_assets']['equal_to_zero']} | {d['sales_to_assets']['missing']} | "
                 f"{d['total_sales_to_assets']['equal_to_zero']} | {d['total_sales_to_assets']['missing']} |")
    L += ["", "## 4. Profile (train medians of the six most important ratios)", "",
          "| Group | Firms | " + " | ".join(f"{a}: {names[a][2]}" for a in res["profile_attrs"]) + " |",
          "|---|---|" + "---|" * len(res["profile_attrs"])]
    for g, d in res["profile_train"].items():
        L.append(f"| {g} | {d['firms']} | " + " | ".join(f3(d[a], 3) for a in res["profile_attrs"]) + " |")
    L += ["", "## 5. Model performance with Attr21 present vs missing", "",
          f"Cuts: Watch {res['bands']['t_watch']:.4f}, High {res['bands']['t_high']:.4f}. Bands cells show flagged firms / precision / recall within the subset. "
          "Validation cuts were fitted on validation (in-sample); train out-of-fold is the independent check. AUCs are not reported when a class has fewer than "
          f"{MIN_CLASS} firms.", "",
          "| Scores | Subset | Firms | Bankrupt | PR AUC (95% CI) | ROC AUC (95% CI) | High: flagged / precision / recall | Watch+High: flagged / precision / recall |",
          "|---|---|---|---|---|---|---|---|",
          *perf_rows("validation", res["performance"]["val"]), *perf_rows("train out-of-fold", res["performance"]["train_oof"]), "",
          "High band composition:", "", "| Scores | High firms | with Attr21 missing (bankrupt) | with Attr21 present (bankrupt) |", "|---|---|---|---|"]
    for name, key in (("validation", "val"), ("train out-of-fold", "train_oof")):
        h = res["performance"][key]["high_band_composition"]
        L.append(f"| {name} | {h['high_firms']} | {h['high_with_attr21_missing']} ({h['bankrupt_in_high_with_missing']}) | "
                 f"{h['high_with_attr21_present']} ({h['bankrupt_in_high_with_present']}) |")
    a = res["phase2_ablation"]
    L += ["", f"Phase 2 ablation (model retrained without Attr21, quoted from {a['source']}): validation PR AUC "
          f"{f3(a['pr_auc_attr21_dropped'])} against {f3(a['pr_auc_all_features'])} with all features.", "",
          "## 6. What-if on the frozen model (validation)", "",
          f"Attr21 changed for the model input only. The train median used for filling is {res['what_if_val']['train_median_attr21']:.4f}. "
          "This describes how the frozen model behaves. It is not a recommendation to impute.", "",
          "| Change | Firms | Low before to after | Watch before to after | High before to after | Firms that change band | Mean change, calibrated log-odds |",
          "|---|---|---|---|---|---|---|"]
    for tag, label in (("present_then_blanked", "Attr21 present, set to missing"),
                       ("missing_then_filled_with_train_median", "Attr21 missing, set to train median")):
        w = res["what_if_val"][tag]
        changed = sum(v for k, v in w["transitions"].items() if k.split("->")[0] != k.split("->")[1])
        L.append(f"| {label} | {w['firms']} | {w['band_counts_before']['Low']} to {w['band_counts_after']['Low']} | "
                 f"{w['band_counts_before']['Watch']} to {w['band_counts_after']['Watch']} | "
                 f"{w['band_counts_before']['High']} to {w['band_counts_after']['High']} | {changed} | {w['mean_change_calibrated_log_odds']:+.2f} |")
    L += ["", "## Flags", "", *[f"{i}. {f}" for i, f in enumerate(res["flags"], 1)], ""]
    return NL.join(L)


# ----------------------------------------------------------------------------------- run
def run(X_tr, y_tr, X_va, y_va, model, calibrator, bands, top_attrs, reports=REPORTS, figures=FIGURES, n_boot: int = N_BOOT) -> dict:
    reports, figures = Path(reports), Path(figures)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    res = compute(X_tr, y_tr, X_va, y_va, model, calibrator, bands, top_attrs, n_boot)
    res["flags"] = make_flags(res)
    plot(res, figures / "phase7_attr21.png")
    (reports / "phase7_attr21.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (reports / "phase7_attr21.md").write_text(render(res), encoding="utf-8")
    return res


def main() -> None:
    np.random.seed(SEED)
    verify_frozen()
    art = C.load_artifacts(ROOT / "artifacts" / "calibration")
    bands = art["bands"]["lightgbm"]
    calibrator = art["calibrators"]["lightgbm"][bands["selected_method"]]
    model = joblib.load(ROOT / "models" / "lightgbm.joblib")
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    imp = json.loads((ROOT / "artifacts" / "explain" / "global_importance.json").read_text())["features"]
    top_attrs = [f["attr"] for f in sorted(imp, key=lambda f: -f["mean_abs_val"])[:6]]
    res = run(X_tr, y_tr, X_va, y_va, model, calibrator, bands, top_attrs)
    print(f"wrote reports/phase7_attr21.md ({len(res['flags'])} flags)")


if __name__ == "__main__":
    main()
