"""Phase 6: SHAP explanations of the frozen LightGBM + Platt model (train and validation only).

Run: python -m scripts.run_phase6

Writes reports/phase6_explanations.{json,md}, reports/figures/phase6_*.png and
artifacts/explain/{feature_dictionary,global_importance,config}.json.
The test split is never loaded. The model, calibrator and cuts are frozen (hash checked against
artifacts/phase5_protocol.json). All randomness uses SEED = 42.
"""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import shap
from scipy.stats import rankdata

from src import calibrate as C
from src import explain as E
from src.plotting import BLUE, GRID, INK2, SURFACE, VERMILLION, apply_style, style_axes
from src.polish_features import CATEGORIES, FEATURES
from src.preprocess import ROOT
from src.train import load_split

SEED = 42
N_BOOT = 1000
NL = "\n"
MODEL = "lightgbm"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts" / "explain"
FROZEN = ("models/lightgbm.joblib", "artifacts/calibration/calibrators.json", "artifacts/calibration/bands.json")
TOP_TABLE = 15


def verify_frozen(root: Path = ROOT) -> None:
    """Abort unless the model, calibrators and bands are the files frozen in Phase 5."""
    proto = json.loads((root / "artifacts" / "phase5_protocol.json").read_text(encoding="utf-8"))
    bad = [rel for rel in FROZEN
           if hashlib.sha256((root / rel).read_bytes()).hexdigest() != proto["files"][rel]["sha256"]]
    if bad:
        raise RuntimeError(f"frozen artifact hash mismatch, aborting: {bad}")


# ------------------------------------------------------------------------------ compute
def pick_examples(y, p, t_watch: float) -> dict:
    """Rule fixed in docs/decisions.md: highest-scoring true positive, false alarm and nearest miss (validation rows)."""
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    miss = pos[p[pos] < t_watch]
    out = {"highest_scoring_true_positive": int(pos[np.argmax(p[pos])]),
           "highest_scoring_false_alarm": int(neg[np.argmax(p[neg])])}
    if len(miss):
        out["nearest_miss_below_watch_cut"] = int(miss[np.argmax(p[miss])])
    return out


def compute(X_tr, y_tr, X_va, y_va, model, calibrator, bands, n_boot: int = N_BOOT) -> dict:
    dic = E.feature_dictionary()
    tr, va = E.explain_dataset(model, calibrator, X_tr), E.explain_dataset(model, calibrator, X_va)
    if not (tr["check"]["ok"] and va["check"]["ok"]):  # report on the raw model scale, say so, never mix scales
        tr, va = E.explain_dataset(model, None, X_tr), E.explain_dataset(model, None, X_va)
    gain = model.booster_.feature_importance("gain")
    imp = E.importance(va["phi"], y_va, tr["phi"], gain, n_boot)
    p_va = calibrator.transform(model.predict_proba(X_va)[:, 1])
    explainer = shap.TreeExplainer(model)
    examples = {}
    for tag, i in pick_examples(y_va, p_va, bands["t_watch"]).items():
        e = E.explain_firm(model, calibrator, X_va.iloc[[i]], bands, top_k=6, explainer=explainer)
        examples[tag] = {"val_row": i, "bankrupt": int(np.asarray(y_va)[i]), **e}
    return {"dictionary": dic, "scale": va["scale"], "slope": va["slope"], "base_log_odds": va["base"],
            "base_probability": float(1 / (1 + np.exp(-va["base"]))) if va["scale"] == "calibrated_log_odds" else None,
            "check": {"train": tr["check"], "val": va["check"]},
            "additivity_error_raw": {"train": tr["additivity_error_raw"], "val": va["additivity_error_raw"]},
            "max_diff_vs_native": {"train": tr["max_diff_vs_native"], "val": va["max_diff_vs_native"]},
            "importance": imp, "gain": gain.tolist(),
            "attr21_missingness": E.missingness_table(X_tr, y_tr, X_va, y_va),
            "attr21_shap": {"train": E.shap_by_missing(tr["phi"], X_tr), "val": E.shap_by_missing(va["phi"], X_va)},
            "examples": examples, "_phi_val": va["phi"], "_phi_train": tr["phi"]}


# --------------------------------------------------------------------------------- plots
def scale_label(res: dict) -> str:
    return "calibrated log-odds" if res["scale"] == "calibrated_log_odds" else "raw model log-odds"


def plot_global(res: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    imp, dic = res["importance"], res["dictionary"]
    mv, mt, ci = np.array(imp["mean_abs_val"]), np.array(imp["mean_abs_train"]), np.array(imp["ci_val"])
    order = np.argsort(-mv)[:TOP_TABLE][::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    y = np.arange(len(order))
    ax.barh(y, mv[order], color=BLUE, alpha=0.9, label="Validation, mean |SHAP| (95% bootstrap CI)")
    ax.errorbar(mv[order], y, xerr=[mv[order] - ci[order, 0], ci[order, 1] - mv[order]], fmt="none", ecolor=INK2, lw=1)
    ax.scatter(mt[order], y, marker="D", color=VERMILLION, s=26, zorder=3, label="Train, mean |SHAP|")
    ax.set_yticks(y, [dic[j]["short_label"] for j in order], fontsize=7.5)
    ax.set_xlabel(f"Mean |SHAP| ({scale_label(res)})")
    ax.set_title("Feature importance, LightGBM + Platt  ([U] = UNVERIFIED definition)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_categories(res: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    cats = res["importance"]["category"]
    order = sorted(CATEGORIES, key=lambda c: cats[c]["mean_abs_sum_val"])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(order))
    v = np.array([cats[c]["mean_abs_sum_val"] for c in order])
    ci = np.array([cats[c]["ci_val"] for c in order])
    ax.barh(y, v, color=BLUE, alpha=0.9)
    ax.errorbar(v, y, xerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", ecolor=INK2, lw=1)
    ax.set_yticks(y, [f"{c} ({cats[c]['n_features']} features)" for c in order])
    ax.set_xlabel(f"Sum of member features' mean |SHAP| ({scale_label(res)}), validation")
    ax.set_title("Importance by project-assigned category", fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beeswarm(res: dict, X_va, out: Path, k: int = 12) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    phi, dic = res["_phi_val"], res["dictionary"]
    order = np.argsort(-np.array(res["importance"]["mean_abs_val"]))[:k][::-1]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    sc = None
    for row, j in enumerate(order):
        v = X_va.iloc[:, j].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        jit = rng.uniform(-0.3, 0.3, len(v))
        if ok.any():
            pct = rankdata(v[ok]) / ok.sum()
            sc = ax.scatter(phi[ok, j], row + jit[ok], c=pct, cmap="cividis", s=7, alpha=0.85, vmin=0, vmax=1)
        if (~ok).any():
            ax.scatter(phi[~ok, j], row + jit[~ok], marker="x", color=INK2, s=11, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(order)), [dic[j]["short_label"] for j in order], fontsize=7.5)
    ax.set_xlabel(f"SHAP value ({scale_label(res)}); right pushes the risk score up")
    ax.set_title("Validation firms by feature (colour = value percentile, x = value missing, [U] = UNVERIFIED)", fontsize=9)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, ticks=[0, 1], pad=0.01)
        cb.ax.set_yticklabels(["low", "high"])
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(res: dict, X_tr, X_va, y_va, out: Path) -> list:
    import matplotlib.pyplot as plt

    apply_style()
    phi, dic = res["_phi_val"], res["dictionary"]
    names = [f[1] for f in FEATURES]
    top = list(np.argsort(-np.array(res["importance"]["mean_abs_val"]))[:3])
    s21 = names.index("sales_growth")
    cols = top + ([] if s21 in top else [s21])
    y = np.asarray(y_va)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    for ax, j in zip(axes.ravel(), cols):
        v = X_va.iloc[:, j].to_numpy(dtype=float)
        tv = np.sort(X_tr.iloc[:, j].dropna().to_numpy(dtype=float))
        miss_x = -0.1
        x = np.where(np.isnan(v), miss_x, np.searchsorted(tv, np.nan_to_num(v), side="right") / len(tv))
        for label, m, color, marker, size in (("healthy", y == 0, BLUE, "o", 8), ("bankrupt", y == 1, VERMILLION, "^", 22)):
            ax.scatter(x[m], phi[m, j], c=color, marker=marker, s=size, alpha=0.6, label=label)
        ax.axhline(0, color=INK2, lw=0.8)
        pcts = [0.05, 0.25, 0.5, 0.75, 0.95]
        ax.set_xticks([miss_x] + pcts, ["missing"] + [f"{np.quantile(tv, q):.3g}{NL}(p{int(q * 100)})" for q in pcts], fontsize=6.5)
        ax.set_title(dic[j]["short_label"], fontsize=8)
        ax.set_ylabel(f"SHAP ({scale_label(res)})", fontsize=8)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Dependence on validation (x axis is the train percentile of the value, labels show the value)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [dic[j]["attr"] for j in cols]


# -------------------------------------------------------------------------------- report
def f3(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def make_flags(res: dict) -> list:
    imp, dic = res["importance"], res["dictionary"]
    top10 = np.argsort(-np.array(imp["mean_abs_val"]))[:10]
    unv = [dic[j]["attr"] for j in top10 if dic[j]["unverified"]]
    m = res["attr21_missingness"]
    flags = []
    if res["scale"] != "calibrated_log_odds":
        flags.append("The Platt slope conversion did NOT reproduce calibrated log-odds, so every SHAP value here is on the "
                     "raw model log-odds scale, not the calibrated scale.")
    flags += [
        "SHAP describes what the model does, not what causes bankruptcy. A large contribution is a model pattern, not a cause.",
        "Correlated ratios split credit, so one ratio can look less important because a close twin shares its signal. "
        "Known near twins: Attr2 and Attr17 (exact reciprocals), Attr9 and Attr36 (near duplicates). Category sums are only a partial remedy.",
        "The five categories are assigned by this project from each formula. They are not in the UCI description or the dictionary. "
        "Borderline cases are Attr27, Attr30, Attr33, Attr34, Attr37 and Attr58. The full map is in the appendix.",
        f"{len(unv)} of the 10 most important features have UNVERIFIED definitions ({', '.join(unv) if unv else 'none'}). "
        "Read their plain-language names as the UCI wording, which the dictionary marks as possibly mistranslated, unclear or inconsistent.",
        f"Importance stability: Spearman correlation of mean |SHAP| between train and validation is "
        f"{imp['spearman_train_vs_val']:.3f}, and {imp['top10_overlap_train_val']} of the top 10 features agree. "
        f"Agreement with LightGBM gain importance (Spearman) is {imp['spearman_gain_vs_val']:.3f}.",
        f"Cross-checks: SHAP additivity error against the model output is at most {max(res['additivity_error_raw'].values()):.1e}. "
        f"The shap library differs from LightGBM's native contributions by at most {max(res['max_diff_vs_native'].values()):.1e}; "
        "a difference of exactly zero suggests the library delegates to LightGBM's own routine for this model type, so that comparison is not an independent check.",
        "The frozen model is refit on all of train, while the Platt slope was fitted on fold models. The slope is applied to the "
        "refit model. Whether that mismatch matters is not verified.",
        "The base value is the model's average output over the training trees, not the observed bankruptcy rate, so it need not equal the prevalence.",
        "Example firms were chosen by a rule fixed in advance, and there is no company id, so they are identified by validation row number.",
        "Validation has 61 bankrupt firms and importance is an average over all 878 firms. The intervals show the sampling noise.",
        "Test split untouched in this phase.",
    ]
    ex_missing = [t for t, e in res["examples"].items() if e["drivers"][0]["attr"] == "Attr21" and e["drivers"][0]["value"] is None]
    if ex_missing:
        flags.append(f"In {len(ex_missing)} of the {len(res['examples'])} example firms ({', '.join(ex_missing)}) the largest driver is Attr21 being missing. "
                     "The model gives such firms a very large score on that signal alone.")
    flags.append(f"Attr21 missing vs present SHAP: mean abs SHAP is {f3(res['attr21_shap']['val']['missing']['mean_abs_shap'])} when missing "
                 f"vs {f3(res['attr21_shap']['val']['present']['mean_abs_shap'])} when present (validation; {res['attr21_shap']['val']['missing']['firms']} firms "
                 "have it missing). Pattern only.")
    for split in ("train", "val"):
        miss, pres = m[split]["missing"], m[split]["present"]
        if miss["rate"] is not None and pres["rate"] is not None:
            lo, hi = miss["wilson_95"], pres["wilson_95"]
            overlap = not (lo[1] < hi[0] or hi[1] < lo[0])
            flags.append(f"Attr21 {split}: bankruptcy rate {f3(miss['rate'])} when missing ({miss['firms']} firms) vs "
                         f"{f3(pres['rate'])} when present ({pres['firms']} firms); the 95% Wilson intervals "
                         f"{'overlap' if overlap else 'do not overlap'}. Pattern only.")
    return flags


def table_top(res: dict) -> list:
    imp, dic = res["importance"], res["dictionary"]
    mv, mt = imp["mean_abs_val"], imp["mean_abs_train"]
    total = imp["total_mean_abs_val"]
    lines = ["| Rank (95% CI) | Feature | Category | Mean abs SHAP, validation (95% CI) | Train | Share of total |",
             "|---|---|---|---|---|---|"]
    for j in np.argsort(-np.array(mv))[:TOP_TABLE]:
        rc, ci = imp["rank_ci_val"][j], imp["ci_val"][j]
        lines.append(f"| {imp['rank_val'][j]} ({rc[0]:.0f} to {rc[1]:.0f}) | {dic[j]['label']} | {dic[j]['category']} | "
                     f"{mv[j]:.3f} ({ci[0]:.3f} to {ci[1]:.3f}) | {mt[j]:.3f} | {mv[j] / total:.1%} |")
    return lines


def render_report(res: dict, bands: dict, dep_attrs: list) -> str:
    imp, dic, chk = res["importance"], res["dictionary"], res["check"]
    cats = imp["category"]
    m, ms = res["attr21_missingness"], res["attr21_shap"]
    L = ["# Phase 6: SHAP explanations (LightGBM + Platt, train and validation only)", "",
         "Generated by scripts/run_phase6.py. The model, calibrator and cuts are frozen (hash checked). The test split was not read. "
         f"Seed {SEED}, {N_BOOT} stratified bootstrap resamples over validation firms.", "",
         "## Scale, and the Platt slope check", "",
         f"SHAP values are reported in **{scale_label(res)}**. Platt is sigmoid(a * logit(score) + b), so SHAP in raw log-odds converts "
         f"by multiplying by a = {res['slope']:.4f} when the calibrator was fitted on raw log-odds. The conversion was verified, not assumed:", "",
         "| Split | Firms | Max error vs calibrated log-odds | Tolerance | Clipped scores | Passed |", "|---|---|---|---|---|---|"]
    for s in ("train", "val"):
        c = chk[s]
        L.append(f"| {s} | {c['n']} | {c['max_abs_error']:.2e} | {c['tol']:.0e} | {c['clipped']} | {c['ok']} |"
                 if c["max_abs_error"] is not None else f"| {s} | {c['n']} | n/a | n/a | n/a | False ({c['reason']}) |")
    L += ["", f"Base value: {res['base_log_odds']:.3f} log-odds"
          + (f" (probability {res['base_probability']:.4f})." if res["base_probability"] is not None else "."), "",
          "## Global importance", "", f"Mean absolute SHAP over validation firms, top {TOP_TABLE} of 64. Feature names are the UCI formula text "
          "with the attribute code. UNVERIFIED marks definitions the dictionary itself flags as possibly mistranslated, unclear or inconsistent.", "",
          *table_top(res), "", "![Importance](figures/phase6_importance.png)", "",
          "![Beeswarm](figures/phase6_beeswarm.png)", "",
          f"Dependence panels cover {', '.join(dep_attrs)} (the three most important features, with Attr21 added if it is not among them). ![Dependence](figures/phase6_dependence.png)", "",
          "## Importance by category", "",
          "Category importance is the sum of its features' mean absolute SHAP. **Correlated ratios split credit**, so single "
          "features understate themselves when a close twin exists, and categories are only a partial remedy. The categories are "
          "assigned by this project from each formula (the dictionary and UCI give none). Borderline: Attr27, Attr30, Attr33, Attr34, Attr37, Attr58.", "",
          "| Category | Features | Sum of mean abs SHAP, validation (95% CI) | Share | Train sum |", "|---|---|---|---|---|"]
    for c in sorted(CATEGORIES, key=lambda c: -cats[c]["mean_abs_sum_val"]):
        v = cats[c]
        L.append(f"| {c} | {v['n_features']} | {v['mean_abs_sum_val']:.3f} ({v['ci_val'][0]:.3f} to {v['ci_val'][1]:.3f}) | "
                 f"{v['share_val']:.1%} | {v['mean_abs_sum_train']:.3f} |")
    border = sum(imp["mean_abs_val"][i] for i, f in enumerate(FEATURES) if f[0] in E.BORDERLINE) / imp["total_mean_abs_val"]
    L += ["", f"The six borderline features ({', '.join(E.BORDERLINE)}) carry {border:.1%} of total importance, so that share of the category "
          "table depends on project judgement.", "", "![Categories](figures/phase6_categories.png)", "", "## Attr21 (sales growth): missing vs present", "",
          "Bankruptcy rate with Attr21 missing vs present, with 95% Wilson intervals. The pattern is stated only, with no causal story.", "",
          "| Split | Attr21 | Firms | Bankrupt | Rate (95% Wilson) | Mean SHAP | Mean abs SHAP |", "|---|---|---|---|---|---|---|"]
    for s in ("train", "val"):
        for tag in ("missing", "present"):
            r, sh = m[s][tag], ms[s][tag]
            L.append(f"| {s} | {tag} | {r['firms']} | {r['bankrupt']} | {f3(r['rate'])} ({f3(r['wilson_95'][0])} to {f3(r['wilson_95'][1])}) | "
                     f"{f3(sh['mean_shap'])} | {f3(sh['mean_abs_shap'])} |")
    L += ["", "## Example firms (rule fixed in advance)", "",
          "From validation: the highest-scoring true positive, the highest-scoring false alarm, and the highest-scoring bankrupt firm "
          f"below the Watch cut ({bands['t_watch']:.4f}). Contributions are in {scale_label(res)}; positive raises the risk score.", ""]
    for tag, e in res["examples"].items():
        L += [f"### {tag.replace('_', ' ')} (validation row {e['val_row']})", "",
              f"Bankrupt: {'yes' if e['bankrupt'] else 'no'}. Calibrated probability {e['probability']:.3f}, band {e['band']}.", ""]
        L += [f"- {d['sentence']}" for d in e["drivers"]] + [""]
    L += ["## Flags", "", *[f"{i}. {f}" for i, f in enumerate(make_flags(res), 1)], "",
          "## Appendix: feature dictionary with category and UNVERIFIED marking", "",
          "| Attribute | Formula (UCI wording) | Category | Definition status | Dictionary note |", "|---|---|---|---|---|"]
    for d in dic:
        L.append(f"| {d['attr']} | {d['formula']} | {d['category']} | {'UNVERIFIED' if d['unverified'] else 'as documented'} | {d['note'] or ''} |")
    L.append("")
    return NL.join(L)


# ---------------------------------------------------------------------------------- run
def run(X_tr, y_tr, X_va, y_va, model, calibrator, bands, reports=REPORTS, artifacts=ARTIFACTS,
        figures=FIGURES, n_boot: int = N_BOOT) -> dict:
    reports, artifacts, figures = Path(reports), Path(artifacts), Path(figures)
    for d in (reports, artifacts, figures):
        d.mkdir(parents=True, exist_ok=True)
    res = compute(X_tr, y_tr, X_va, y_va, model, calibrator, bands, n_boot)
    plot_global(res, figures / "phase6_importance.png")
    plot_categories(res, figures / "phase6_categories.png")
    plot_beeswarm(res, X_va, figures / "phase6_beeswarm.png")
    dep_attrs = plot_dependence(res, X_tr, X_va, y_va, figures / "phase6_dependence.png")
    dic, imp = res["dictionary"], res["importance"]
    (artifacts / "feature_dictionary.json").write_text(json.dumps(dic, indent=1), encoding="utf-8")
    glob = [{"attr": d["attr"], "category": d["category"], "unverified": d["unverified"],
             "mean_abs_val": imp["mean_abs_val"][j], "mean_abs_train": imp["mean_abs_train"][j],
             "ci_val": imp["ci_val"][j], "rank_val": imp["rank_val"][j]} for j, d in enumerate(dic)]
    (artifacts / "global_importance.json").write_text(json.dumps(
        {"scale": res["scale"], "features": glob, "category": imp["category"]}, indent=1), encoding="utf-8")
    config = {"seed": SEED, "bootstrap_n": n_boot, "scale": res["scale"], "platt_slope_a": res["slope"],
              "slope_check": res["check"], "shap_version": shap.__version__, "python": platform.python_version(),
              "fit_data": "none (explains the frozen model on train and validation)", "test_split_read": False}
    (artifacts / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    out = {k: v for k, v in res.items() if not k.startswith("_phi")}
    out["flags"] = make_flags(res)
    (reports / "phase6_explanations.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (reports / "phase6_explanations.md").write_text(render_report(res, bands, dep_attrs), encoding="utf-8")
    return out


def main() -> None:
    np.random.seed(SEED)
    verify_frozen()
    art = C.load_artifacts(ROOT / "artifacts" / "calibration")
    bands = art["bands"][MODEL]
    calibrator = art["calibrators"][MODEL][bands["selected_method"]]
    model = joblib.load(ROOT / "models" / "lightgbm.joblib")
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    res = run(X_tr, y_tr, X_va, y_va, model, calibrator, bands)
    imp = res["importance"]
    print(f"scale={res['scale']} slope check val ok={res['check']['val']['ok']} "
          f"spearman train/val={imp['spearman_train_vs_val']:.3f}; wrote reports/phase6_explanations.md")


if __name__ == "__main__":
    main()
