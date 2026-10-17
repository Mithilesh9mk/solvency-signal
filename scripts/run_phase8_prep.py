"""Phase 8 prep: numbers the app shows, the blank-attribute report and the demo firms (train and validation only).

Run: python -m scripts.run_phase8_prep

Writes reports/phase8_app_numbers.json, reports/phase8_app_prep.md, reports/figures/phase8_reliability_present.png
and app/demo_firms.json. The frozen model, calibrator and cuts are unchanged and the test split is never loaded.
Rules are fixed in docs/decisions.md (Phase 8 protocol). All randomness uses SEED = 42.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_phase6 import verify_frozen
from src import app_logic as A
from src import calibrate as C
from src import explain as E
from src.plotting import BLUE, INK2, VERMILLION, apply_style, style_axes
from src.polish_features import FEATURES
from src.preprocess import ROOT
from src.train import load_split, make_lgbm

SEED = 42
N_BOOT = 1000
NL = "\n"
KEY = "sales_growth"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DEMO = ROOT / "app" / "demo_firms.json"
BANDS = ("Low", "Watch", "High")


# ------------------------------------------------------------------------------ numbers
def pct(draws) -> list:
    return np.nanpercentile(draws, [2.5, 97.5]).tolist()


def subset_block(y, p, bands: dict, n_boot: int = N_BOOT) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    t_w, t_h = bands["t_watch"], bands["t_high"]
    pos = int(y.sum())
    idx = C.bootstrap_indices(y, n_boot, SEED)
    d = {k: [] for k in ("pr", "roc", "ece", "hp", "hr", "fp", "fr")}
    for i in idx:
        yy, pp = y[i], p[i]
        d["pr"].append(average_precision_score(yy, pp))
        d["roc"].append(roc_auc_score(yy, pp))
        d["ece"].append(C.ece(yy, pp))
        pos_i = max(int(yy.sum()), 1)
        for tag, t in (("h", t_h), ("f", t_w)):
            f = pp >= t
            tp = int((f & (yy == 1)).sum())
            d[tag + "p"].append(tp / f.sum() if f.sum() else np.nan)
            d[tag + "r"].append(tp / pos_i)
    out = {"firms": int(len(y)), "bankrupt": pos, "no_skill_pr_auc": float(y.mean()),
           "pr_auc": float(average_precision_score(y, p)), "pr_auc_ci": pct(d["pr"]),
           "roc_auc": float(roc_auc_score(y, p)), "roc_auc_ci": pct(d["roc"]),
           "ece": C.ece(y, p), "ece_ci": pct(d["ece"]), "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
           "reliability": C.reliability_points(y, p)}
    for tag, key, t in (("h", "high", t_h), ("f", "watch_plus_high", t_w)):
        f = p >= t
        tp = int((f & (y == 1)).sum())
        out[key] = {"flagged": int(f.sum()), "bankrupt": tp, "precision": tp / f.sum() if f.sum() else None,
                    "precision_ci": pct(d[tag + "p"]), "recall": tp / pos, "recall_ci": pct(d[tag + "r"])}
    out["band_table"] = [{**r, "wilson_95": E.wilson(r["bankrupt"], r["firms"])} for r in C.band_table(y, p, t_w, t_h)]
    return out


def split_numbers(X, y, p, bands) -> dict:
    present = ~X[KEY].isna().to_numpy()
    y, p = np.asarray(y), np.asarray(p)
    return {"present": subset_block(y[present], p[present], bands), "all": subset_block(y, p, bands),
            "attr21_missing_firms": int((~present).sum()), "attr21_missing_bankrupt": int(y[~present].sum())}


def blank_report(X, y, limit: int = A.MAX_OTHER_BLANKS) -> dict:
    """Counts of blanks. Attr21-present firms are the ones the app would score."""
    y = np.asarray(y)
    present = ~X[KEY].isna().to_numpy()
    Xp, yp = X[present], y[present]
    other = Xp.drop(columns=[KEY])
    n_other = other.isna().sum(axis=1).to_numpy()
    names = {f[1]: f[0] for f in FEATURES}
    per = other.isna().sum()
    feats = [{"attr": names[c], "name": c, "blank_firms": int(n), "blank_rate": float(n / len(Xp))}
             for c, n in per.sort_values(ascending=False).items() if n > 0]
    over = n_other > limit
    return {"firms": int(len(X)), "attr21_blank_firms": int((~present).sum()),
            "attr21_present_firms": int(present.sum()),
            "present_firms_with_any_other_blank": int((n_other > 0).sum()),
            "present_firms_with_any_other_blank_share": float((n_other > 0).mean()),
            "other_blank_count_distribution": {int(k): int(v) for k, v in zip(*np.unique(n_other, return_counts=True))},
            "limit": limit, "present_firms_over_limit": int(over.sum()), "bankrupt_among_over_limit": int(yp[over].sum()),
            "features_with_blanks": feats}


def blank_effects(X_tr, y_tr, X_va, y_va, model, calibrator, min_firms: int = 20) -> dict:
    """Descriptive history for each attribute (other than Attr21) blank in at least min_firms Attr21-present train firms."""
    names = {f[1]: f[0] for f in FEATURES}
    a = calibrator.a if isinstance(calibrator, C.PlattCalibrator) else 1.0
    present = ~X_tr[KEY].isna().to_numpy()
    contrib = model.predict(X_tr[present], pred_contrib=True)[:, :-1] * a
    cols = list(X_tr.columns)
    out = {}
    for c in cols:
        if c == KEY:
            continue
        blank_tr = X_tr.loc[present, c].isna().to_numpy()
        if blank_tr.sum() < min_firms:
            continue
        rec = {}
        for split, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va)):
            keep = ~X[KEY].isna().to_numpy()
            yy, bl = np.asarray(y)[keep], X.loc[keep, c].isna().to_numpy()
            rec[split] = {tag: {"firms": int(m.sum()), "bankrupt": int(yy[m].sum()), "rate": float(yy[m].mean()) if m.any() else None,
                                "wilson_95": E.wilson(int(yy[m].sum()), int(m.sum()))}
                          for tag, m in (("blank", bl), ("present", ~bl))}
        j = cols.index(c)
        rec["mean_contribution_when_blank_train"] = float(contrib[blank_tr, j].mean())
        rec["mean_contribution_when_present_train"] = float(contrib[~blank_tr, j].mean())
        out[names[c]] = rec
    return out


def pick_demo(X, y, p, bands, blocked: dict = None) -> dict:
    """Phase 6 rule restricted to validation firms the app would score (Phase 9 refusal rules included)."""
    y, p = np.asarray(y), np.asarray(p)
    blocked = A.blocked_blanks() if blocked is None else blocked
    names = {f[1]: f[0] for f in FEATURES}
    present = np.array([A.assess(r, blocked=blocked)["status"] == "scored" for _, r in X.rename(columns=names).iterrows()])
    pos, neg = np.flatnonzero((y == 1) & present), np.flatnonzero((y == 0) & present)
    miss = pos[p[pos] < bands["t_watch"]]
    picks = {"highest_scoring_true_positive": int(pos[np.argmax(p[pos])]),
             "highest_scoring_false_alarm": int(neg[np.argmax(p[neg])])}
    if len(miss):
        picks["nearest_miss_below_watch_cut"] = int(miss[np.argmax(p[miss])])
    return picks


def demo_payload(X_va, y_va, p_va, bands, picks: dict) -> dict:
    names = {f[1]: f[0] for f in FEATURES}
    firms = []
    for role, i in picks.items():
        row = X_va.iloc[i]
        values = {names[c]: (None if np.isnan(row[c]) else float(row[c])) for c in X_va.columns}
        assess = A.assess(__import__("pandas").Series(values, dtype=float))
        firms.append({"id": f"validation row {i}", "role": role.replace("_", " "), "val_row": i,
                      "dataset_label": "bankrupt within the year" if int(y_va.iloc[i]) else "still operating",
                      "model_probability": float(p_va[i]), "model_band": str(C.assign_bands([p_va[i]], bands["t_watch"], bands["t_high"])[0]),
                      "n_other_blank": assess["n_other_blank"], "would_be_scored": assess["status"] == "scored",
                      "values": values})
    return {"rule": "Phase 6 rule restricted to validation firms the app would score (Attr21 present, no refused blank, at most 6 other "
                    "blanks): highest-scoring true positive, highest-scoring false alarm, highest-scoring bankrupt firm below the Watch cut. "
                    "Rows are from the validation split of the "
                    "Polish 5year data (UCI, CC BY 4.0), identified by row number, with no company id.",
            "firms": firms}


# ---------------------------------------------------------------------------------- plot
def plot_reliability(nums: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lim = 0
    for key, label, color, ls, marker in (("val", "Validation, Attr21 present", VERMILLION, "--", "s"),
                                          ("train_oof", "Train out-of-fold, Attr21 present", BLUE, "-", "o")):
        pts = nums[key]["present"]["reliability"]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        lim = max(lim, max(xs), max(ys))
        ax.plot(xs, ys, color=color, ls=ls, marker=marker, lw=1.5, label=label)
    lim *= 1.1
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls=":", label="Perfect calibration")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted probability (10 equal-frequency bins)")
    ax.set_ylabel("Observed bankruptcy rate")
    ax.set_title("Reliability on firms with Attr21 present (report only)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------- report
def f3(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def ci(v, c, nd=3):
    return f"{f3(v, nd)} ({f3(c[0], nd)} to {f3(c[1], nd)})"


def make_flags(nums: dict, bands: dict, blanks: dict, effects: dict, demo: dict) -> list:
    v, o = nums["val"], nums["train_oof"]
    floor = C.BAND_RULE["high_min_precision"]
    top = blanks["train"]["features_with_blanks"][0]
    strong = [a for a, e in effects.items() if e["mean_contribution_when_blank_train"] > 1.0]
    flags = [
        (f"Blanks in other attributes can carry signal too. In train, a blank {', '.join(strong)} adds more than 1.0 calibrated log-odds on average "
         "to that attribute's contribution. The app's warning therefore also shows the train and validation history for each blank attribute. "
         f"{sum(1 for f in demo['firms'] if f['n_other_blank'] > 0)} of the {len(demo['firms'])} demo firms have a blank attribute. "
         "This is descriptive and is not a test of the effect.") if strong else
        "No attribute other than Attr21 adds more than 1.0 calibrated log-odds on average when blank (train).",
        f"A blank-attribute warning will appear for about half of scorable firms: {blanks['train']['present_firms_with_any_other_blank_share']:.1%} of train and "
        f"{blanks['val']['present_firms_with_any_other_blank_share']:.1%} of validation firms with Attr21 present have at least one other blank, most often "
        f"{top['attr']} ({top['blank_rate']:.1%} of train firms). The limit of {blanks['train']['limit']} would block {blanks['train']['present_firms_over_limit']} train and "
        f"{blanks['val']['present_firms_over_limit']} validation firms. The effect of these blanks on the score is untested.",
        f"The cuts were set on all firms. High precision on firms with Attr21 present is {f3(v['present']['high']['precision'])} on validation and "
        f"{f3(o['present']['high']['precision'])} on train out-of-fold, against {f3(v['all']['high']['precision'])} and {f3(o['all']['high']['precision'])} "
        f"for all firms, and the band rule's floor is {floor:.2f}. Nothing was re-cut or recalibrated.",
        f"Calibration on firms with Attr21 present is report only: mean predicted probability {f3(v['present']['mean_predicted'], 4)} against an observed rate of "
        f"{f3(v['present']['observed_rate'], 4)} on validation, and {f3(o['present']['mean_predicted'], 4)} against {f3(o['present']['observed_rate'], 4)} on train out-of-fold. "
        f"ECE is {ci(v['present']['ece'], v['present']['ece_ci'], 4)} on validation and {ci(o['present']['ece'], o['present']['ece_ci'], 4)} out-of-fold. "
        "Bootstrap ECE is biased upward on small data, so the point estimate can sit outside its own interval.",
        f"Validation Watch band on firms with Attr21 present has {v['present']['band_table'][1]['firms']} firms, so its rate carries very little information.",
        "Train out-of-fold scores use the saved Platt calibrator that was fitted on those same scores, so the out-of-fold calibration numbers are mildly optimistic.",
        "The validation cuts were fitted on validation, so validation precision and recall at the cuts are in-sample. Train out-of-fold is the independent check.",
        "Sector composition of the Polish data is not stated on the UCI page, so the sector is UNVERIFIED. The data is a random split, with no company id or year.",
        "Test split untouched in this phase. No model, calibrator or cut was changed.",
    ]
    return flags


def block_rows(label: str, d: dict) -> list:
    rows = []
    for tag, name in (("present", "Attr21 present"), ("all", "All firms")):
        b = d[tag]
        rows.append(f"| {label} | {name} | {b['firms']} | {b['bankrupt']} | {b['no_skill_pr_auc']:.3f} | {ci(b['pr_auc'], b['pr_auc_ci'])} | "
                    f"{ci(b['roc_auc'], b['roc_auc_ci'])} | {ci(b['high']['precision'], b['high']['precision_ci'])} | "
                    f"{ci(b['high']['recall'], b['high']['recall_ci'])} | {ci(b['watch_plus_high']['recall'], b['watch_plus_high']['recall_ci'])} |")
    return rows


def render(nums: dict, blanks: dict, demo: dict, bands: dict, flags: list, effects: dict) -> str:
    L = ["# Phase 8 prep: app numbers, blank-attribute report and demo firms", "",
         "Generated by scripts/run_phase8_prep.py. Train and validation only, frozen model, calibrator and cuts unchanged, test split not read. "
         "Rules were fixed in docs/decisions.md (Phase 8 protocol) before this ran.", "",
         "## 1. Performance on firms with Attr21 present, beside all firms", "",
         f"Cuts: Watch {bands['t_watch']:.4f}, High {bands['t_high']:.4f}. No-skill PR AUC is the share of bankrupt firms in the subset. "
         "Validation cuts were fitted on validation (in-sample), train out-of-fold is the independent check.", "",
         "| Scores | Subset | Firms | Bankrupt | No-skill PR AUC | PR AUC (95% CI) | ROC AUC (95% CI) | High precision (95% CI) | High recall (95% CI) | Watch+High recall (95% CI) |",
         "|---|---|---|---|---|---|---|---|---|---|",
         *block_rows("validation", nums["val"]), *block_rows("train out-of-fold", nums["train_oof"]), "",
         f"Attr21 missing: {nums['val']['attr21_missing_firms']} validation firms ({nums['val']['attr21_missing_bankrupt']} bankrupt) and "
         f"{nums['train_oof']['attr21_missing_firms']} train firms ({nums['train_oof']['attr21_missing_bankrupt']} bankrupt). "
         "The all-firms figures include them, which flatters them.", "",
         "## 2. Historical bankrupt rate per band, firms with Attr21 present", "",
         "| Scores | Band | Firms | Bankrupt | Rate (95% Wilson) |", "|---|---|---|---|---|"]
    for label, key in (("validation", "val"), ("train out-of-fold", "train_oof")):
        for r in nums[key]["present"]["band_table"]:
            L.append(f"| {label} | {r['band']} | {r['firms']} | {r['bankrupt']} | {f3(r['bankrupt_rate'])} ({f3(r['wilson_95'][0])} to {f3(r['wilson_95'][1])}) |")
    L += ["", "## 3. Calibration on firms with Attr21 present (report only)", "",
          "| Scores | Mean predicted | Observed rate | ECE (95% CI) |", "|---|---|---|---|"]
    for label, key in (("validation", "val"), ("train out-of-fold", "train_oof")):
        b = nums[key]["present"]
        L.append(f"| {label} | {b['mean_predicted']:.4f} | {b['observed_rate']:.4f} | {ci(b['ece'], b['ece_ci'], 4)} |")
    L += ["", "![Reliability](figures/phase8_reliability_present.png)", "",
          "## 4. Blank attributes in train and validation", "",
          f"Rule fixed before looking: with Attr21 present, more than {blanks['train']['limit']} of the other 63 attributes blank blocks scoring. "
          "Counts only. Features are blank in the raw data and the app would score them with LightGBM's native handling, with a warning.", "",
          "| Split | Firms | Attr21 blank (not scored) | Attr21 present | With any other blank | Share | Over the limit (blocked) | Bankrupt among those |",
          "|---|---|---|---|---|---|---|---|"]
    for s in ("train", "val"):
        b = blanks[s]
        L.append(f"| {s} | {b['firms']} | {b['attr21_blank_firms']} | {b['attr21_present_firms']} | {b['present_firms_with_any_other_blank']} | "
                 f"{b['present_firms_with_any_other_blank_share']:.1%} | {b['present_firms_over_limit']} | {b['bankrupt_among_over_limit']} |")
    L += ["", "Distribution of other blank attributes per firm (Attr21 present):", ""]
    for s in ("train", "val"):
        L.append(f"- {s}: " + ", ".join(f"{k} blank: {v} firms" for k, v in blanks[s]["other_blank_count_distribution"].items()))
    names = {f[0]: f for f in FEATURES}
    L += ["", "Attributes with blanks, Attr21-present firms (train and validation):", "",
          "| Attribute | Train blank firms (rate) | Validation blank firms (rate) |", "|---|---|---|"]
    tr = {f["attr"]: f for f in blanks["train"]["features_with_blanks"]}
    va = {f["attr"]: f for f in blanks["val"]["features_with_blanks"]}
    for a in sorted(set(tr) | set(va), key=lambda a: -(tr.get(a, {}).get("blank_rate", 0))):
        t, v = tr.get(a), va.get(a)
        L.append(f"| {a}: {names[a][2]} | {t['blank_firms'] if t else 0} ({t['blank_rate'] if t else 0:.3f}) | "
                 f"{v['blank_firms'] if v else 0} ({v['blank_rate'] if v else 0:.3f}) |")
    L += ["", "Descriptive history for attributes blank in at least 20 Attr21-present train firms (not a test of the effect):", "",
          "| Attribute | Train: bankrupt when blank | Train: when present | Validation: bankrupt when blank | Validation: when present | Mean contribution when blank (train) | When present (train) |",
          "|---|---|---|---|---|---|---|"]
    for a, e in effects.items():
        t, v = e["train"], e["val"]
        L.append(f"| {a} | {t['blank']['bankrupt']} of {t['blank']['firms']} ({f3(t['blank']['rate'])}) | {f3(t['present']['rate'])} | "
                 f"{v['blank']['bankrupt']} of {v['blank']['firms']} ({f3(v['blank']['rate'])}) | {f3(v['present']['rate'])} | "
                 f"{e['mean_contribution_when_blank_train']:+.2f} | {e['mean_contribution_when_present_train']:+.2f} |")
    L += ["", "## 5. Demo firms (Phase 6 rule, restricted to validation firms the app would score)", "", demo["rule"], "",
          "| Role | Validation row | Dataset label | Model probability | Band | Other blanks | Would be scored |", "|---|---|---|---|---|---|---|"]
    for f in demo["firms"]:
        L.append(f"| {f['role']} | {f['val_row']} | {f['dataset_label']} | {f['model_probability']:.3f} | {f['model_band']} | {f['n_other_blank']} | {f['would_be_scored']} |")
    L += ["", "## Flags", "", *[f"{i}. {f}" for i, f in enumerate(flags, 1)], ""]
    return NL.join(L)


# ----------------------------------------------------------------------------------- run
def run(X_tr, y_tr, X_va, y_va, model, calibrator, bands, reports=REPORTS, figures=FIGURES, demo_path=DEMO,
        n_boot: int = N_BOOT) -> dict:
    reports, figures, demo_path = Path(reports), Path(figures), Path(demo_path)
    for d in (reports, figures, demo_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    oof = calibrator.transform(C.oof_predictions(make_lgbm, X_tr, y_tr))
    p_va = calibrator.transform(model.predict_proba(X_va)[:, 1])
    nums = {"val": split_numbers(X_va, y_va, p_va, bands), "train_oof": split_numbers(X_tr, y_tr, oof, bands)}
    blanks = {"train": blank_report(X_tr, y_tr), "val": blank_report(X_va, y_va)}
    demo = demo_payload(X_va, y_va, p_va, bands, pick_demo(X_va, y_va, p_va, bands))
    effects = blank_effects(X_tr, y_tr, X_va, y_va, model, calibrator)
    flags = make_flags(nums, bands, blanks, effects, demo)
    out = {"numbers": nums, "blanks": blanks, "blank_effects": effects, "bands": {"t_watch": bands["t_watch"], "t_high": bands["t_high"]},
           "blank_limit": A.MAX_OTHER_BLANKS, "flags": flags, "seed": SEED, "bootstrap_n": n_boot}
    plot_reliability(nums, figures / "phase8_reliability_present.png")
    (reports / "phase8_app_numbers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (reports / "phase8_app_prep.md").write_text(render(nums, blanks, demo, bands, flags, effects), encoding="utf-8")
    demo_path.write_text(json.dumps(demo, indent=1), encoding="utf-8")
    return out


def main() -> None:
    np.random.seed(SEED)
    verify_frozen()
    art = C.load_artifacts(ROOT / "artifacts" / "calibration")
    bands = art["bands"]["lightgbm"]
    calibrator = art["calibrators"]["lightgbm"][bands["selected_method"]]
    model = joblib.load(ROOT / "models" / "lightgbm.joblib")
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    out = run(X_tr, y_tr, X_va, y_va, model, calibrator, bands)
    phase4 = json.loads((REPORTS / "phase4_calibration.json").read_text())["models"]["lightgbm"]["val"]["platt"]
    assert abs(out["numbers"]["val"]["all"]["pr_auc"] - phase4["pr_auc"]) < 1e-9, "all-firms PR AUC does not match Phase 4"
    print(f"wrote reports/phase8_app_prep.md and app/demo_firms.json ({len(out['flags'])} flags)")


if __name__ == "__main__":
    main()
