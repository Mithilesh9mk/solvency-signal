"""Phase 4: calibration, thresholds and risk bands (train and validation only).

Run: python -m scripts.run_phase4

Writes reports/phase4_calibration.{json,md}, reports/phase4_pr_tradeoff.csv,
reports/figures/phase4_*.png and artifacts/calibration/{calibrators,bands,config}.json.
The test split is never loaded. All randomness uses SEED = 42.
"""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import calibrate as C
from src.plotting import GRID, INK2, METHOD_STYLE, MODEL_STYLE, SURFACE, apply_style, style_axes
from src.preprocess import PROCESSED, ROOT
from src.train import FACTORIES, load_split

SEED = 42
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts" / "calibration"
MODELS = ROOT / "models"
N_BOOT = 1000

RECOMMENDED = "lightgbm"  # written after reading the validation comparison; the user makes the final call


def build_recommendation(res: dict) -> str:
    """Recommendation text. The choice is Claude's judgement, every number in it comes from `res`."""
    M = res["models"]
    if not {RECOMMENDED, "neural_net"} <= set(M):
        return "(No recommendation: this run did not include every candidate model.)"
    nm = {n: MODEL_STYLE[n][0] for n in M}
    r, o = M[RECOMMENDED], M["neural_net"]
    rv, ov = r["val"][r["selected_method"]], o["val"][o["selected_method"]]
    lg = r["val"]["platt"]["delta_vs_none"]["log_loss"]
    n_pos = res["n_val_pos"]
    hp_r, hp_o = (next(x for x in m["bands"]["val_table"] if x["band"] == "High")["bankrupt_rate"] for m in (r, o))
    return "\n".join([
        f"**Recommendation (Claude's, the decision is yours): {nm[RECOMMENDED]} with {r['selected_method']} calibration "
        f"as the headline model, and the neural network kept as the documented neural benchmark.**",
        "",
        f"- On validation it has the best PR AUC ({rv['pr_auc']:.3f} vs {ov['pr_auc']:.3f} for the neural network), "
        f"Brier ({rv['brier']:.4f} vs {ov['brier']:.4f}) and log loss ({rv['log_loss']:.4f} vs {ov['log_loss']:.4f}) "
        "of the calibrated models, and the same order on train CV (Phase 3).",
        f"- Calibration helps it most: Platt changes its validation log loss by "
        f"{r['val']['platt']['log_loss'] - r['val']['none']['log_loss']:+.4f} (95% CI {lg['ci'][0]:+.4f} to "
        f"{lg['ci'][1]:+.4f}), a change whose interval excludes zero. Raw LightGBM scores are not probabilities.",
        "",
        "**Risks of choosing it:**",
        f"1. The PR AUC lead over the neural network is borderline at this sample size ({n_pos} "
        "validation positives; the Phase 3 paired CI upper bound was just below zero). It is a lean, not a proof.",
        "2. It relies heavily on `sales_growth` (Attr21, needs last year's sales) and `Attr27` (undefined when "
        "financial expenses are zero), see the Phase 2 flags. A product needs a plan for firms missing them.",
        f"3. Its Platt calibration is a real correction (slope {r['platt']['a']:.2f}); the calibrator was fitted on fold "
        "models, not on the saved model, so some mismatch is possible.",
        "4. Under the approved band rule its Watch band is empty and High is broad (see Risk bands). That is a property of "
        "the rule on this data, not specific to LightGBM, but the user still needs to settle the band rule.",
        "5. Random split, no time ordering, so all of this may overstate performance on future years.",
        "",
        "**Why you might pick the neural network instead:** it is the project's stated concept, it is well calibrated "
        f"without any calibrator (validation ECE {ov['ece']:.4f}), and under the approved rule its High band had "
        f"validation precision {hp_o:.3f} against {hp_r:.3f} for {nm[RECOMMENDED]}. The price is a lower PR AUC "
        f"({ov['pr_auc']:.3f}).",
        "",
        "**Decision needed:** (a) headline model, (b) band rule: keep the approved one, adopt the post-hoc 0.50 "
        "alternative, or define another.",
    ])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(X_tr, y_tr, X_va, y_va, final_models: dict, names: list, n_boot: int) -> dict:
    idx = C.bootstrap_indices(y_va, n=n_boot)
    res = {"models": {}, "_arrays": {}, "n_val_pos": int(np.sum(y_va))}
    for name in names:
        oof = C.oof_predictions(FACTORIES[name], X_tr, y_tr)
        sel = C.select_method(oof, y_tr)
        cals = {m: C.make_calibrator(m).fit(oof, y_tr) for m in C.METHODS}
        raw_va = final_models[name].predict_proba(X_va)[:, 1]
        oof_cal = {m: cals[m].transform(oof) for m in C.METHODS}
        va_cal = {m: cals[m].transform(raw_va) for m in C.METHODS}
        boot = C.bootstrap_metrics(y_va, va_cal, idx)
        val = {}
        for m in C.METHODS:
            val[m] = {**C.metrics(y_va, va_cal[m]), **boot[m]}
            val[m]["n_exact_zero_or_one"] = int(((va_cal[m] <= 0) | (va_cal[m] >= 1)).sum())
        chosen = sel["selected"]
        bands = C.band_thresholds(y_tr, oof_cal[chosen])
        bands.update({"selected_method": chosen})
        bands["oof_table"] = C.band_table(y_tr, oof_cal[chosen], bands["t_watch"], bands["t_high"])
        bands["val_table"] = C.band_table(y_va, va_cal[chosen], bands["t_watch"], bands["t_high"])
        alt = C.band_thresholds(y_tr, oof_cal[chosen], **C.ALT_BAND_RULE)
        alt["oof_table"] = C.band_table(y_tr, oof_cal[chosen], alt["t_watch"], alt["t_high"])
        alt["val_table"] = C.band_table(y_va, va_cal[chosen], alt["t_watch"], alt["t_high"])
        bands["alt"] = alt
        cost = {str(r): C.cost_thresholds(y_tr, oof_cal[chosen], y_va, va_cal[chosen], r) for r in C.COST_RATIOS}
        res["models"][name] = {
            "precision_floor_diagnostic": C.precision_floor_diagnostic(y_tr, oof_cal[chosen]),
            "selection": sel, "selected_method": chosen, "val": val, "bands": bands, "cost": cost,
            "oof_metrics": {m: C.metrics(y_tr, oof_cal[m]) for m in C.METHODS},
            "tradeoff_oof": C.tradeoff_rows(y_tr, oof_cal[chosen]),
            "tradeoff_val": C.tradeoff_rows(y_va, va_cal[chosen]),
            "platt": {"a": cals["platt"].a, "b": cals["platt"].b},
            "isotonic_knots": int(len(cals["isotonic"].x)),
        }
        res["_arrays"][name] = {"oof": oof, "oof_cal": oof_cal, "raw_va": raw_va, "va_cal": va_cal, "cals": cals}
        print(f"{name}: selected {chosen}; bands watch>={bands['t_watch']:.4f} high>={bands['t_high']:.4f}")
    return res


def make_flags(res: dict, y_tr, y_va) -> list:
    flags = [f"Validation has only {int(np.sum(y_va))} bankrupt firms and ECE uses {C.ECE_BINS} equal-frequency bins "
             f"of about {len(y_va) // C.ECE_BINS} firms (about {int(np.sum(y_va)) / C.ECE_BINS:.0f} positives each), "
             "so ECE and the reliability points are noisy. Read them with the CIs.",
             f"Calibrators and bands were fitted on {FOLDS_TXT} out-of-fold scores of the train split "
             f"({int(np.sum(y_tr))} bankrupt). The saved models are refit on all of train, so they are slightly "
             "stronger than the fold models the calibrators saw. Direction of the bias is not verified.",
             "Cost ratios 5:1, 10:1 and 20:1 are assumptions, not estimates from any lender or data.",
             "Band rule wording: the approved text said 'lowest threshold' for Watch recall, which taken literally "
             "flags every firm. It was applied as the highest threshold that still reaches the recall floor."]
    outside = sum(1 for m in res["models"].values() for meth, v in m["val"].items()
                  if not v["ci"]["ece"][0] <= v["ece"] <= v["ci"]["ece"][1])
    total = sum(len(m["val"]) for m in res["models"].values())
    if outside:
        flags.append(f"ECE bootstrap intervals exclude the point estimate in {outside} of {total} model-method cells "
                     "(equal-frequency bins with ties and resampling make ECE biased). Treat ECE as descriptive and "
                     "rely on Brier and log loss.")
    for name, m in res["models"].items():
        b = m["bands"]
        for note in b["notes"]:
            flags.append(f"{name}: {note}.")
        hi_o = next(r for r in b["oof_table"] if r["band"] == "High")
        hi_v = next(r for r in b["val_table"] if r["band"] == "High")
        if hi_v["bankrupt_rate"] is not None and hi_o["bankrupt_rate"] is not None \
                and hi_v["bankrupt_rate"] < C.BAND_RULE["high_min_precision"]:
            flags.append(f"{name}: validation High-band precision ({hi_v['bankrupt_rate']:.3f}) is below the "
                         f"{C.BAND_RULE['high_min_precision']:.2f} target met on OOF train ({hi_o['bankrupt_rate']:.3f}).")
        wa_o = sum(r["share_of_all_bankrupt"] for r in b["oof_table"] if r["band"] != "Low")
        wa_v = sum(r["share_of_all_bankrupt"] for r in b["val_table"] if r["band"] != "Low")
        if wa_v < C.BAND_RULE["watch_min_recall"]:
            flags.append(f"{name}: Watch+High recall on validation ({wa_v:.3f}) is below the "
                         f"{C.BAND_RULE['watch_min_recall']:.2f} target met on OOF train ({wa_o:.3f}).")
        alt = b["alt"]
        alt_v = {r["band"]: r for r in alt["val_table"]}
        alt_o = {r["band"]: r for r in alt["oof_table"]}
        for note in alt["notes"]:
            flags.append(f"{name} (alternative rule): {note}.")
        if alt_v["High"]["bankrupt_rate"] is not None and alt_o["High"]["bankrupt_rate"] is not None                 and alt_v["High"]["bankrupt_rate"] < alt_o["High"]["bankrupt_rate"] - 0.05:
            flags.append(f"{name} (alternative rule): validation High-band precision ({alt_v['High']['bankrupt_rate']:.3f}) "
                         f"is more than 0.05 below OOF train ({alt_o['High']['bankrupt_rate']:.3f}).")
        iso = m["val"]["isotonic"]
        if iso["n_exact_zero_or_one"]:
            flags.append(f"{name}: isotonic gives exactly 0 or 1 to {iso['n_exact_zero_or_one']} validation firms "
                         f"(log loss clips at {C.LL_EPS}), which is overconfident and costs log loss.")
        if m["platt"]["a"] <= 0:
            flags.append(f"{name}: Platt slope is not positive, so the map is not monotone.")
    return flags


FOLDS_TXT = f"{C.FOLDS}-fold"


# ------------------------------------------------------------------------ figures
def plot_reliability(res, y_va, out_png) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    names = list(res["models"])
    fig, axes = plt.subplots(2, len(names), figsize=(4.4 * len(names), 8.4), facecolor=SURFACE, squeeze=False)
    for j, name in enumerate(names):
        for i, lim in enumerate((1.0, 0.25)):
            ax = axes[i, j]
            style_axes(ax)
            ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls=(0, (1, 3)))
            for m in C.METHODS:
                label, color, ls, marker = METHOD_STYLE[m]
                pts = C.reliability_points(y_va, res["_arrays"][name]["va_cal"][m])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, marker=marker, lw=1.8, ms=6,
                        label=label)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed bankruptcy rate")
            ax.set_title(f"{MODEL_STYLE[name][0]}" + ("" if i == 0 else " (zoom)"), loc="left", fontsize=11)
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left")
    fig.suptitle(f"Reliability on validation ({len(y_va)} firms, {int(np.sum(y_va))} bankrupt), "
                 f"{C.ECE_BINS} equal-frequency bins", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pr_bands(res, y_va, out_png) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2), facecolor=SURFACE)
    style_axes(ax)
    for name, m in res["models"].items():
        label, color, ls, marker = MODEL_STYLE[name]
        p = res["_arrays"][name]["va_cal"][m["selected_method"]]
        prec, rec, _ = precision_recall_curve(y_va, p)
        ax.plot(rec, prec, color=color, ls=ls, lw=1.8, drawstyle="steps-post",
                label=f"{label} ({m['selected_method']})")
        b = m["bands"]
        for t, mk in ((b["t_watch"], "X"), (b["t_high"], "P")):
            f = p >= t
            tp = int((f & (np.asarray(y_va) == 1)).sum())
            if f.sum():
                ax.plot(tp / np.sum(y_va), tp / f.sum(), marker=mk, color=color, ms=11, mec="black", mew=1.0, ls="")
    ax.plot([], [], marker="X", color="white", mec="black", ls="", ms=10, label="Watch cut (fixed on train OOF)")
    ax.plot([], [], marker="P", color="white", mec="black", ls="", ms=10, label="High cut (fixed on train OOF)")
    ax.axhline(np.mean(y_va), color=INK2, lw=1, ls=(0, (1, 3)))
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Validation precision and recall at the saved band cuts", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------- report
def fmt_ci(v, ci, nd=3):
    return f"{v:.{nd}f} ({ci[0]:.{nd}f} to {ci[1]:.{nd}f})"


def render_report(res: dict, y_tr, y_va, config: dict) -> str:
    L = []
    a = L.append
    names = list(res["models"])
    nm = {n: MODEL_STYLE[n][0] for n in names}
    a("# Phase 4: calibration, thresholds and risk bands")
    a("")
    a(f"Polish 5year, 1-year horizon. Train {len(y_tr):,} firms ({int(np.sum(y_tr))} bankrupt), validation "
      f"{len(y_va):,} ({int(np.sum(y_va))} bankrupt, no-skill PR AUC {np.mean(y_va):.3f}). The test split was not read. "
      "Seed 42. Every number below is generated by `python -m scripts.run_phase4`; raw values are in "
      "[phase4_calibration.json](phase4_calibration.json).")
    a("")
    a("## Method and the choices made")
    a("")
    a(f"- **Calibration data: {C.FOLDS}-fold out-of-fold (OOF) predictions on train.** Each firm is scored by a model that "
      f"never saw it, and all {int(np.sum(y_tr))} train positives are used. The alternative, a calibration split carved "
      f"from train, would leave about {int(round(np.sum(y_tr) * 0.3))} positives (at 30%), too few for isotonic regression, "
      "which needs many points per step. Cross-fitting costs one thing: the saved models are refit on all of train, "
      "so they differ slightly from the fold models the calibrators were fitted on.")
    a("- **Validation is never used to fit a calibrator or choose a threshold.** It only reports. "
      "The calibration method per model is also chosen on train OOF (below), not on validation.")
    a("- **Methods:** none, Platt (sigmoid on the logit of the score) and isotonic (monotone steps).")
    a("- **Method rule, set before results:** lowest cross-validated log loss on the OOF scores, ties within one "
      "standard error resolved toward the simpler method (none, Platt, isotonic).")
    a(f"- **ECE:** {C.ECE_BINS} equal-frequency bins (equal-width bins would be mostly empty at a 7% base rate). "
      f"Log loss clips probabilities to [{C.LL_EPS}, 1 - {C.LL_EPS}]. Bootstrap CIs: {config['bootstrap_n']} stratified "
      "resamples of validation, percentile 95%.")
    a("")
    a("## Calibration method chosen on train (cross-validated, OOF)")
    a("")
    a("| Model | " + " | ".join(f"{m} log loss (SE)" for m in C.METHODS) + " | Chosen |")
    a("|---|" + "---|" * (len(C.METHODS) + 1))
    for n in names:
        s = res["models"][n]["selection"]["table"]
        a(f"| {nm[n]} | " + " | ".join(f"{s[m]['log_loss_mean']:.4f} ({s[m]['log_loss_se']:.4f})" for m in C.METHODS)
          + f" | **{res['models'][n]['selected_method']}** |")
    a("")
    a("## Validation: before and after calibration")
    a("")
    a("Lower is better for Brier, log loss and ECE. PR AUC is higher-better; Platt cannot change ranking, "
      "so any PR AUC change comes from ties (isotonic).")
    a("")
    a("| Model | Method | Brier (95% CI) | Log loss | ECE | PR AUC |")
    a("|---|---|---|---|---|---|")
    for n in names:
        for m in C.METHODS:
            v = res["models"][n]["val"][m]
            star = " (chosen)" if m == res["models"][n]["selected_method"] else ""
            a(f"| {nm[n]} | {m}{star} | {fmt_ci(v['brier'], v['ci']['brier'], 4)} | "
              f"{fmt_ci(v['log_loss'], v['ci']['log_loss'], 4)} | {fmt_ci(v['ece'], v['ci']['ece'], 4)} | "
              f"{fmt_ci(v['pr_auc'], v['ci']['pr_auc'])} |")
    a("")
    a("Paired bootstrap, change from calibrating (method minus none), 95% CI:")
    a("")
    a("| Model | Method | Brier | Log loss | ECE | PR AUC |")
    a("|---|---|---|---|---|---|")
    for n in names:
        for m in ("platt", "isotonic"):
            v = res["models"][n]["val"][m]
            d = v["delta_vs_none"]
            cells = []
            for k, nd in (("brier", 4), ("log_loss", 4), ("ece", 4), ("pr_auc", 3)):
                point = v[k] - res["models"][n]["val"]["none"][k]
                lo, hi = d[k]["ci"]
                cells.append(f"{point:+.{nd}f} ({lo:+.{nd}f} to {hi:+.{nd}f})")
            a(f"| {nm[n]} | {m} | " + " | ".join(cells) + " |")
    a("")
    a("![Reliability diagrams](figures/phase4_reliability.png)")
    a("")
    a("Platt fit on OOF train: " + "; ".join(
        f"{nm[n]} slope {res['models'][n]['platt']['a']:.3f}, intercept {res['models'][n]['platt']['b']:.3f}" for n in names)
      + ". A slope below 1 means the raw model was overconfident. Isotonic knots: "
      + "; ".join(f"{nm[n]} {res['models'][n]['isotonic_knots']}" for n in names) + ".")
    a("")
    a("## Risk bands")
    a("")
    a("**Rule, stated before results.** On the calibrated probability of the chosen method, fitted on OOF train only: "
      f"High starts at the lowest threshold whose OOF precision is at least {C.BAND_RULE['high_min_precision']:.2f}. "
      f"Watch starts at the highest threshold whose OOF recall is still at least {C.BAND_RULE['watch_min_recall']:.2f}. "
      "Low is everything below. Nothing was tuned afterwards.")
    a("")
    a("**Result: the approved rule degenerates on this data.** Precision of "
      f"{C.BAND_RULE['high_min_precision']:.2f} is already reached at a recall above the "
      f"{C.BAND_RULE['watch_min_recall']:.2f} Watch target, so the Watch cut lands above the High cut and the "
      "Watch band is empty. High becomes a broad band. The tables below report this honestly. "
      "Nothing was changed to make it look better.")
    a("")
    a("| Model | Method | Watch cut (p >=) | High cut (p >=) |")
    a("|---|---|---|---|")
    for n in names:
        b = res["models"][n]["bands"]
        a(f"| {nm[n]} | {b['selected_method']} | {b['t_watch']:.4f} | {b['t_high']:.4f} |")
    a("")

    def rate(r):
        return "n/a" if r["bankrupt_rate"] is None else f"{r['bankrupt_rate']:.3f}"

    def band_block(b):
        a("| Band | OOF train firms | OOF bankrupt rate | OOF share of all bankrupt | Val firms | Val bankrupt | Val bankrupt rate | Val share of all bankrupt |")
        a("|---|---|---|---|---|---|---|---|")
        for ro, rv in zip(b["oof_table"], b["val_table"]):
            a(f"| {ro['band']} | {ro['firms']} ({ro['share_of_firms']:.1%}) | {rate(ro)} | {ro['share_of_all_bankrupt']:.1%} | "
              f"{rv['firms']} ({rv['share_of_firms']:.1%}) | {rv['bankrupt']} | {rate(rv)} | {rv['share_of_all_bankrupt']:.1%} |")
        a("")

    a("### Bands under the approved rule")
    a("")
    for n in names:
        b = res["models"][n]["bands"]
        a(f"**{nm[n]}** ({b['selected_method']})")
        a("")
        band_block(b)
        for note in b["notes"]:
            a(f"> Note: {note}.")
            a("")
    a("### What each precision floor implies (OOF train, chosen method)")
    a("")
    a("For the lowest threshold meeting each precision floor: the share of OOF recall and of firms flagged. "
      "This shows why the 0.25 floor is too low to separate High from Watch.")
    a("")
    a("| Model | " + " | ".join(f"Floor {f:.2f}: recall / firms flagged" for f in C.PRECISION_FLOORS) + " |")
    a("|---|" + "---|" * len(C.PRECISION_FLOORS))
    for n in names:
        cells = []
        for r in res["models"][n]["precision_floor_diagnostic"]:
            cells.append("not reached" if r["recall"] is None else f"{r['recall']:.3f} / {r['share_flagged']:.1%}")
        a(f"| {nm[n]} | " + " | ".join(cells) + " |")
    a("")
    a(f"### Alternative rule (post-hoc, NOT approved, needs your decision)")
    a("")
    a(f"Proposed only after seeing that the approved rule degenerates on train OOF. High: lowest threshold with OOF "
      f"precision at least {C.ALT_BAND_RULE['high_min_precision']:.2f}. Watch: unchanged (OOF recall at least "
      f"{C.ALT_BAND_RULE['watch_min_recall']:.2f}). The 0.50 floor is one round number chosen once; no other floor was "
      "scanned for the result and validation played no part in the choice. If you approve it, it replaces the rule above.")
    a("")
    a("| Model | Watch cut (p >=) | High cut (p >=) |")
    a("|---|---|---|")
    for n in names:
        b = res["models"][n]["bands"]["alt"]
        a(f"| {nm[n]} | {b['t_watch']:.4f} | {b['t_high']:.4f} |")
    a("")
    for n in names:
        b = res["models"][n]["bands"]["alt"]
        a(f"**{nm[n]}** ({res['models'][n]['selected_method']}, alternative rule)")
        a("")
        band_block(b)
        for note in b["notes"]:
            a(f"> Note: {note}.")
            a("")
    a("![Validation precision and recall at the band cuts](figures/phase4_pr_bands.png)")
    a("")
    a("### Precision-recall tradeoff (calibrated probability cutoffs, chosen method)")
    a("")
    a("Cutoffs are fixed in advance on a probability grid, not tuned. Full grid for every model in "
      "[phase4_pr_tradeoff.csv](phase4_pr_tradeoff.csv).")
    a("")
    for n in names:
        a(f"**{nm[n]}** (validation; OOF train in the CSV)")
        a("")
        a("| Cutoff | Firms flagged | Precision | Recall |")
        a("|---|---|---|---|")
        for r in res["models"][n]["tradeoff_val"]:
            if r["flagged"] == 0:
                continue
            a(f"| {r['threshold']:.3f} | {r['flagged']} ({r['share_flagged']:.1%}) | {r['precision']:.3f} | {r['recall']:.3f} |")
        a("")
    a("## Cost framing (assumptions)")
    a("")
    a("Cost = ratio x missed distress + 1 x false alarm. The ratios 5:1, 10:1 and 20:1 are **assumptions**, not "
      "estimates. The empirical cut minimises cost on OOF train; the theoretical cut 1 / (1 + ratio) is optimal only "
      "if the probabilities are perfectly calibrated. Both are scored on validation as cost per 1,000 firms; "
      "for reference, flagging nobody and flagging everybody cost the amounts in the last two columns.")
    a("")
    for n in names:
        a(f"**{nm[n]}** ({res['models'][n]['selected_method']})")
        a("")
        a("| Ratio | Theory cut | Theory: flagged / recall / precision / cost | Empirical cut | Empirical: flagged / recall / precision / cost | Flag nobody | Flag everybody |")
        a("|---|---|---|---|---|---|---|")
        for r, c in res["models"][n]["cost"].items():
            def cell(d):
                prec = "n/a" if d["precision"] is None else f"{d['precision']:.3f}"
                return f"{d['flagged']} / {d['recall']:.3f} / {prec} / {d['cost_per_1000']:.0f}"
            a(f"| {r}:1 | {c['t_theory']:.3f} | {cell(c['theory'])} | {c['t_empirical']:.4f} | {cell(c['empirical'])} | "
              f"{c['flag_nobody_cost_per_1000']:.0f} | {c['flag_everybody_cost_per_1000']:.0f} |")
        a("")
    a("## Model decision (for the user)")
    a("")
    a("Validation comparison with the chosen calibration per model:")
    a("")
    a("| Model | Method | PR AUC (95% CI) | Brier | Log loss | ECE | Watch+High recall | High precision |")
    a("|---|---|---|---|---|---|---|---|")
    for n in names:
        m = res["models"][n]
        v = m["val"][m["selected_method"]]
        vt = {r["band"]: r for r in m["bands"]["val_table"]}
        wh = vt["Watch"]["share_of_all_bankrupt"] + vt["High"]["share_of_all_bankrupt"]
        hp = "n/a" if vt["High"]["bankrupt_rate"] is None else f"{vt['High']['bankrupt_rate']:.3f}"
        a(f"| {nm[n]} | {m['selected_method']} | {fmt_ci(v['pr_auc'], v['ci']['pr_auc'])} | {v['brier']:.4f} | "
          f"{v['log_loss']:.4f} | {v['ece']:.4f} | {wh:.3f} | {hp} |")
    a("")
    a(build_recommendation(res))
    a("")
    a("## Flags")
    a("")
    for i, f in enumerate(res["flags"], 1):
        a(f"{i}. {f}")
    a("")
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def run(X_tr, y_tr, X_va, y_va, final_models: dict, names: list, reports=REPORTS, artifacts=ARTIFACTS,
        n_boot: int = N_BOOT, write_report: bool = True) -> dict:
    reports.mkdir(parents=True, exist_ok=True)
    res = compute(X_tr, y_tr, X_va, y_va, final_models, names, n_boot)
    res["flags"] = make_flags(res, y_tr, y_va)
    config = {"seed": SEED, "folds": C.FOLDS, "ece_bins": C.ECE_BINS, "log_loss_eps": C.LL_EPS,
              "methods": list(C.METHODS), "band_rule": C.BAND_RULE, "cost_ratios_assumed": list(C.COST_RATIOS),
              "alt_band_rule_post_hoc": C.ALT_BAND_RULE, "bootstrap_n": n_boot, "selection_rule": "lowest CV log loss on train OOF, simplest within 1 SE",
              "fit_data": "train OOF only; validation reports only", "test_split_read": False,
              "train_rows": int(len(y_tr)), "val_rows": int(len(y_va)),
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "numpy": np.__version__},
              "data_sha256": {k: sha256(PROCESSED / f"polish_5year_{k}.csv") for k in ("train", "val")}
              if (PROCESSED / "polish_5year_train.csv").exists() else {}}
    cals = {n: res["_arrays"][n]["cals"] for n in names}
    def strip(d):
        return {k: (strip(v) if isinstance(v, dict) else v) for k, v in d.items() if k not in ("oof_table", "val_table")}

    bands = {n: strip(res["models"][n]["bands"]) for n in names}
    C.save_artifacts(artifacts, cals, bands, config)
    if write_report:
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "figures").mkdir(exist_ok=True)
        plot_reliability(res, y_va, reports / "figures" / "phase4_reliability.png")
        plot_pr_bands(res, y_va, reports / "figures" / "phase4_pr_bands.png")
        rows = []
        for n in names:
            for split in ("oof", "val"):
                for r in res["models"][n][f"tradeoff_{'oof' if split == 'oof' else 'val'}"]:
                    rows.append({"model": n, "split": "train_oof" if split == "oof" else "val", **r})
        pd.DataFrame(rows).to_csv(reports / "phase4_pr_tradeoff.csv", index=False)
        (reports / "phase4_calibration.md").write_text(render_report(res, y_tr, y_va, config), encoding="utf-8")
    public = {"config": config, "flags": res["flags"], "models": res["models"]}
    (reports / "phase4_calibration.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    return res


def main() -> None:
    np.random.seed(SEED)
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    names = list(FACTORIES)
    final = {n: joblib.load(MODELS / f"{n}.joblib") for n in names}
    run(X_tr, y_tr, X_va, y_va, final, names)


if __name__ == "__main__":
    main()
