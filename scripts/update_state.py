"""Regenerate docs/STATE.md and data/processed/summary.json from code.

Run at the end of every phase:
    python -m scripts.update_state --phase "Phase 4: calibration" [--commit <hash>]

Reads the train and validation splits only. The test split is never opened,
not even to count rows: its size comes from the Phase 1 data report.
All numbers in the outputs are computed or copied from generated reports.
"""

import argparse
import json
import re
import subprocess

import numpy as np

from src.polish_features import FEATURE_NAMES, TARGET
from src.preprocess import PROCESSED, ROOT
from src.train import load_split

DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
SUMMARY = PROCESSED / "summary.json"
CLAUDE_MD = ROOT / "CLAUDE.md"

# Hand-written statements that contain no numbers. Numeric flags come from the
# Phase 4 results file.
STATIC_FLAGS = [
    "Validation set is small, so metrics carry wide intervals. Use the CIs.",
    "Random split only. No company id or year exists, so a time-based split is impossible.",
    "Attr21 (sales_growth) needs last year's sales, and it is the second strongest feature.",
]
LOCK = ROOT / "artifacts" / "phase5.lock"
RESULTS5 = REPORTS / "phase5_test_results.json"


def test_split_status() -> str:
    """Read from the lock file, never from the test split itself."""
    if LOCK.exists() and json.loads(LOCK.read_text())["status"] == "complete":
        return "READ ONCE in Phase 5, locked, results in reports/phase5_test_results.md"
    return "LOCKED, unread"


def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def test_split_from_report() -> dict:
    """Test split size as recorded by Phase 1 (parsed from the report, file not read)."""
    text = (REPORTS / "data_report.md").read_text(encoding="utf-8")
    m = re.search(r"\| Test \(\d+%\) \| ([\d,]+) \| ([\d,]+) \|", text)
    if not m:
        return {"rows": None, "bankrupt": None, "source": "reports/data_report.md (not found)"}
    rows, pos = (int(g.replace(",", "")) for g in m.groups())
    return {"rows": rows, "bankrupt": pos, "bankrupt_rate": pos / rows,
            "source": "reports/data_report.md (size parsed from the report)"}


def build_summary() -> dict:
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    splits = {}
    for name, y in (("train", y_tr), ("val", y_va)):
        splits[name] = {"rows": int(len(y)), "bankrupt": int(y.sum()), "bankrupt_rate": float(y.mean()),
                        "path": f"data/processed/polish_5year_{name}.csv"}
    splits["test"] = {**test_split_from_report(), "path": "data/processed/polish_5year_test.csv",
                      "status": test_split_status()}
    stats = {}
    for c in FEATURE_NAMES:
        v = X_tr[c]
        q = v.quantile([0.01, 0.5, 0.99])
        stats[c] = {"missing_share": float(v.isna().mean()), "mean": float(v.mean()), "std": float(v.std()),
                    "min": float(v.min()), "p01": float(q[0.01]), "median": float(q[0.5]),
                    "p99": float(q[0.99]), "max": float(v.max())}
    return {"dataset": "Polish companies bankruptcy, 5year.arff (UCI 365), 1-year horizon",
            "seed": 42, "target": TARGET, "n_features": len(FEATURE_NAMES), "features": FEATURE_NAMES,
            "splits": splits, "feature_stats_train": stats}


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def results_table() -> list:
    path = REPORTS / "models_val.json"
    if not path.exists():
        return ["(models_val.json missing)"]
    res = json.loads(path.read_text())
    lines = ["| Model | PR AUC (95% CI) | ROC AUC | Brier | Recall at 5% FPR |", "|---|---|---|---|---|"]
    for name, r in res.items():
        v, ci = r["val"], r["val_ci"]["pr_auc_ci"]
        lines.append(f"| {name} | {v['pr_auc']:.3f} ({ci[0]:.3f} to {ci[1]:.3f}) | {v['roc_auc']:.3f} | "
                     f"{v['brier']:.4f} | {v['recall_at_fpr5']:.3f} |")
    return lines


def calibration_table() -> tuple:
    path = REPORTS / "phase4_calibration.json"
    if not path.exists():
        return ["(Phase 4 not run yet)"], []
    res = json.loads(path.read_text())
    lines = [f"Headline model: {res['config'].get('headline_model')} (user decision). Other models are comparisons, not deployed.",
             "", "| Model | Calibration chosen | Brier | Log loss | ECE | PR AUC | Bands | Watch cut | High cut |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in res["models"].items():
        sel = m["selected_method"]
        v = m["val"][sel]
        b = m["bands"]
        lines.append(f"| {name} | {sel} | {v['brier']:.4f} | {v['log_loss']:.4f} | {v['ece']:.4f} | "
                     f"{v['pr_auc']:.3f} | {'two' if b['t_watch'] >= b['t_high'] else 'three'} | "
                     f"{b['t_watch']:.4f} | {b['t_high']:.4f} |")
    return lines, res.get("flags", [])


def static_flags() -> list:
    done = test_split_status().startswith("READ ONCE")
    last = ("The test split was read once, in Phase 5, and is locked. Nothing may be tuned on it. Any further model change needs new data."
            if done else "Test split is unread. It stays locked until the user approves a final model.")
    return STATIC_FLAGS + [last]


def phase5_section() -> tuple:
    """Headline test results and the Flags list, copied from the Phase 5 outputs. Empty before Phase 5."""
    if not RESULTS5.exists():
        return [], []
    res = json.loads(RESULTS5.read_text())
    h = res["models"]["lightgbm"]
    m, ci, v = h["metrics"], h["ci"], res["verdict"]
    st = h["band_stats"]
    rows = {r["band"]: r for r in h["band_table"]}
    lines = [f"One-shot result on the test split ({res['rows']} firms, {res['bankrupt']} bankrupt), frozen LightGBM + Platt, "
             f"cuts {h['t_watch']:.4f} and {h['t_high']:.4f}. Verdict on the pre-stated rule: "
             f"{'holds up' if v['holds_up'] else 'does not fully hold up'}. Full report: reports/phase5_test_results.md.", "",
             "| Metric | Test (95% CI) |", "|---|---|",
             f"| PR AUC (baseline {res['prevalence']:.3f}) | {m['pr_auc']:.3f} ({ci['pr_auc'][0]:.3f} to {ci['pr_auc'][1]:.3f}) |",
             f"| ROC AUC | {m['roc_auc']:.3f} ({ci['roc_auc'][0]:.3f} to {ci['roc_auc'][1]:.3f}) |",
             f"| Brier | {m['brier']:.4f} ({ci['brier'][0]:.4f} to {ci['brier'][1]:.4f}) |",
             f"| Log loss | {m['log_loss']:.4f} ({ci['log_loss'][0]:.4f} to {ci['log_loss'][1]:.4f}) |",
             f"| ECE | {m['ece']:.4f} ({ci['ece'][0]:.4f} to {ci['ece'][1]:.4f}) |",
             f"| High band (firms, bankrupt, precision) | {rows['High']['firms']}, {rows['High']['bankrupt']}, {st['high']['precision']:.3f} ({ci['high_precision'][0]:.3f} to {ci['high_precision'][1]:.3f}) |",
             f"| High alone recall | {st['high']['recall']:.3f} ({ci['high_recall'][0]:.3f} to {ci['high_recall'][1]:.3f}) |",
             f"| Watch+High recall | {st['watch_plus_high']['recall']:.3f} ({ci['flag_recall'][0]:.3f} to {ci['flag_recall'][1]:.3f}) |",
             f"| Watch band (firms, bankrupt) | {rows['Watch']['firms']}, {rows['Watch']['bankrupt']} |"]
    text = (REPORTS / "phase5_test_results.md").read_text(encoding="utf-8")
    block = text.split("## Flags")[1].split("## Provenance")[0]
    flags = [re.sub(r"^\d+\.\s*", "", ln) for ln in block.splitlines() if re.match(r"^\d+\.", ln)]
    return lines, [f"Phase 5: {f}" for f in flags]


def phase6_section() -> tuple:
    """SHAP summary and its flags, copied from reports/phase6_explanations.json. Empty before Phase 6."""
    path = REPORTS / "phase6_explanations.json"
    if not path.exists():
        return [], []
    res = json.loads(path.read_text())
    imp, dic = res["importance"], res["dictionary"]
    top = sorted(range(len(dic)), key=lambda j: -imp["mean_abs_val"][j])[:5]
    lines = [f"SHAP for the frozen LightGBM + Platt model on validation, in {res['scale'].replace('_', ' ')} "
             f"(Platt slope check passed on train and validation: {res['check']['train']['ok'] and res['check']['val']['ok']}). "
             "Categories are assigned by this project (UCI gives none). Full report: reports/phase6_explanations.md.", "",
             "| Top feature | Category | Mean abs SHAP (95% CI) |", "|---|---|---|"]
    for j in top:
        ci = imp["ci_val"][j]
        lines.append(f"| {dic[j]['label']} | {dic[j]['category']} | {imp['mean_abs_val'][j]:.3f} ({ci[0]:.3f} to {ci[1]:.3f}) |")
    lines += ["", "| Category | Share of total importance (validation) |", "|---|---|"]
    for c, v in sorted(imp["category"].items(), key=lambda kv: -kv[1]["share_val"]):
        lines.append(f"| {c} | {v['share_val']:.1%} |")
    m = res["attr21_missingness"]
    lines += ["", "| Attr21 | Split | Firms | Bankrupt | Rate |", "|---|---|---|---|---|"]
    for s in ("train", "val"):
        for tag in ("missing", "present"):
            r = m[s][tag]
            lines.append(f"| {tag} | {s} | {r['firms']} | {r['bankrupt']} | {r['rate']:.3f} |")
    return lines, [f"Phase 6: {f}" for f in res["flags"]]


def phase7_section() -> tuple:
    """Attr21 missingness summary and its flags, copied from reports/phase7_attr21.json. Empty before Phase 7."""
    path = REPORTS / "phase7_attr21.json"
    if not path.exists():
        return [], []
    res = json.loads(path.read_text())
    c, perf, w = res["counts"], res["performance"], res["what_if_val"]
    lines = ["Attr21 (sales growth) missingness, train and validation only, frozen model unchanged. Full report: reports/phase7_attr21.md.", "",
             "| Split | Attr21 | Firms | Bankrupt | Rate |", "|---|---|---|---|---|"]
    for s in ("train", "val"):
        for tag in ("missing", "present"):
            r = c[s][tag]
            lines.append(f"| {s} | {tag} | {r['firms']} | {r['bankrupt']} | {r['rate']:.3f} |")
    lines += ["", "| Scores | Subset | PR AUC | High precision |", "|---|---|---|---|"]
    for name, key in (("validation", "val"), ("train out-of-fold", "train_oof")):
        for tag in ("all", "present"):
            b = perf[key][tag]
            lines.append(f"| {name} | Attr21 {tag} | {b['pr_auc']:.3f} | {b['bands']['high']['precision']:.3f} |")
    wb, wf = w["present_then_blanked"], w["missing_then_filled_with_train_median"]
    lines += ["", f"What-if on the frozen model (validation): blanking Attr21 moves {wb['band_counts_before']['Low'] + wb['band_counts_before']['Watch']} "
              f"Low or Watch firms to High-heavy bands ({wb['band_counts_after']['High']} of {wb['firms']} firms High after blanking, "
              f"{wb['band_counts_before']['High']} before)."]
    return lines, [f"Phase 7: {f}" for f in res["flags"]]


def phase8_section() -> tuple:
    """App summary and its flags, copied from reports/phase8_app_numbers.json. Empty before Phase 8."""
    path = REPORTS / "phase8_app_numbers.json"
    if not path.exists():
        return [], []
    res = json.loads(path.read_text())
    n = res["numbers"]
    lines = ["Streamlit app (app/streamlit_app.py) scoring firms with the frozen LightGBM + Platt model. Policy: Attr21 blank is not scored, "
             f"more than {res['blank_limit']} other blank attributes is not scored. Numbers below are on firms with Attr21 present. "
             "Details: reports/phase8_app_prep.md.", "",
             "| Scores | PR AUC | High precision | High recall |", "|---|---|---|---|"]
    for name, key in (("validation", "val"), ("train out-of-fold", "train_oof")):
        b = n[key]["present"]
        lines.append(f"| {name} | {b['pr_auc']:.3f} | {b['high']['precision']:.3f} | {b['high']['recall']:.3f} |")
    return lines, [f"Phase 8: {f}" for f in res["flags"]]


def render_state(summary: dict, phase: str, commit: str) -> str:
    sp = summary["splits"]
    cal_lines, cal_flags = calibration_table()
    p5_lines, p5_flags = phase5_section()
    p6_lines, p6_flags = phase6_section()
    p7_lines, p7_flags = phase7_section()
    p8_lines, p8_flags = phase8_section()
    out = [f"# Project state (generated by scripts/update_state.py, do not edit)", "",
           f"Phase: {phase}. Last phase commit: {commit}.", "",
           "## Results, validation only", "", *results_table(), "", "## Calibration and bands (Phase 4)", "",
           *cal_lines, "",
           *(["## Final test evaluation (Phase 5)", "", *p5_lines, ""] if p5_lines else []),
           *(["## Explanations (Phase 6)", "", *p6_lines, ""] if p6_lines else []),
           *(["## Attr21 missingness (Phase 7)", "", *p7_lines, ""] if p7_lines else []),
           *(["## App (Phase 8)", "", *p8_lines, ""] if p8_lines else []),
           "## Splits (seed 42, stratified 70/15/15)", "",
           "| Split | Rows | Bankrupt | Rate | Path |", "|---|---|---|---|---|"]
    for name, s in sp.items():
        out.append(f"| {name} | {s['rows']} | {s['bankrupt']} | {fmt(s.get('bankrupt_rate'), 4)} | `{s['path']}`"
                   f"{' (' + s['status'] + ')' if 'status' in s else ''} |")
    out += ["", f"Features: {summary['n_features']} ratios, names in `data/processed/summary.json`.", "",
            "## Artifacts", ""]
    for rel in ["models/logistic_regression.joblib", "models/lightgbm.joblib", "models/neural_net.joblib",
                "models/val_predictions.csv", "artifacts/calibration/config.json",
                "artifacts/calibration/bands.json", "artifacts/calibration/calibrators.json",
                "reports/phase4_calibration.json", "reports/phase4_calibration.md",
                "artifacts/phase5_protocol.json", "artifacts/phase5.lock", "reports/phase5_test_results.json",
                "reports/phase5_test_results.md", "artifacts/explain/feature_dictionary.json",
                "artifacts/explain/global_importance.json", "artifacts/explain/config.json",
                "reports/phase6_explanations.json", "reports/phase6_explanations.md",
                "reports/phase7_attr21.json", "reports/phase7_attr21.md", "app/streamlit_app.py", "app/demo_firms.json",
                "app/requirements.txt", "reports/phase8_app_numbers.json", "reports/phase8_app_prep.md"]:
        p = ROOT / rel
        out.append(f"- `{rel}`: {'present' if p.exists() else 'MISSING'}"
                   f"{' (gitignored, rebuild with python -m src.train)' if rel.startswith('models/') else ''}")
    out += ["", "## Open flags", "", *[f"- {f}" for f in static_flags() + cal_flags + p5_flags + p6_flags + p7_flags + p8_flags], ""]
    return "\n".join(out)


README = ROOT / "README.md"
MARK = ("<!-- RESULTS:START (generated by scripts/update_state.py) -->", "<!-- RESULTS:END -->")


def update_readme(summary: dict) -> None:
    """Rewrite the generated results and flags block in README.md, so no figure there is typed by hand."""
    if not README.exists():
        return
    sp = summary["splits"]
    cal_lines, cal_flags = calibration_table()
    p5_lines, p5_flags = phase5_section()
    p6_lines, p6_flags = phase6_section()
    p7_lines, p7_flags = phase7_section()
    p8_lines, p8_flags = phase8_section()
    block = [MARK[0], "",
             f"Validation split, Polish 5year: {sp['val']['rows']} firms, {sp['val']['bankrupt']} went bankrupt within a "
             f"year (no-skill PR AUC {sp['val']['bankrupt_rate']:.3f}). Test split: {sp['test']['status']}.", "",
             *results_table(), "", "Calibration and risk bands (fitted on train out-of-fold scores, reported on validation):",
             "", *cal_lines, "",
             *(["Final one-shot evaluation on the test split (Phase 5):", "", *p5_lines, ""] if p5_lines else []),
             *(["What drives the model, SHAP on train and validation (Phase 6):", "", *p6_lines, ""] if p6_lines else []),
             *(["Attr21 missingness (Phase 7):", "", *p7_lines, ""] if p7_lines else []),
             *(["App (Phase 8):", "", *p8_lines, ""] if p8_lines else []),
             "Details, reliability diagrams, band tables, cost framing and the model decision: "
             "[reports/phase4_calibration.md](reports/phase4_calibration.md). Earlier phases: "
             "[reports/models_val.md](reports/models_val.md).", "",
             "### Flags", "", *[f"{i}. {f}" for i, f in enumerate(static_flags() + cal_flags + p5_flags + p6_flags + p7_flags + p8_flags, 1)], "", MARK[1]]
    text = README.read_text(encoding="utf-8")
    if MARK[0] in text:
        text = re.sub(re.escape(MARK[0]) + r".*?" + re.escape(MARK[1]), lambda m: "\n".join(block), text, flags=re.S)
    README.write_text(text, encoding="utf-8")


def update_claude_md(phase: str, commit: str) -> None:
    if not CLAUDE_MD.exists():
        return
    text = CLAUDE_MD.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^Current phase:.*$", f"Current phase: {phase}", text)
    text = re.sub(r"(?m)^Last phase commit:.*$", f"Last phase commit: {commit}", text)
    CLAUDE_MD.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--commit", default=None, help="defaults to HEAD")
    args = ap.parse_args()
    commit = args.commit or git_head()
    summary = build_summary()
    SUMMARY.write_text(json.dumps(summary, indent=1))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "STATE.md").write_text(render_state(summary, args.phase, commit), encoding="utf-8")
    update_claude_md(args.phase, commit)
    update_readme(summary)
    print(f"wrote {SUMMARY.relative_to(ROOT)}, docs/STATE.md; phase={args.phase!r} commit={commit}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
