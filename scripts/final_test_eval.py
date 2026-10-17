"""Phase 5: one-shot final evaluation on the locked test split.

    python -m scripts.final_test_eval --freeze     write artifacts/phase5_protocol.json (hashes), once
    python -m scripts.final_test_eval --dry-run    same code path on the VALIDATION split, temp output
    python -m scripts.final_test_eval              the real run, reads the test split exactly once
    python -m scripts.final_test_eval --render-only   rebuild the markdown from the saved results

Protocol (docs/decisions.md, "Phase 5 protocol"): frozen artifacts only, nothing is refit, recalibrated or
re-cut. The run refuses if the results file or the lock file exists. The lock is written BEFORE the test
split is read, so a crash after the read cannot be followed by a second read. Only this file reads the
test split (tests/test_no_test_access.py scans for that). All randomness uses SEED = 42.
"""

import argparse
import hashlib
import importlib.metadata as md
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import calibrate as C
from src.plotting import GRID, INK2, MODEL_STYLE, SURFACE, apply_style, style_axes
from src.polish_features import FEATURE_NAMES, TARGET
from src.preprocess import PROCESSED, ROOT

SEED = 42
N_BOOT = 1000
FROZEN_COMMIT = "fdef552"
HEADLINE = "lightgbm"
REFERENCE = ("logistic_regression", "neural_net")
NL = "\n"
DECISIONS_MARK = "<!-- PHASE5_HASHES -->"

PROTOCOL = ROOT / "artifacts" / "phase5_protocol.json"
LOCK = ROOT / "artifacts" / "phase5.lock"
RESULTS = ROOT / "reports" / "phase5_test_results.json"
REPORT_MD = ROOT / "reports" / "phase5_test_results.md"
FIGURE = ROOT / "reports" / "figures" / "phase5_test_reliability.png"
TEST_FILE = PROCESSED / "polish_5year_test.csv"

FROZEN_FILES = {
    "models": ["models/lightgbm.joblib", "models/logistic_regression.joblib", "models/neural_net.joblib"],
    "calibration": ["artifacts/calibration/calibrators.json", "artifacts/calibration/bands.json",
                    "artifacts/calibration/config.json"],
    "comparison_sources": ["reports/phase4_calibration.json", "reports/models_val.json"],
}
METRICS_STATED = ["pr_auc", "roc_auc", "prevalence_pr_auc", "brier", "log_loss", "ece",
                  "band_counts", "recall_precision_watch_plus_high", "recall_precision_high", "confusion_at_high_cut"]
VERDICT_RULE = [
    "a. the test PR AUC point estimate lies inside the validation 95% CI of the headline model",
    "b. the lower bound of the test PR AUC 95% CI is above the prevalence baseline",
    "c. the 95% CI of High-band precision on test contains the 0.60 precision floor",
]


class Paths:
    """Output and input locations. Tests inject temp locations."""

    def __init__(self, root=ROOT, protocol=PROTOCOL, lock=LOCK, results=RESULTS, report_md=REPORT_MD,
                 figure=FIGURE, test_file=TEST_FILE):
        self.root, self.protocol, self.lock = Path(root), Path(protocol), Path(lock)
        self.results, self.report_md, self.figure = Path(results), Path(report_md), Path(figure)
        self.test_file = Path(test_file)


# ------------------------------------------------------------------------ helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def git(*args) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def tree_clean() -> bool:
    return git("status", "--porcelain") == ""


def versions() -> dict:
    v = {"python": platform.python_version()}
    for pkg in ("scikit-learn", "numpy", "pandas", "lightgbm", "torch", "joblib"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = "not installed"
    return v


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------- freeze
def freeze(paths: Paths = None) -> dict:
    """Record SHA256 of every frozen artifact. Refuses to overwrite an existing protocol."""
    paths = paths or Paths()
    if paths.protocol.exists():
        raise RuntimeError(f"{paths.protocol} already exists; the protocol is written once")
    files = {}
    for group, rels in FROZEN_FILES.items():
        for rel in rels:
            data = (paths.root / rel).read_bytes()
            entry = {"group": group, "sha256": sha256_bytes(data), "bytes": len(data)}
            if group == "calibration":
                blob = subprocess.run(["git", "show", f"{FROZEN_COMMIT}:{rel}"], cwd=ROOT, capture_output=True).stdout
                entry["matches_commit_blob"] = blob.replace(b"\r\n", b"\n") == data.replace(b"\r\n", b"\n")
            else:
                entry["in_git"] = False if group == "models" else True
            files[rel] = entry
    bands = json.loads((paths.root / "artifacts/calibration/bands.json").read_text())[HEADLINE]
    phase4 = json.loads((paths.root / "reports/phase4_calibration.json").read_text())
    proto = {
        "frozen_commit": FROZEN_COMMIT, "git_head_at_freeze": git("rev-parse", "--short", "HEAD"),
        "seed": SEED, "bootstrap_n": N_BOOT, "ece_bins": C.ECE_BINS, "headline_model": HEADLINE,
        "reference_models": list(REFERENCE),
        "headline_cuts": {"method": bands["selected_method"], "t_watch": bands["t_watch"], "t_high": bands["t_high"]},
        "band_rule": C.BAND_RULE, "metrics_stated_in_advance": METRICS_STATED, "verdict_rule": VERDICT_RULE,
        "validation_pr_auc_ci_headline": phase4["models"][HEADLINE]["val"][bands["selected_method"]]["ci"]["pr_auc"],
        "versions": versions(), "files": files,
    }
    paths.protocol.parent.mkdir(parents=True, exist_ok=True)
    paths.protocol.write_text(json.dumps(proto, indent=1), encoding="utf-8")
    return proto


def hash_table_markdown(proto: dict) -> str:
    rows = ["| File | SHA256 | Note |", "|---|---|---|"]
    for rel, e in proto["files"].items():
        note = ("gitignored, hash pins the exact file" if e["group"] == "models" else
                f"equals the {proto['frozen_commit']} blob: {e['matches_commit_blob']}" if e["group"] == "calibration"
                else "comparison source, committed")
        rows.append(f"| `{rel}` | `{e['sha256']}` | {note} |")
    cuts = proto["headline_cuts"]
    return NL.join([f"Frozen commit: `{proto['frozen_commit']}`. Git HEAD when frozen: `{proto['git_head_at_freeze']}`. "
                    f"Headline cuts read from bands.json: Watch {cuts['t_watch']:.4f}, High {cuts['t_high']:.4f}, "
                    f"calibration {cuts['method']}.", "", *rows])


def fill_decisions(proto: dict, decisions_path: Path) -> None:
    text = decisions_path.read_text(encoding="utf-8")
    if DECISIONS_MARK not in text:
        raise RuntimeError("decisions.md has no hash placeholder")
    decisions_path.write_text(text.replace(DECISIONS_MARK, hash_table_markdown(proto)), encoding="utf-8")


# ----------------------------------------------------------------------- verification
def load_protocol(paths: Paths) -> dict:
    if not paths.protocol.exists():
        raise RuntimeError("no protocol file: run --freeze first and commit it")
    return json.loads(paths.protocol.read_text(encoding="utf-8"))


def verify_files(proto: dict, paths: Paths) -> None:
    """Abort on any hash mismatch. Runs before anything is unpickled or the test split is read."""
    bad = []
    for rel, e in proto["files"].items():
        p = paths.root / rel
        if not p.exists() or sha256_file(p) != e["sha256"]:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"hash mismatch, aborting before any test data is read: {bad}")


def refuse_if_used(paths: Paths) -> None:
    if paths.results.exists():
        raise RuntimeError(f"{paths.results} exists: the test split was already evaluated. Refusing to run again.")
    if paths.lock.exists():
        raise RuntimeError(f"{paths.lock} exists: a previous attempt started reading the test split. "
                           "Refusing to run again. Inspect it manually.")


# ---------------------------------------------------------------------------- scoring
def band_stats(y, p, t_watch: float, t_high: float) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pos = int(y.sum())

    def one(mask):
        n = int(mask.sum())
        tp = int((mask & (y == 1)).sum())
        return {"flagged": n, "tp": tp, "fp": n - tp, "fn": pos - tp, "tn": int(len(y) - n - (pos - tp)),
                "precision": tp / n if n else None, "recall": tp / pos if pos else None}

    return {"high": one(p >= t_high), "watch_plus_high": one(p >= t_watch)}


def bootstrap_all(y, p, t_watch: float, t_high: float, indices: list, alpha: float = 0.05) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    keys = ("pr_auc", "roc_auc", "brier", "log_loss", "ece", "high_precision", "high_recall",
            "flag_precision", "flag_recall")
    draws = {k: [] for k in keys}
    for idx in indices:
        yy, pp = y[idx], p[idx]
        m = C.metrics(yy, pp)
        for k in ("pr_auc", "brier", "log_loss", "ece"):
            draws[k].append(m[k])
        draws["roc_auc"].append(float(roc_auc_score(yy, pp)))
        pos = max(int(yy.sum()), 1)
        for tag, t in (("high", t_high), ("flag", t_watch)):
            f = pp >= t
            tp = int((f & (yy == 1)).sum())
            draws[f"{tag}_precision"].append(tp / f.sum() if f.sum() else np.nan)
            draws[f"{tag}_recall"].append(tp / pos)
    q = [alpha / 2 * 100, (1 - alpha / 2) * 100]
    return {"ci": {k: np.nanpercentile(draws[k], q).tolist() for k in keys},
            "undefined_draws": {k: int(np.isnan(draws[k]).sum()) for k in keys}}


def score_models(X, y, proto: dict, paths: Paths, n_boot: int) -> dict:
    """Score the frozen models with the saved calibrators and cuts. Nothing is fitted here."""
    art = C.load_artifacts(paths.root / "artifacts" / "calibration")
    y = np.asarray(y)
    prevalence = float(y.mean())
    indices = C.bootstrap_indices(y, n_boot, SEED)
    out = {}
    for name in (HEADLINE, *REFERENCE):
        model = joblib.load(paths.root / "models" / f"{name}.joblib")
        raw = model.predict_proba(X)[:, 1]
        p, _ = C.risk_band(name, raw, art)
        b = art["bands"][name]
        m = C.metrics(y, p)
        m["roc_auc"] = float(roc_auc_score(y, p))
        m["prevalence_pr_auc"] = prevalence
        out[name] = {
            "role": "headline" if name == HEADLINE else "reference only, not eligible to replace the headline",
            "method": b["selected_method"], "t_watch": b["t_watch"], "t_high": b["t_high"],
            "metrics": m, **bootstrap_all(y, p, b["t_watch"], b["t_high"], indices),
            "band_table": C.band_table(y, p, b["t_watch"], b["t_high"]),
            "band_stats": band_stats(y, p, b["t_watch"], b["t_high"]),
            "reliability": C.reliability_points(y, p),
        }
    return out


def check_reproduces_validation(proto: dict, paths: Paths) -> None:
    """Before the test split is touched: frozen artifacts must reproduce the Phase 4 validation numbers."""
    X_va, y_va = read_split("val", paths)[:2]
    phase4 = json.loads((paths.root / "reports" / "phase4_calibration.json").read_text())["models"][HEADLINE]
    got = score_models(X_va, y_va, proto, paths, n_boot=2)[HEADLINE]
    sel = phase4["selected_method"]
    want = phase4["val"][sel]
    for k in ("pr_auc", "brier", "log_loss", "ece"):
        if abs(got["metrics"][k] - want[k]) > 1e-9:
            raise RuntimeError(f"frozen artifacts do not reproduce Phase 4 validation {k}: "
                               f"{got['metrics'][k]} vs {want[k]}. Aborting before the test split is read.")
    counts = [r["firms"] for r in got["band_table"]]
    if counts != [r["firms"] for r in phase4["bands"]["val_table"]]:
        raise RuntimeError("frozen artifacts do not reproduce the Phase 4 validation band counts")


def read_split(split: str, paths: Paths):
    """Read a split's bytes once, hash them, parse from the same bytes."""
    path = paths.test_file if split == "test" else PROCESSED / f"polish_5year_{split}.csv"
    raw = Path(path).read_bytes()
    df = pd.read_csv(io.BytesIO(raw))
    return df[FEATURE_NAMES], df[TARGET], sha256_bytes(raw)


# ----------------------------------------------------------------------------- verdict
def compute_verdict(res: dict, proto: dict) -> dict:
    h = res["models"][HEADLINE]
    lo_val, hi_val = proto["validation_pr_auc_ci_headline"]
    pr = h["metrics"]["pr_auc"]
    hp = h["ci"]["high_precision"]
    floor = proto["band_rule"]["high_min_precision"]
    crit = [
        {"id": "a", "rule": VERDICT_RULE[0], "pass": bool(lo_val <= pr <= hi_val),
         "detail": f"test {pr:.3f}, validation CI {lo_val:.3f} to {hi_val:.3f}"},
        {"id": "b", "rule": VERDICT_RULE[1], "pass": bool(h["ci"]["pr_auc"][0] > res["prevalence"]),
         "detail": f"lower bound {h['ci']['pr_auc'][0]:.3f}, baseline {res['prevalence']:.3f}"},
        {"id": "c", "rule": VERDICT_RULE[2], "pass": bool(not np.isnan(hp[0]) and hp[0] <= floor <= hp[1]),
         "detail": f"High precision CI {hp[0]:.3f} to {hp[1]:.3f}, floor {floor:.2f}"},
    ]
    return {"criteria": crit, "holds_up": all(c["pass"] for c in crit)}


# ------------------------------------------------------------------------------ run
def run(paths: Paths = None, split: str = "test", dry: bool = False, n_boot: int = N_BOOT,
        render: bool = True) -> dict:
    paths = paths or Paths()
    proto = load_protocol(paths)
    verify_files(proto, paths)
    refuse_if_used(paths)
    check_reproduces_validation(proto, paths)  # validation only, still before the lock and the test read
    if not dry:
        paths.lock.parent.mkdir(parents=True, exist_ok=True)
        paths.lock.write_text(json.dumps({"status": "started", "started_utc": utc_now(),
                                          "git_head": git("rev-parse", "--short", "HEAD"),
                                          "protocol_sha256": sha256_file(paths.protocol)}, indent=1), encoding="utf-8")
    X, y, data_sha = read_split(split, paths)
    res = {"split": split, "dry_run": dry, "rows": int(len(y)), "bankrupt": int(np.sum(y)),
           "prevalence": float(np.mean(y)), "data_sha256": data_sha, "seed": SEED, "bootstrap_n": n_boot,
           "git_head": git("rev-parse", "--short", "HEAD"), "versions": versions(),
           "protocol_sha256": sha256_file(paths.protocol), "models": score_models(X, y, proto, paths, n_boot)}
    res["verdict"] = compute_verdict(res, proto)
    paths.results.parent.mkdir(parents=True, exist_ok=True)
    paths.results.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not dry:
        lock = json.loads(paths.lock.read_text())
        lock.update({"status": "complete", "finished_utc": utc_now(), "test_sha256": data_sha,
                     "results_sha256": sha256_file(paths.results)})
        paths.lock.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    if render:
        render_outputs(res, proto, paths)
    return res


# --------------------------------------------------------------------------- rendering
def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def with_ci(v, ci, nd=3):
    return f"{f3(v, nd)} ({f3(ci[0], nd)} to {f3(ci[1], nd)})" if ci else f3(v, nd)


def stats_from_rows(rows: list) -> dict:
    """High and Watch+High precision/recall from a saved band table."""
    by = {r["band"]: r for r in rows}
    pos = sum(r["bankrupt"] for r in rows)
    n = sum(r["firms"] for r in rows)

    def one(bands):
        flagged = sum(by[b]["firms"] for b in bands)
        tp = sum(by[b]["bankrupt"] for b in bands)
        return {"flagged": flagged, "tp": tp, "fp": flagged - tp, "fn": pos - tp, "tn": n - flagged - (pos - tp),
                "precision": tp / flagged if flagged else None, "recall": tp / pos if pos else None}

    return {"high": one(["High"]), "watch_plus_high": one(["High", "Watch"])}


def plot_reliability(res: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    h = res["models"][HEADLINE]
    pts = h["reliability"]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    label, color, ls, marker = MODEL_STYLE[HEADLINE]
    lim = max(max(xs), max(ys)) * 1.1
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls=":", label="Perfect calibration")
    ax.plot(xs, ys, color=color, ls=ls, marker=marker, lw=1.6, label=f"{label} + Platt, test split")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted probability (10 equal-frequency bins)")
    ax.set_ylabel("Observed bankruptcy rate")
    ax.set_title("Test reliability, headline model (one-shot)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.patch.set_facecolor(SURFACE)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def comparison_table(res: dict, paths: Paths) -> list:
    phase4 = json.loads((paths.root / "reports" / "phase4_calibration.json").read_text())["models"][HEADLINE]
    models_val = json.loads((paths.root / "reports" / "models_val.json").read_text())[HEADLINE]
    sel = phase4["selected_method"]
    h = res["models"][HEADLINE]
    oof_m, val_m, test_m = phase4["oof_metrics"][sel], phase4["val"][sel], h["metrics"]
    val_ci, test_ci = phase4["val"][sel]["ci"], h["ci"]
    oof_s, val_s = stats_from_rows(phase4["bands"]["oof_table"]), stats_from_rows(phase4["bands"]["val_table"])
    test_s = h["band_stats"]
    oof_rows = {r["band"]: r for r in phase4["bands"]["oof_table"]}
    val_rows = {r["band"]: r for r in phase4["bands"]["val_table"]}
    test_rows = {r["band"]: r for r in h["band_table"]}

    def row(label, o, v, t):
        return f"| {label} | {o} | {v} | {t} |"

    def sub(stats, key, fld):
        return f3(stats[key][fld])

    lines = ["| Metric | Train OOF (independent of cuts) | Validation (cuts fitted here) | Test (one shot) |",
             "|---|---|---|---|",
             row("Firms, bankrupt", f"{sum(r['firms'] for r in oof_rows.values())}, "
                 f"{sum(r['bankrupt'] for r in oof_rows.values())}",
                 f"{sum(r['firms'] for r in val_rows.values())}, {sum(r['bankrupt'] for r in val_rows.values())}",
                 f"{res['rows']}, {res['bankrupt']}"),
             row("Prevalence (PR AUC baseline)", f3(oof_m["observed_rate"]), f3(val_m["observed_rate"]),
                 f3(res["prevalence"])),
             row("PR AUC (95% CI)", f3(oof_m["pr_auc"]), with_ci(val_m["pr_auc"], val_ci["pr_auc"]),
                 with_ci(test_m["pr_auc"], test_ci["pr_auc"])),
             row("ROC AUC (95% CI)", "not saved in Phase 4", with_ci(models_val["val"]["roc_auc"], models_val["val_ci"]["roc_auc_ci"]),
                 with_ci(test_m["roc_auc"], test_ci["roc_auc"])),
             row("Brier (95% CI)", f3(oof_m["brier"], 4), with_ci(val_m["brier"], val_ci["brier"], 4),
                 with_ci(test_m["brier"], test_ci["brier"], 4)),
             row("Log loss (95% CI)", f3(oof_m["log_loss"], 4), with_ci(val_m["log_loss"], val_ci["log_loss"], 4),
                 with_ci(test_m["log_loss"], test_ci["log_loss"], 4)),
             row("ECE, 10 bins (95% CI)", f3(oof_m["ece"], 4), with_ci(val_m["ece"], val_ci["ece"], 4),
                 with_ci(test_m["ece"], test_ci["ece"], 4)),
             row("Mean predicted probability", f3(oof_m["mean_pred"], 4), f3(val_m["mean_pred"], 4),
                 f3(test_m["mean_pred"], 4)),
             row("Watch band: firms, bankrupt, rate",
                 f"{oof_rows['Watch']['firms']}, {oof_rows['Watch']['bankrupt']}, {f3(oof_rows['Watch']['bankrupt_rate'])}",
                 f"{val_rows['Watch']['firms']}, {val_rows['Watch']['bankrupt']}, {f3(val_rows['Watch']['bankrupt_rate'])}",
                 f"{test_rows['Watch']['firms']}, {test_rows['Watch']['bankrupt']}, {f3(test_rows['Watch']['bankrupt_rate'])}"),
             row("High band: firms, bankrupt, rate",
                 f"{oof_rows['High']['firms']}, {oof_rows['High']['bankrupt']}, {f3(oof_rows['High']['bankrupt_rate'])}",
                 f"{val_rows['High']['firms']}, {val_rows['High']['bankrupt']}, {f3(val_rows['High']['bankrupt_rate'])}",
                 f"{test_rows['High']['firms']}, {test_rows['High']['bankrupt']}, {f3(test_rows['High']['bankrupt_rate'])}"),
             row("Low band: firms, bankrupt, rate",
                 f"{oof_rows['Low']['firms']}, {oof_rows['Low']['bankrupt']}, {f3(oof_rows['Low']['bankrupt_rate'])}",
                 f"{val_rows['Low']['firms']}, {val_rows['Low']['bankrupt']}, {f3(val_rows['Low']['bankrupt_rate'])}",
                 f"{test_rows['Low']['firms']}, {test_rows['Low']['bankrupt']}, {f3(test_rows['Low']['bankrupt_rate'])}"),
             row("High alone: precision (95% CI)", sub(oof_s, "high", "precision"), sub(val_s, "high", "precision"),
                 with_ci(test_s["high"]["precision"], test_ci["high_precision"])),
             row("High alone: recall (95% CI)", sub(oof_s, "high", "recall"), sub(val_s, "high", "recall"),
                 with_ci(test_s["high"]["recall"], test_ci["high_recall"])),
             row("Watch+High: precision (95% CI)", sub(oof_s, "watch_plus_high", "precision"),
                 sub(val_s, "watch_plus_high", "precision"),
                 with_ci(test_s["watch_plus_high"]["precision"], test_ci["flag_precision"])),
             row("Watch+High: recall (95% CI)", sub(oof_s, "watch_plus_high", "recall"),
                 sub(val_s, "watch_plus_high", "recall"),
                 with_ci(test_s["watch_plus_high"]["recall"], test_ci["flag_recall"]))]
    return lines


def confusion_lines(res: dict) -> list:
    c = res["models"][HEADLINE]["band_stats"]["high"]
    return ["| | Actually bankrupt | Actually healthy |", "|---|---|---|",
            f"| Flagged High | {c['tp']} (true positives) | {c['fp']} (false alarms) |",
            f"| Not High | {c['fn']} (missed) | {c['tn']} (true negatives) |"]


def reference_lines(res: dict, paths: Paths) -> list:
    phase4 = json.loads((paths.root / "reports" / "phase4_calibration.json").read_text())["models"]
    models_val = json.loads((paths.root / "reports" / "models_val.json").read_text())
    lines = ["| Model (calibration) | Split | PR AUC | ROC AUC | Brier | Log loss | ECE | Watch cut | High cut | High precision | Watch+High recall |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in REFERENCE:
        m = res["models"][name]
        sel = m["method"]
        v = phase4[name]["val"][sel]
        vs = stats_from_rows(phase4[name]["bands"]["val_table"])
        t, ts = m["metrics"], m["band_stats"]
        lines.append(f"| {name} ({sel}) | validation | {f3(v['pr_auc'])} | {f3(models_val[name]['val']['roc_auc'])} | "
                     f"{f3(v['brier'], 4)} | {f3(v['log_loss'], 4)} | {f3(v['ece'], 4)} | {m['t_watch']:.4f} | "
                     f"{m['t_high']:.4f} | {f3(vs['high']['precision'])} | {f3(vs['watch_plus_high']['recall'])} |")
        lines.append(f"| {name} ({sel}) | test | {with_ci(t['pr_auc'], m['ci']['pr_auc'])} | "
                     f"{with_ci(t['roc_auc'], m['ci']['roc_auc'])} | {f3(t['brier'], 4)} | {f3(t['log_loss'], 4)} | "
                     f"{f3(t['ece'], 4)} | {m['t_watch']:.4f} | {m['t_high']:.4f} | "
                     f"{f3(ts['high']['precision'])} | {f3(ts['watch_plus_high']['recall'])} |")
    return lines


def make_flags(res: dict, proto: dict) -> list:
    h = res["models"][HEADLINE]
    floor, target = proto["band_rule"]["high_min_precision"], proto["band_rule"]["watch_min_recall"]
    ts, rows = h["band_stats"], {r["band"]: r for r in h["band_table"]}
    ci = h["ci"]
    flags = []
    p_hi = ts["high"]["precision"]
    if p_hi is None:
        flags.append("No test firm landed in the High band, so High precision is undefined.")
    elif p_hi < floor:
        flags.append(f"High-band precision on test ({p_hi:.3f}) is below the {floor:.2f} floor the cut was built to reach on validation.")
    rec = ts["watch_plus_high"]["recall"]
    if rec is not None and rec < target:
        flags.append(f"Watch+High recall on test ({rec:.3f}) is below the {target:.2f} target the Watch cut reached on validation.")
    if rows["Watch"]["firms"] < 30:
        flags.append(f"The Watch band is thin on test ({rows['Watch']['firms']} firms, {rows['Watch']['bankrupt']} bankrupt), "
                     "so its rate carries almost no information. Fewer than 30 firms is the thin-band line.")
    mp, obs = h["metrics"]["mean_pred"], h["metrics"]["observed_rate"]
    if not (0.8 * obs <= mp <= 1.2 * obs):
        flags.append(f"Mean predicted probability ({mp:.4f}) is more than 20% away from the observed rate ({obs:.4f}), "
                     "so calibration-in-the-large is off.")
    if not (ci["ece"][0] <= h["metrics"]["ece"] <= ci["ece"][1]):
        flags.append("The ECE point estimate lies outside its own bootstrap interval (bootstrap ECE is biased upward on small data). Read ECE as descriptive.")
    for name in REFERENCE:
        if res["models"][name]["metrics"]["pr_auc"] > h["metrics"]["pr_auc"]:
            flags.append(f"{name} has a higher test PR AUC than the headline model. By protocol it is reference only and does not replace the headline; the user may choose to revisit.")
    undefined = {k: v for k, v in h["undefined_draws"].items() if v}
    if undefined:
        flags.append(f"Some bootstrap draws had an empty flagged group, so those precision intervals use fewer draws: {undefined}.")
    flags += [
        "One test set of 61 bankrupt firms. Every interval is wide, and this is a single draw, not a repeated trial.",
        "Random split only, no company id or year, so this says nothing about performance on future years.",
        "Calibrators and band cuts were fitted on fold models, while the scored model is refit on all of train. The test run does not isolate this mismatch.",
        "Attr21 (sales_growth) needs last year's sales, and it is the second strongest feature.",
        "Nothing was retrained, recalibrated or re-cut after seeing these results, by protocol.",
    ]
    return flags


EXPECTED_OPTIMISM = [
    "Validation band cuts: fitted on validation, so validation precision and recall at the cuts are in-sample and optimistic.",
    "Validation model choice: LightGBM was picked over two other models using validation, so its validation PR AUC is mildly optimistic (winner's curse).",
    "Train OOF calibration metrics: the Platt calibrator was fitted on those same OOF scores, so OOF Brier, log loss and ECE are in-sample for a two-parameter map (mildly optimistic).",
    "Train OOF model strength: OOF scores come from models trained on 80% of train, so they understate the refit model slightly.",
    "Test: no part of the model, calibrator or cuts saw it. A gap from validation in the direction of these optimism sources is expected. A gap the other way, or a much larger one, is not explained by them.",
]


def render_outputs(res: dict, proto: dict, paths: Paths) -> None:
    h = res["models"][HEADLINE]
    v = res["verdict"] if "verdict" in res else compute_verdict(res, proto)
    plot_reliability(res, paths.figure)
    verdict_head = ("HOLDS UP on the pre-stated rule" if v["holds_up"]
                    else "DOES NOT FULLY HOLD UP on the pre-stated rule")
    lines = [
        "# Phase 5: final test evaluation (one shot)",
        "",
        f"Generated by scripts/final_test_eval.py. Split: **{res['split']}**"
        f"{' (DRY RUN on validation, not the test result)' if res['dry_run'] else ''}. "
        f"{res['rows']} firms, {res['bankrupt']} bankrupt. Seed {res['seed']}, {res['bootstrap_n']} stratified bootstrap resamples.",
        f"Headline: {HEADLINE} with {h['method']} calibration, cuts Watch {h['t_watch']:.4f} and High {h['t_high']:.4f}, "
        f"frozen at commit {proto['frozen_commit']}. Nothing was retrained, recalibrated or re-cut.",
        "",
        f"## Verdict rule result: {verdict_head}",
        "",
        "The rule was fixed before the run (docs/decisions.md, Phase 5 protocol).",
        "",
        *[f"- ({c['id']}) {c['rule'][3:]}: **{'pass' if c['pass'] else 'fail'}** ({c['detail']})" for c in v["criteria"]],
        "",
        "## Test next to validation and train out-of-fold (headline model)",
        "",
        *comparison_table(res, paths),
        "",
        "Which gaps are expected optimism (stated before the run):",
        "",
        *[f"- {t}" for t in EXPECTED_OPTIMISM],
        "",
        "## Confusion matrix at the High cut (test)",
        "",
        *confusion_lines(res),
        "",
        "## Reference models (deep learning and linear comparison, not eligible to replace the headline)",
        "",
        *reference_lines(res, paths),
        "",
        "![Test reliability](figures/phase5_test_reliability.png)",
        "",
        "## Flags",
        "",
        *[f"{i}. {f}" for i, f in enumerate(make_flags(res, proto), 1)],
        "",
        "## Provenance",
        "",
        f"- Git HEAD at run: `{res['git_head']}`. Protocol file SHA256: `{res['protocol_sha256']}`.",
        f"- Test file SHA256 (bytes read once): `{res['data_sha256']}`.",
        f"- Versions: {json.dumps(res['versions'])}.",
        "",
    ]
    paths.report_md.parent.mkdir(parents=True, exist_ok=True)
    paths.report_md.write_text(NL.join(lines), encoding="utf-8")


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="use the validation split, write to --out-dir")
    ap.add_argument("--render-only", action="store_true", help="rebuild markdown and figure from saved results")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    np.random.seed(SEED)
    if args.freeze:
        proto = freeze()
        fill_decisions(proto, ROOT / "docs" / "decisions.md")
        print(f"wrote {PROTOCOL.relative_to(ROOT)} and filled the hash table in docs/decisions.md")
        return
    if args.render_only:
        paths = Paths()
        render_outputs(json.loads(paths.results.read_text()), load_protocol(paths), paths)
        print("re-rendered from saved results (test split not read)")
        return
    if args.dry_run:
        out = Path(args.out_dir or ROOT / "reports" / "_phase5_dry_run")
        paths = Paths(lock=out / "phase5.lock", results=out / "results.json", report_md=out / "report.md",
                      figure=out / "reliability.png")
        run(paths, split="val", dry=True, n_boot=100)
        print(f"dry run on the validation split written to {out}")
        return
    if not tree_clean():
        sys.exit("working tree is not clean: commit the protocol and script first")
    res = run(Paths(), split="test", dry=False)
    print(f"done. verdict rule holds_up={res['verdict']['holds_up']}. Results are locked.")


if __name__ == "__main__":
    main()
