"""US model: fit on 1999 to 2011, report on 2012 to 2014. The test years are never loaded here.

Protocol: docs/us_protocol.md. Writes artifacts/us/us_model.json and reports/us_validation.{json,md}.
Run: python -m scripts.run_us_model
"""

import json

import numpy as np

from src import us_data as D
from src import us_model as M
from src.preprocess import ROOT

OUT = ROOT / "artifacts" / "us" / "us_model.json"
REPORTS = ROOT / "reports"
NAMES = ("lightgbm", "logistic")


def pct(x, nd=1):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(v, nd=3):
    return f"{f3(v[0], nd)} to {f3(v[1], nd)}"


def main() -> None:
    train, val = D.load_period("train"), D.load_period("val")
    Xtr, ytr, gtr = train[D.FEATURES].to_numpy(float), train["y"].to_numpy(), train["company_name"].to_numpy()
    Xva, yva = val[D.FEATURES].to_numpy(float), val["y"].to_numpy()
    fits = {n: M.fit_all(n, Xtr, ytr, gtr) for n in NAMES}
    raw, cal = {}, {}
    for n in NAMES:
        raw[n], cal[n] = M.predict(fits[n], Xva)
    idx = M.cluster_indices(val["company_name"].to_numpy())
    res = M.evaluate(yva, val["company_name"].to_numpy(), val["year"].to_numpy(), raw, cal,
                     {n: fits[n]["cuts"] for n in NAMES}, idx, y_company=val["y_company"].to_numpy())
    seen = D.seen_mask(train, val)
    res["seen_in_train"] = {n: {"seen": M.subgroup_pr(yva, raw[n], seen), "unseen": M.subgroup_pr(yva, raw[n], ~seen)} for n in NAMES}
    res["oof_train"] = {n: {"pr_auc": float(__import__("sklearn.metrics", fromlist=["x"]).average_precision_score(ytr, fits[n]["oof_raw"])),
                            "calibration": M.calibration(ytr, fits[n]["oof_cal"])} for n in NAMES}
    res["data"] = {"train": D.counts(train), "val": D.counts(val), "train_by_year": D.per_year(train), "val_by_year": D.per_year(val),
                   "companies_in_both": int(len(set(train["company_name"]) & set(val["company_name"])))}
    res["platt"] = {n: {"a": fits[n]["platt"].a, "b": fits[n]["platt"].b} for n in NAMES}
    M.save_artifact(M.to_artifact(fits, {"train_years": D.PERIODS["train"], "seed": M.SEED, "label": "one-year-ahead (last row of failed company)"}), OUT)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "us_validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REPORTS / "us_validation.md").write_text(render(res), encoding="utf-8")
    print("wrote", OUT.name, "and reports/us_validation.{json,md}")


def render(r: dict) -> str:
    d = r["data"]
    L = ["# US model, validation report", "",
         "Protocol: docs/us_protocol.md. Fitted on 1999 to 2011, reported on 2012 to 2014. The test years were not loaded. "
         "Every number below is written by scripts/run_us_model.py.", "", "## Data", "",
         "| Split | Firm-years | Companies | Positives | Positive rate | Positives under the provided company label |", "|---|---|---|---|---|---|"]
    for k, t in (("train", "Train 1999 to 2011"), ("val", "Validation 2012 to 2014")):
        c = d[k]
        L.append(f"| {t} | {c['rows']} | {c['companies']} | {c['positives']} | {pct(c['positive_rate'], 2)} | {c['positives_company_label']} |")
    L += ["", f"Companies present in both train and validation: {d['companies_in_both']}.", "",
          "Positives by year (derived one-year-ahead label):", "", "| Year | Firm-years | Positives | Rate |", "|---|---|---|---|"]
    for row in d["train_by_year"] + d["val_by_year"]:
        L.append(f"| {row['year']} | {row['rows']} | {row['positives']} | {pct(row['positive_rate'], 2)} |")
    L += ["", f"## Discrimination on validation (no-skill PR AUC {f3(r['no_skill_pr_auc'], 4)}, {r['positives']} positives in {r['rows']} firm-years)", "",
          "Intervals resample companies (1000 draws, seed 42).", "",
          "| Model | PR AUC (95% CI) | ROC AUC (95% CI) | PR AUC on provided company label (diagnostic) |", "|---|---|---|---|"]
    for n, m in r["models"].items():
        L.append(f"| {n} | {f3(m['pr_auc'])} ({ci(m['pr_auc_ci'])}) | {f3(m['roc_auc'])} ({ci(m['roc_auc_ci'])}) | "
                 f"{f3(m['pr_auc_company_label'])} (no skill {f3(m['no_skill_company_label'], 4)}) |")
    pdif = r["paired_difference"]
    L += ["", f"Paired difference, {pdif['a']} minus {pdif['b']}: PR AUC {f3(pdif['pr_auc'])} ({ci(pdif['pr_auc_ci'])}), "
          f"ROC AUC {f3(pdif['roc_auc'])} ({ci(pdif['roc_auc_ci'])}).", "",
          "## Calibration (Platt fitted on train out-of-fold, not refitted)", "",
          "| Model | Mean predicted | Observed rate | Brier | ECE |", "|---|---|---|---|---|"]
    for n, m in r["models"].items():
        c = m["calibration"]
        L.append(f"| {n} | {pct(c['mean_predicted'], 2)} | {pct(c['observed_rate'], 2)} | {f3(c['brier'], 5)} | {f3(c['ece'], 5)} |")
    L += ["", "Per validation year, mean predicted against observed:", "", "| Model | Year | Positives | Mean predicted | Observed |", "|---|---|---|---|---|"]
    for n, m in r["models"].items():
        for y in m["by_year"]:
            L.append(f"| {n} | {y['year']} | {y['positives']} | {pct(y['mean_predicted'], 2)} | {pct(y['observed_rate'], 2)} |")
    L += ["", "## Flagging the top of the ranking (cuts from train out-of-fold percentiles)", "",
          "| Model | Flag | Cut | Share flagged on validation | Precision | Recall |", "|---|---|---|---|---|---|"]
    for n, m in r["models"].items():
        for k, f in m["flag"].items():
            L.append(f"| {n} | {k} | {pct(f['cut'], 2)} | {pct(f['share_flagged'])} | {pct(f['precision'])} | {pct(f['recall'])} |")
    L += ["", "## Firms seen in train against firms not seen", "", "| Model | Group | Firm-years | Positives | PR AUC |", "|---|---|---|---|---|"]
    for n, g in r["seen_in_train"].items():
        for k, v in g.items():
            L.append(f"| {n} | {k} | {v['rows']} | {v['positives']} | {f3(v['pr_auc'])} |")
    L += ["", "PR AUC is shown only where a group has at least 5 positives.", "", "## Train out-of-fold (company-grouped)", "",
          "| Model | PR AUC | Mean predicted | Observed |", "|---|---|---|---|"]
    for n, v in r["oof_train"].items():
        L.append(f"| {n} | {f3(v['pr_auc'])} | {pct(v['calibration']['mean_predicted'], 2)} | {pct(v['calibration']['observed_rate'], 2)} |")
    L += ["", "## Limits", "",
          "- The 18 variables have no dictionary in the repo and their units look mixed, so they are used raw and not interpreted.",
          "- The label is derived from the README statement that the last fiscal year is the one before the filing.",
          "- Failures cluster in 2002 to 2009, so each period has a different regime and base rate.",
          "- US listed firms only. Not comparable to the Polish figures, because the label, base rate and features differ.",
          "- Nothing here is deployed and no headline model is chosen. That is the user's decision."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
