"""US model: one-shot evaluation on the test years (2015 to 2018). Run only after the user approves.

    python -m scripts.us_final_test

Frozen artifact only (artifacts/us/us_model.json): nothing is refit, recalibrated or re-cut. The run refuses if the lock or the
results file exists. The lock is written BEFORE the test rows are read, so a crash after the read cannot be followed by a second
read. Protocol: docs/us_protocol.md.
"""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from src import us_data as D
from src import us_model as M
from src.preprocess import ROOT
from scripts.run_us_model import NAMES, ci, f3, pct

ART = ROOT / "artifacts" / "us" / "us_model.json"
LOCK = ROOT / "artifacts" / "us" / "test.lock"
RESULTS = ROOT / "reports" / "us_test_results.json"
REPORT = ROOT / "reports" / "us_test_results.md"


def main() -> None:
    if LOCK.exists() or RESULTS.exists():
        raise SystemExit("The US test years were already read. This script refuses a second read.")
    digest = hashlib.sha256(ART.read_bytes()).hexdigest()
    LOCK.write_text(json.dumps({"artifact_sha256": digest, "time": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    art = json.loads(ART.read_text(encoding="utf-8"))
    test = D.final_read()
    X, y = test[D.FEATURES].to_numpy(float), test["y"].to_numpy()
    raw, cal = {}, {}
    for n in NAMES:
        raw[n], cal[n] = M.predict_from_artifact(art, n, X)
    idx = M.cluster_indices(test["company_name"].to_numpy())
    res = M.evaluate(y, test["company_name"].to_numpy(), test["year"].to_numpy(), raw, cal,
                     {n: art["models"][n]["cuts"] for n in NAMES}, idx, y_company=test["y_company"].to_numpy())
    res["data"] = {"test": D.counts(test), "test_by_year": D.per_year(test), "artifact_sha256": digest}
    RESULTS.write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# US model, one-shot test results (2015 to 2018)", "", f"Frozen artifact sha256 {digest[:16]}. Read once. Protocol: docs/us_protocol.md.", "",
         f"No-skill PR AUC {f3(res['no_skill_pr_auc'], 4)}, {res['positives']} positives in {res['rows']} firm-years.", "",
         "| Model | PR AUC (95% CI) | ROC AUC (95% CI) | Mean predicted | Observed |", "|---|---|---|---|---|"]
    for n, m in res["models"].items():
        c = m["calibration"]
        L.append(f"| {n} | {f3(m['pr_auc'])} ({ci(m['pr_auc_ci'])}) | {f3(m['roc_auc'])} ({ci(m['roc_auc_ci'])}) | "
                 f"{pct(c['mean_predicted'], 2)} | {pct(c['observed_rate'], 2)} |")
    L += ["", "| Model | Flag | Share flagged | Precision | Recall |", "|---|---|---|---|---|"]
    for n, m in res["models"].items():
        for k, f in m["flag"].items():
            L.append(f"| {n} | {k} | {pct(f['share_flagged'])} | {pct(f['precision'])} | {pct(f['recall'])} |")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", REPORT.name)


if __name__ == "__main__":
    main()
