"""US model (protocol: docs/us_protocol.md): label rule, split guard, company-level resampling and the one-shot script."""

import re

import numpy as np
import pandas as pd
import pytest

from src import us_data as D
from src import us_model as M
from src.preprocess import ROOT


def frame():
    rows = [("A", 2000, "failed"), ("A", 2001, "failed"), ("A", 2002, "failed"), ("B", 2000, "alive"), ("B", 2001, "alive"),
            ("C", 2005, "failed")]
    return pd.DataFrame([{"company_name": c, "fyear": float(y), "status_label": s} for c, y, s in rows])


def test_label_is_positive_only_on_a_failed_companys_last_year():
    d = D.derive_labels(frame())
    pos = d[d["y"] == 1][["company_name", "year"]].values.tolist()
    assert pos == [["A", 2002], ["C", 2005]]
    assert d["y_company"].sum() == 4 and d[d["company_name"] == "B"]["y"].sum() == 0


def test_test_years_cannot_be_loaded_through_the_normal_loader():
    with pytest.raises(PermissionError):
        D.load_period("test")
    assert D.PERIODS["train"][1] + 1 == D.PERIODS["val"][0] and D.PERIODS["val"][1] + 1 == D.PERIODS["test"][0]


def test_cluster_bootstrap_draws_whole_companies():
    groups = np.array(["a"] * 3 + ["b"] * 2 + ["c"])
    for ix in M.cluster_indices(groups, n=20):
        counts = {g: int((groups[ix] == g).sum()) for g in "abc"}
        assert counts["a"] % 3 == 0 and counts["b"] % 2 == 0


def test_logit_artifact_roundtrip_and_slog_is_odd():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 18)) * 100
    y = (X[:, 0] + rng.normal(size=200) * 50 > 0).astype(int)
    m = M.LogitModel().fit(X, y)
    m2 = M.LogitModel.from_dict(m.to_dict())
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))
    assert np.allclose(M.slog(-X), -M.slog(X))


def test_validation_script_never_touches_the_test_years_and_final_script_locks_first():
    run = (ROOT / "scripts" / "run_us_model.py").read_text(encoding="utf-8")
    assert "final_read" not in run and not re.search(r"load_period\(\s*[\"']test", run)
    fin = (ROOT / "scripts" / "us_final_test.py").read_text(encoding="utf-8")
    assert fin.index("LOCK.write_text") < fin.index("D.final_read()")
    for rel in ("src/us_data.py", "src/us_model.py", "scripts/run_us_model.py", "scripts/us_final_test.py", "docs/us_protocol.md"):
        assert "—" not in (ROOT / rel).read_text(encoding="utf-8"), rel


art = ROOT / "artifacts" / "us" / "us_model.json"


@pytest.mark.skipif(not (art.exists() and D.RAW_ZIP.exists()), reason="needs the raw zip and the fitted artifact")
def test_artifact_scores_validation_like_the_report_says():
    import json

    from sklearn.metrics import average_precision_score

    val = D.load_period("val")
    rep = json.loads((ROOT / "reports" / "us_validation.json").read_text(encoding="utf-8"))
    a = json.loads(art.read_text(encoding="utf-8"))
    for n in ("lightgbm", "logistic"):
        raw, _ = M.predict_from_artifact(a, n, val[D.FEATURES].to_numpy(float))
        assert average_precision_score(val["y"], raw) == pytest.approx(rep["models"][n]["pr_auc"], abs=1e-9)
    assert rep["data"]["val"]["rows"] == len(val) and max(val["year"]) == D.PERIODS["val"][1]
