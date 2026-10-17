"""Phase 6: SHAP explanations. The Platt slope conversion must reproduce calibrated log-odds."""

import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap

from scripts import run_phase6
from src import calibrate as C
from src import explain as E
from src.polish_features import CATEGORIES, CATEGORY, FEATURE_NAMES, FEATURES
from src.preprocess import PROCESSED, ROOT
from tests.test_no_test_access import guard  # noqa: F401  (pytest fixture)

SEED = 42
needs_frozen = pytest.mark.skipif(
    not ((ROOT / "models" / "lightgbm.joblib").exists() and (PROCESSED / "polish_5year_val.csv").exists()
         and (ROOT / "artifacts" / "calibration" / "bands.json").exists()),
    reason="needs the saved model, calibration artifacts and processed data",
)


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(700, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    X.loc[rng.random(len(X)) < 0.1, "sales_growth"] = np.nan
    z = 1.2 * X.iloc[:, 0] - 0.9 * X.iloc[:, 1] + 1.5 * X["sales_growth"].isna() - 3.0
    y = pd.Series((rng.random(len(X)) < 1 / (1 + np.exp(-z))).astype(int), name="bankrupt")
    model = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, random_state=SEED, n_jobs=1, verbose=-1,
                               deterministic=True, force_row_wise=True).fit(X.iloc[:450], y.iloc[:450])
    cal = C.PlattCalibrator().fit(model.predict_proba(X.iloc[:450])[:, 1], y.iloc[:450])
    return {"X_tr": X.iloc[:450], "y_tr": y.iloc[:450], "X_va": X.iloc[450:], "y_va": y.iloc[450:],
            "model": model, "cal": cal}


# ------------------------------------------------------------------ Platt slope conversion
def test_slope_conversion_reproduces_calibrated_log_odds(synth):
    chk = E.check_slope_conversion(synth["model"], synth["X_va"], synth["cal"])
    assert chk["ok"], chk
    assert chk["max_abs_error"] <= E.TOL and chk["clipped"] == 0


def test_converted_shap_sums_to_calibrated_logit_row_by_row(synth):
    out = E.explain_dataset(synth["model"], synth["cal"], synth["X_va"])
    p = synth["cal"].transform(synth["model"].predict_proba(synth["X_va"])[:, 1])
    assert out["scale"] == "calibrated_log_odds"
    assert np.max(np.abs(out["base"] + out["phi"].sum(1) - np.log(p / (1 - p)))) <= E.TOL


def test_check_fails_when_calibrator_was_not_fitted_on_raw_log_odds(synth):
    class ProbabilityScalePlatt(C.PlattCalibrator):
        """Fitted on the probability instead of the log-odds: the slope conversion does not apply."""

        def transform(self, scores):
            return 1 / (1 + np.exp(-(self.a * np.asarray(scores, dtype=float) + self.b)))

    bad = ProbabilityScalePlatt(a=3.0, b=-2.0)
    assert not E.check_slope_conversion(synth["model"], synth["X_va"], bad)["ok"]
    iso = C.IsotonicCalibrator().fit(synth["model"].predict_proba(synth["X_tr"])[:, 1], synth["y_tr"])
    assert not E.check_slope_conversion(synth["model"], synth["X_va"], iso)["ok"]


def test_pipeline_falls_back_to_raw_scale_when_check_fails(synth):
    out = E.explain_dataset(synth["model"], C.NoCalibration(), synth["X_va"])
    assert out["scale"] == "raw_log_odds" and out["slope"] == 1.0
    raw = synth["model"].predict(synth["X_va"], raw_score=True)
    assert np.max(np.abs(out["base"] + out["phi"].sum(1) - raw)) <= 1e-9


@needs_frozen
def test_frozen_calibrator_slope_conversion_on_real_validation():
    import joblib

    from src.train import load_split

    art = C.load_artifacts(ROOT / "artifacts" / "calibration")
    cal = art["calibrators"]["lightgbm"][art["bands"]["lightgbm"]["selected_method"]]
    model = joblib.load(ROOT / "models" / "lightgbm.joblib")
    X, _ = load_split("val")
    chk = E.check_slope_conversion(model, X, cal)
    assert chk["ok"], chk
    out = E.explain_dataset(model, cal, X)
    p = cal.transform(model.predict_proba(X)[:, 1])
    assert np.max(np.abs(out["base"] + out["phi"].sum(1) - np.log(p / (1 - p)))) <= E.TOL


# ------------------------------------------------------------------------ SHAP checks
def test_additivity_and_agreement_with_native(synth):
    out = E.explain_dataset(synth["model"], synth["cal"], synth["X_va"])
    assert out["additivity_error_raw"] <= 1e-9
    assert out["max_diff_vs_native"] <= 1e-9


def test_importance_is_deterministic(synth):
    ex = E.explain_dataset(synth["model"], synth["cal"], synth["X_va"])
    tr = E.explain_dataset(synth["model"], synth["cal"], synth["X_tr"])
    gain = synth["model"].booster_.feature_importance("gain")
    a = E.importance(ex["phi"], synth["y_va"], tr["phi"], gain, n_boot=30)
    b = E.importance(ex["phi"], synth["y_va"], tr["phi"], gain, n_boot=30)
    assert a == b
    assert abs(sum(c["mean_abs_sum_val"] for c in a["category"].values()) - a["total_mean_abs_val"]) < 1e-9


def test_explain_firm_matches_dataset_values(synth):
    bands = {"t_watch": 0.1, "t_high": 0.3}
    ds = E.explain_dataset(synth["model"], synth["cal"], synth["X_va"])
    i = 3
    e = E.explain_firm(synth["model"], synth["cal"], synth["X_va"].iloc[[i]], bands, top_k=4)
    assert len(e["drivers"]) == 4 and e["scale"] == "calibrated_log_odds"
    mags = [abs(d["contribution"]) for d in e["drivers"]]
    assert mags == sorted(mags, reverse=True)
    idx = {f[0]: k for k, f in enumerate(FEATURES)}
    for d in e["drivers"]:
        assert abs(d["contribution"] - ds["phi"][i, idx[d["attr"]]]) < 1e-9
    p = synth["cal"].transform(synth["model"].predict_proba(synth["X_va"].iloc[[i]])[:, 1])[0]
    assert abs(e["probability"] - p) < 1e-12


# ---------------------------------------------------------------------- dictionary
def test_feature_dictionary_labels_categories_and_unverified():
    d = {r["attr"]: r for r in E.feature_dictionary()}
    assert len(d) == 64 and set(CATEGORY) == set(d)
    assert set(CATEGORY.values()) <= set(CATEGORIES)
    for attr in ("Attr19", "Attr24", "Attr38", "Attr41", "Attr29", "Attr55"):
        assert d[attr]["unverified"], attr
    for attr in ("Attr1", "Attr2", "Attr9", "Attr21", "Attr27", "Attr37"):
        assert not d[attr]["unverified"], attr
    for attr, r in d.items():
        assert r["label"].startswith(f"{attr}: ")
        assert r["unverified"] == ("UNVERIFIED" in r["label"])
        if r["unverified"]:
            assert f"UNVERIFIED {attr}" in r["label"]
    assert all(r["formula"] for r in d.values())


def test_category_index_partitions_all_features():
    cats = E.category_index()
    flat = sorted(i for ix in cats.values() for i in ix)
    assert flat == list(range(64))
    assert set(E.BORDERLINE) <= set(CATEGORY)


# ---------------------------------------------------------------------- Attr21
def test_missingness_table_counts_and_wilson():
    X = pd.DataFrame({"sales_growth": [np.nan, np.nan, 1.0, 2.0, 3.0, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 1])
    t = E.missingness_table(X, y, X, y)
    assert t["train"]["missing"] == {"firms": 3, "bankrupt": 2, "rate": 2 / 3, "wilson_95": E.wilson(2, 3)}
    assert t["val"]["present"]["firms"] == 3 and t["val"]["present"]["bankrupt"] == 1
    lo, hi = E.wilson(2, 3)
    assert 0 <= lo < 2 / 3 < hi <= 1


def test_pick_examples_follows_the_stated_rule():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.05, 0.12, 0.8, 0.2, 0.1])
    ex = run_phase6.pick_examples(y, p, t_watch=0.15)
    assert ex == {"highest_scoring_true_positive": 0, "highest_scoring_false_alarm": 3,
                  "nearest_miss_below_watch_cut": 2}


# ----------------------------------------------------------------------- pipeline
def test_phase6_pipeline_never_touches_test_split(guard, synth, tmp_path):
    bands = {"t_watch": 0.1, "t_high": 0.3}
    res = run_phase6.run(synth["X_tr"], synth["y_tr"], synth["X_va"], synth["y_va"], synth["model"], synth["cal"],
                         bands, reports=tmp_path / "reports", artifacts=tmp_path / "artifacts",
                         figures=tmp_path / "figures", n_boot=20)
    assert guard == []
    assert res["scale"] == "calibrated_log_odds"
    for f in ("phase6_importance.png", "phase6_categories.png", "phase6_beeswarm.png", "phase6_dependence.png"):
        assert (tmp_path / "figures" / f).exists()
    for f in ("feature_dictionary.json", "global_importance.json", "config.json"):
        assert (tmp_path / "artifacts" / f).exists()
    text = (tmp_path / "reports" / "phase6_explanations.md").read_text(encoding="utf-8")
    assert "UNVERIFIED" in text and "project" in text and "—" not in text


@needs_frozen
def test_verify_frozen_passes_and_detects_tampering(tmp_path):
    run_phase6.verify_frozen()
    for rel in run_phase6.FROZEN:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    shutil.copy(ROOT / "artifacts" / "phase5_protocol.json", tmp_path / "artifacts" / "phase5_protocol.json")
    run_phase6.verify_frozen(tmp_path)
    with open(tmp_path / "artifacts" / "calibration" / "bands.json", "ab") as fh:
        fh.write(b" ")
    with pytest.raises(RuntimeError, match="mismatch"):
        run_phase6.verify_frozen(tmp_path)


def test_shap_is_pinned_to_the_installed_version():
    lines = (ROOT / "requirements.txt").read_text().split()
    assert f"shap=={shap.__version__}" in lines
