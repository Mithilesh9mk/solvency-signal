"""Phase 7: Attr21 missingness analysis. Descriptive only, train and validation only."""

import numpy as np
import pandas as pd
import pytest

from scripts import run_phase7_attr21 as P7
from src.preprocess import ROOT
from tests.test_explain import synth  # noqa: F401  (pytest fixture)
from tests.test_no_test_access import guard  # noqa: F401  (pytest fixture)

BANDS = {"t_watch": 0.1, "t_high": 0.3}


def small():
    X = pd.DataFrame({"sales_growth": [np.nan, np.nan, 1.0, 2.0, 3.0, 1.5],
                      "sales_to_assets": [0.5, 0.0, 1.0, 1.2, np.nan, 0.9],
                      "total_sales_to_assets": [0.5, 0.0, 1.0, 1.2, 0.3, 0.9],
                      "cash_ratio": [np.nan, 1.0, np.nan, 2.0, 3.0, 4.0]})
    y = pd.Series([1, 1, 0, 0, 1, 0])
    return X, y


def test_counts_include_shares_of_all_bankrupt():
    X, y = small()
    c = P7.counts(X, y, X, y)
    assert c["train"]["missing"]["firms"] == 2 and c["train"]["missing"]["bankrupt"] == 2
    assert c["train"]["missing_share_of_all_bankrupt"] == pytest.approx(2 / 3)
    assert c["val"]["missing_share_of_firms"] == pytest.approx(2 / 6)


def test_sales_check_and_comissing_on_a_known_frame():
    X, _ = small()
    s = P7.sales_check(X)
    assert s["firms_missing"] == 2 and s["sales_to_assets"] == {"equal_to_zero": 1, "missing": 0}
    cm = P7.comissing(X, top=2)
    assert cm["count_distribution_missing_group"] == {1: 1, 2: 1} or cm["count_distribution_missing_group"] == {2: 1, 1: 1}
    assert cm["median_missing_attributes"]["missing"] == 1.5
    assert len(cm["top_attributes"]) == 2


def test_auc_block_is_withheld_for_tiny_classes():
    y = np.array([1] * 11 + [0])
    out = P7.auc_block(y, np.linspace(0, 1, 12), n_boot=10)
    assert out["pr_auc"] is None and "not reported" in out["note"]
    ok = P7.auc_block(np.array([1] * 8 + [0] * 12), np.linspace(0, 1, 20), n_boot=10)
    assert ok["pr_auc"] is not None and len(ok["pr_auc_ci"]) == 2


def test_what_if_does_not_mutate_inputs_and_counts_add_up(synth):
    X_va, X_tr = synth["X_va"].copy(), synth["X_tr"]
    before = synth["model"].predict_proba(X_va)[:, 1].copy()
    w = P7.what_if(synth["model"], synth["cal"], BANDS, synth["X_va"], X_tr)
    pd.testing.assert_frame_equal(X_va, synth["X_va"])
    np.testing.assert_array_equal(before, synth["model"].predict_proba(synth["X_va"])[:, 1])
    miss = synth["X_va"]["sales_growth"].isna().sum()
    assert w["present_then_blanked"]["firms"] == len(X_va) - miss
    assert w["missing_then_filled_with_train_median"]["firms"] == miss
    for tag in ("present_then_blanked", "missing_then_filled_with_train_median"):
        assert sum(w[tag]["transitions"].values()) == w[tag]["firms"]
        assert sum(w[tag]["band_counts_after"].values()) == w[tag]["firms"]


def test_high_band_composition_adds_up(synth):
    p = synth["cal"].transform(synth["model"].predict_proba(synth["X_va"])[:, 1])
    perf = P7.performance(synth["X_va"], synth["y_va"], p, BANDS, n_boot=10)
    h = perf["high_band_composition"]
    assert h["high_firms"] == h["high_with_attr21_missing"] + h["high_with_attr21_present"]
    assert perf["all"]["firms"] == perf["present"]["firms"] + perf["missing"]["firms"]


def test_phase2_ablation_is_quoted_from_the_phase2_report():
    a = P7.phase2_ablation()
    if (ROOT / "reports" / "models_val.md").exists():
        assert a["pr_auc_attr21_dropped"] is not None and a["pr_auc_all_features"] is not None
        assert a["pr_auc_attr21_dropped"] < a["pr_auc_all_features"]


def test_pipeline_never_touches_test_split_and_writes_outputs(guard, synth, tmp_path):
    res = P7.run(synth["X_tr"], synth["y_tr"], synth["X_va"], synth["y_va"], synth["model"], synth["cal"], BANDS,
                 ["Attr1", "Attr2"], reports=tmp_path / "r", figures=tmp_path / "f", n_boot=10)
    assert guard == []
    assert (tmp_path / "f" / "phase7_attr21.png").exists()
    text = (tmp_path / "r" / "phase7_attr21.md").read_text(encoding="utf-8")
    assert "Flags" in text and "—" not in text
    assert len(res["flags"]) >= 8
