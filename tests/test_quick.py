"""Phase 10: the 4-ratio Quick check. Pure logic, no data files, no test split."""

import json
import math

import numpy as np
import pytest

from src import quick as Q
from src.preprocess import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / Q.QUICK_PATH).exists(), reason="needs artifacts/quick/quick_model.json")
BASE = {"current_ratio": 1.5, "debt_to_equity": 1.0, "net_profit_margin": 0.05, "cash_earnings_ratio": 0.15}


@pytest.fixture(scope="module")
def model():
    return Q.load()


def prob(model, **kw):
    return Q.assess({**BASE, **kw}, model)["probability"]


def test_bands_cover_one_to_ten_and_cut_edges_count_as_above():
    assert [Q.band_for(s) for s in range(1, 11)] == ["Low"] * 3 + ["Watch"] * 3 + ["Elevated"] * 2 + ["High"] * 2
    with pytest.raises(ValueError):
        Q.band_for(0)
    cuts = list(np.linspace(0.1, 0.9, 9))
    assert Q.score_for(0.05, cuts) == 1 and Q.score_for(0.1, cuts) == 2 and Q.score_for(0.99, cuts) == 10


@pytest.mark.parametrize("key,direction", [("current_ratio", -1), ("net_profit_margin", -1), ("cash_earnings_ratio", -1),
                                           ("debt_to_equity", +1)])
def test_risk_moves_the_pre_registered_way(model, key, direction):
    grid = np.linspace(0.05, 6, 80) if key in ("current_ratio", "debt_to_equity") else np.linspace(-0.6, 0.6, 80)
    ps = np.array([prob(model, **{key: float(v)}) for v in grid])
    assert np.all(np.diff(ps) * direction >= -1e-12), f"{key} is not monotone in the registered direction"


def test_scores_stay_in_range_and_extremes_do_not_crash(model):
    for kw in ({}, {"debt_to_equity": 0}, {"debt_to_equity": -3}, {"current_ratio": 1e9}, {"net_profit_margin": -50},
               {"cash_earnings_ratio": 1e6}, {"debt_to_equity": 1e-12}):
        out = Q.assess({**BASE, **kw}, model)
        assert 1 <= out["score"] <= 10 and 0 < out["probability"] < 1 and out["band"] == Q.band_for(out["score"])
    assert prob(model, debt_to_equity=0) <= prob(model, debt_to_equity=1.0)  # no debt is not riskier than debt


@pytest.mark.parametrize("bad", [{"current_ratio": float("nan")}, {"debt_to_equity": float("inf")}, {"net_profit_margin": "abc"},
                                 {"cash_earnings_ratio": None}])
def test_bad_inputs_are_refused(model, bad):
    with pytest.raises(ValueError):
        Q.assess({**BASE, **bad}, model)
    with pytest.raises(ValueError):
        Q.assess({k: v for k, v in BASE.items() if k != "current_ratio"}, model)


def test_zero_debt_to_equity_maps_to_infinite_equity_ratio_and_negative_equity_is_kept():
    assert Q.to_model_space({**BASE, "debt_to_equity": 0})[1] == math.inf
    assert Q.to_model_space({**BASE, "debt_to_equity": 2.0})[1] == 0.5
    assert Q.to_model_space({**BASE, "debt_to_equity": -2.0})[1] == -0.5


def test_contributions_and_baseline_add_up_to_the_probability(model):
    out = Q.assess(BASE, model)
    total = out["baseline_logodds"] + sum(out["contributions"].values())
    assert abs(Q.sigmoid(total) - out["probability"]) < 1e-9
    x = Q.to_model_space(BASE)
    assert abs(float(Q.predict_proba(x, model)[0]) - out["probability"]) < 1e-9


def test_flags_mark_inputs_outside_the_training_range(model):
    assert Q.assess(BASE, model)["outside_range"] == []
    assert "current_ratio" in Q.assess({**BASE, "current_ratio": 1e6}, model)["outside_range"]


def test_artifact_matches_the_report_and_the_protocol(model):
    rep = json.loads((ROOT / "reports" / "phase10_quick.json").read_text(encoding="utf-8"))
    assert rep["cuts"] == model["cuts"]
    assert rep["table_train_oof"] == model["score_rates"]
    assert rep["cv"]["challenger_wins"] and rep["cv"]["difference"] >= rep["protocol_margin"]
    assert sum(r["n"] for r in model["score_rates"]) == model["n_train"]
    assert sum(r["bankrupt"] for r in model["score_rates"]) == model["n_train_bankrupt"]
    assert len(model["cuts"]) == 9 and model["features"] == Q.FEATURE_COLS
    assert all(r["n"] > 300 for r in model["score_rates"])  # tenths of about 4000 firms
    assert not rep["stopped"]


def test_validate_rejects_a_broken_file():
    good = json.loads((ROOT / Q.QUICK_PATH).read_text(encoding="utf-8"))
    Q.validate(good)
    for edit in ({"cuts": good["cuts"][:-1]}, {"cuts": list(reversed(good["cuts"]))}, {"platt": {"a": -1.0, "b": 0.0}},
                 {"score_rates": good["score_rates"][:9]}, {"features": list(reversed(good["features"]))}, {"lo": [0, 0]}):
        with pytest.raises(Q.QuickModelError):
            Q.validate({**good, **edit})
    with pytest.raises(Q.QuickModelError):
        Q.validate({k: v for k, v in good.items() if k != "model_str"})


def test_load_fails_closed_when_the_file_is_missing_or_garbled(tmp_path):
    with pytest.raises(Q.QuickModelError):
        Q.load(tmp_path)
    (tmp_path / "artifacts" / "quick").mkdir(parents=True)
    (tmp_path / Q.QUICK_PATH).write_text("{not json", encoding="utf-8")
    with pytest.raises(Q.QuickModelError):
        Q.load(tmp_path)
    good = json.loads((ROOT / Q.QUICK_PATH).read_text(encoding="utf-8"))
    (tmp_path / Q.QUICK_PATH).write_text(json.dumps({**good, "model_str": "garbage"}), encoding="utf-8")
    with pytest.raises(Q.QuickModelError):
        Q.load(tmp_path)


def test_the_four_inputs_are_the_ones_the_user_chose():
    assert [i["key"] for i in Q.INPUTS] == ["current_ratio", "debt_to_equity", "net_profit_margin", "cash_earnings_ratio"]
    assert [i["higher"] for i in Q.INPUTS] == ["safer", "riskier", "safer", "safer"]
    assert [i["proxy"] for i in Q.INPUTS] == [False, False, False, True]
    assert [a for _n, a in Q.FEATURES] == ["Attr4", "Attr8", "Attr23", "Attr26"]
