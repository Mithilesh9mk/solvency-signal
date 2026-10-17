"""Tests for src.calibrate: output range, monotonicity, ordered bands, saved artifacts."""

import numpy as np
import pytest

from src import calibrate as C


def _scores(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * z - 3.0)))).astype(int)
    s = 1 / (1 + np.exp(-(3.0 * z - 2.0)))  # deliberately miscalibrated
    return s, y


@pytest.mark.parametrize("method", C.METHODS)
def test_calibrator_output_in_unit_interval(method):
    s, y = _scores()
    p = C.make_calibrator(method).fit(s, y).transform(np.concatenate([s, [0.0, 1.0, 1e-9]]))
    assert np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p))


@pytest.mark.parametrize("method", C.METHODS)
def test_calibrator_mapping_is_monotone(method):
    s, y = _scores()
    grid = np.linspace(0, 1, 2001)
    p = C.make_calibrator(method).fit(s, y).transform(grid)
    assert np.all(np.diff(p) >= -1e-12)


def test_platt_improves_log_loss_on_miscalibrated_scores():
    s, y = _scores()
    before = C.log_loss_clipped(y, s)
    after = C.log_loss_clipped(y, C.PlattCalibrator().fit(s, y).transform(s))
    assert after < before


@pytest.mark.parametrize("method", C.METHODS)
def test_calibrator_roundtrips_through_dict(method):
    s, y = _scores()
    cal = C.make_calibrator(method).fit(s, y)
    again = C.calibrator_from_dict(cal.to_dict())
    assert np.allclose(cal.transform(s), again.transform(s))


def test_band_thresholds_are_ordered_and_follow_the_rule():
    s, y = _scores()
    b = C.band_thresholds(y, s)
    assert 0 <= b["t_watch"] <= b["t_high"] <= 1.0 + 1e-9
    pts = C.pr_points(y, s)
    if b["notes"]:
        assert b["t_high"] == b["t_watch"]  # two-band fallback, single cut at the recall cut
    else:
        ok = pts["precision"] >= C.BAND_RULE["high_min_precision"]
        assert b["t_high"] == pytest.approx(pts["threshold"][ok].min())  # lowest threshold meeting the floor
    # the Watch cut is the highest threshold still reaching the recall target
    rec = pts["recall"][pts["threshold"] == b["t_watch"]][0]
    assert rec >= C.BAND_RULE["watch_min_recall"]
    higher = pts["threshold"] > b["t_watch"]
    assert np.all(pts["recall"][higher] < C.BAND_RULE["watch_min_recall"])


def test_band_rule_constants_match_decisions_log():
    assert C.BAND_RULE == {"high_min_precision": 0.60, "watch_min_recall": 0.80}


def test_bands_partition_all_firms_and_are_ordered_by_risk():
    s, y = _scores()
    b = C.band_thresholds(y, s, high_min_precision=0.5)
    rows = {r["band"]: r for r in C.band_table(y, s, b["t_watch"], b["t_high"])}
    assert sum(r["firms"] for r in rows.values()) == len(y)
    assert sum(r["bankrupt"] for r in rows.values()) == y.sum()
    rates = [rows[k]["bankrupt_rate"] for k in ("Low", "Watch", "High") if rows[k]["bankrupt_rate"] is not None]
    assert rates == sorted(rates)


def test_collision_falls_back_to_two_bands_without_forcing_watch():
    # precision floor reached at a recall well above the 0.8 target
    y = np.array([1] * 20 + [0] * 380)
    s = np.concatenate([np.linspace(0.9, 0.5, 20), np.linspace(0.45, 0.0, 380)])
    b = C.band_thresholds(y, s, high_min_precision=0.01, watch_min_recall=0.8)
    assert b["t_high"] == b["t_watch"]
    assert any("two bands" in n for n in b["notes"])
    rows = {r["band"]: r for r in C.band_table(y, s, b["t_watch"], b["t_high"])}
    assert rows["Watch"]["firms"] == 0 and rows["High"]["firms"] > 0


def test_select_method_prefers_simpler_method_on_ties():
    s, y = _scores()
    sel = C.select_method(s, y)
    assert sel["selected"] in C.METHODS
    assert C.METHODS.index(sel["selected"]) <= C.METHODS.index(sel["best_by_mean"])


def test_ece_is_zero_for_perfect_and_positive_for_biased():
    y = np.array([0, 1] * 500)
    assert C.ece(y, np.full(1000, 0.5)) == pytest.approx(0.0)
    assert C.ece(y, np.full(1000, 0.9)) == pytest.approx(0.4)


def test_cost_threshold_shifts_down_as_miss_cost_rises():
    s, y = _scores()
    cuts = [C.cost_thresholds(y, s, y, s, r)["t_empirical"] for r in C.COST_RATIOS]
    assert cuts[0] >= cuts[1] >= cuts[2]


def test_artifacts_load_without_refitting(tmp_path):
    s, y = _scores()
    cals = {"m": {meth: C.make_calibrator(meth).fit(s, y) for meth in C.METHODS}}
    bands = {"m": {**C.band_thresholds(y, cals["m"]["platt"].transform(s)), "selected_method": "platt"}}
    C.save_artifacts(tmp_path, cals, bands, {"seed": 42})
    art = C.load_artifacts(tmp_path)
    p, band = C.risk_band("m", s, art)
    assert np.allclose(p, cals["m"]["platt"].transform(s))
    assert set(band) <= set(C.BANDS)
    assert art["config"]["seed"] == 42


def test_saved_phase4_artifacts_are_consistent():
    from src.preprocess import ROOT

    d = ROOT / "artifacts" / "calibration"
    if not (d / "bands.json").exists():
        pytest.skip("run python -m scripts.run_phase4 first")
    art = C.load_artifacts(d)
    assert art["config"]["seed"] == 42 and art["config"]["test_split_read"] is False
    for model, b in art["bands"].items():
        assert 0 <= b["t_watch"] <= b["t_high"] <= 1.0 + 1e-9
        assert b["fit_on"] == "validation"
        assert set(art["calibrators"][model]) == set(C.METHODS)
