"""Presentation helpers (src/app_ui.py). No model numbers are asserted; app-level tests are in test_app_ui.py."""

from pathlib import Path

import pytest

from src import app_ui as U

ROOT = Path(__file__).resolve().parents[1]


def test_formatters():
    assert U.pct(0.0402) == "4.0%" and U.pct(None) == "n/a"
    assert U.pct_ci(0.5, (0.25, 0.75), 0) == "50% (25% to 75%)"
    assert U.num(0.12345) == "0.123" and U.num_ci(0.5, (0.4, 0.6)) == "0.500 (0.400 to 0.600)"


def test_dynamic_text_is_escaped():
    assert "<script>" not in U.callout("err", "<script>x</script>", "a & b")
    assert "&lt;b&gt;" in U.tile("<b>k</b>", "1") and "<script>" not in U.empty_state("t", "<script>")


def test_strip_marks_exactly_one_active_cell():
    bands = ["Low", "Watch", "High"]
    html = U.strip(["Low", "Watch", "High"], 1, bands, [(b, b) for b in bands])
    assert html.count("&#9650;") == 1 and html.count("opacity:1") == 1


def test_driver_bars_scale_and_direction():
    html = U.driver_bars([{"label": "a", "value": 2.0}, {"label": "b", "value": -1.0}, {"label": "c", "value": 0.0}])
    assert "raises" in html and "lowers" in html and "width:50.0%" in html and "width:25.0%" in html


def test_driver_bars_handles_all_zero_and_empty():
    assert "width:0.0%" in U.driver_bars([{"label": "a", "value": 0.0}])
    assert U.driver_bars([]).startswith('<div class="ss-bars">')


def test_band_colours_cover_every_band_name():
    from src import quick as Q

    assert {name for _, _, name in Q.BANDS} <= set(U.BAND_COLORS) and {"Low", "Watch", "High"} <= set(U.BAND_COLORS)


def test_no_em_dashes_in_ui_source():
    text = (ROOT / "src" / "app_ui.py").read_text(encoding="utf-8")
    assert "—" not in text
