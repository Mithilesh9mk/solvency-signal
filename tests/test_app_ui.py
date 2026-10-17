"""Phase 8: Streamlit smoke tests with streamlit.testing. Demo firms only, no uploads, no test split."""

import re

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import app_logic as A
from src.preprocess import ROOT
from tests.test_no_test_access import guard  # noqa: F401  (pytest fixture)

pytestmark = pytest.mark.skipif(
    not ((ROOT / "models" / "lightgbm.joblib").exists() and (ROOT / "app" / "demo_firms.json").exists()
         and (ROOT / "reports" / "phase8_app_numbers.json").exists()),
    reason="needs the committed model, demo firms and generated numbers",
)
APP = str(ROOT / "app" / "streamlit_app.py")


def page_text(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info]
    parts += [e.value for e in at.warning] + [e.value for e in at.error] + [e.value for e in at.subheader]
    parts += [e.label for e in at.expander]
    return "\n".join(parts)


def verdicts(at: AppTest, kind: str) -> list:
    """(band, score) of each result card of this kind. The cards carry data attributes for exactly this."""
    html = "\n".join(e.value for e in at.markdown)
    return re.findall(rf'data-kind="{kind}" data-band="(\w+)" data-score="(\d*)"', html)


def test_default_demo_firm_is_scored_and_page_has_required_text(guard):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Quick check: 4 ratios", "Full model: 64 ratios", "Evidence", "Limits and data"]
    bands = verdicts(at, "full")
    assert bands and bands[0][0] in ("Low", "Watch", "High")
    text = page_text(at)
    assert "is not written to disk by this app" in text
    assert "flattered" in text
    assert "accura" not in text.lower()
    assert "CC BY 4.0" in text
    assert guard == []


def test_each_demo_firm_scores_without_error():
    at = AppTest.from_file(APP, default_timeout=120).run()
    options = at.selectbox[0].options
    assert len(options) == 3
    seen = set()
    for opt in options:
        at.selectbox[0].select(opt).run()
        assert not at.exception
        seen.add(verdicts(at, "full")[0][0])
    assert len(seen) >= 2


def test_upload_mode_shows_policy_text_and_no_result_without_a_file():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio[0].set_value("Upload CSV").run()
    assert not at.exception
    text = page_text(at)
    assert "Attr21 blank" in text and "Your results appear here" in text
    assert not verdicts(at, "full")


def test_app_stops_with_an_error_when_frozen_files_do_not_verify(monkeypatch):
    st.cache_resource.clear()

    def bad(root=A.ROOT):
        raise A.FrozenArtifactError("frozen artifact hash mismatch, scoring is disabled: ['models/lightgbm.joblib']")

    monkeypatch.setattr(A, "verify_frozen", bad)
    at = AppTest.from_file(APP, default_timeout=120).run()
    st.cache_resource.clear()
    assert at.error and "do not match" in at.error[0].value
    assert not verdicts(at, "full") and not verdicts(at, "quick")


def test_page_states_the_refused_blanks_and_the_rule_numbers_come_from_files():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = page_text(at)
    rule = A.load_rule()
    names = A.refused_blank_names(rule["blocked"])
    assert f"Firms with {names} blank are not scored" in text
    assert "Firms this app scores under all its blank rules" in text
    at.segmented_control[0].set_value("What drives the model").run()
    assert not at.exception
    text = page_text(at)
    for reason in rule["blocked"].values():
        assert reason in text
    at.radio[0].set_value("Upload CSV").run()
    assert f"or {names} blank" in page_text(at)


def test_app_stops_with_an_error_when_the_blank_rule_cannot_be_read(monkeypatch):
    st.cache_resource.clear()

    def bad(root=A.ROOT):
        raise A.BlankRuleError("rule file missing")

    monkeypatch.setattr(A, "load_rule", bad)
    at = AppTest.from_file(APP, default_timeout=120).run()
    st.cache_resource.clear()
    assert at.error and "blank-attribute rule" in at.error[0].value
    assert not verdicts(at, "full") and not verdicts(at, "quick")


def test_quick_check_tab_scores_four_ratios_and_moves_the_right_way():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [n.label.split(" (")[0] for n in at.number_input][:4] == ["Current ratio", "Debt-to-equity ratio", "Net profit margin",
                                                                      "Cash earnings ratio"]
    (band, score), = verdicts(at, "quick")
    assert band in ("Low", "Watch", "Elevated", "High") and 1 <= int(score) <= 10
    text = page_text(at)
    assert "proxy" in text.lower() and "Not lending advice" in text and "much less informative than all 64" in text
    safe_score = int(score)
    for key, val in (("q_current_ratio", 0.2), ("q_debt_to_equity", 8.0), ("q_net_profit_margin", -0.5), ("q_cash_earnings_ratio", -0.5)):
        at.number_input(key=key).set_value(val)
    at.run()
    assert not at.exception
    (risky_band, risky_score), = verdicts(at, "quick")
    assert int(risky_score) > safe_score and risky_band in ("Elevated", "High")


def test_quick_check_is_switched_off_but_the_rest_of_the_app_works_when_its_file_is_unreadable(monkeypatch):
    from src import quick as Q
    st.cache_resource.clear()

    def bad(root=Q.ROOT):
        raise Q.QuickModelError("quick model file is not readable: quick_model.json")

    monkeypatch.setattr(Q, "load", bad)
    at = AppTest.from_file(APP, default_timeout=120).run()
    st.cache_resource.clear()
    assert any("quick check could not be loaded" in e.value for e in at.error)
    assert not verdicts(at, "quick")
    assert verdicts(at, "full")


def test_quick_check_says_when_values_are_the_example_and_resets_them():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert "Result for the example values" in page_text(at)
    at.number_input(key="q_current_ratio").set_value(2.5).run()
    assert not at.exception and "Result for your values" in page_text(at)
    at.button(key="q_reset").click().run()
    assert not at.exception and at.number_input(key="q_current_ratio").value == pytest.approx(1.5)
    assert "Result for the example values" in page_text(at)


def test_quick_check_flags_an_outside_range_entry_inline():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.number_input(key="q_current_ratio").set_value(500.0).run()
    assert not at.exception
    text = page_text(at)
    assert "Outside the usual range" in text and "Treat this score with care" in text
