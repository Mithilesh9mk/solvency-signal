"""Phase 8: Streamlit smoke tests with streamlit.testing. Demo firms only, no uploads, no test split."""

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
    parts += [str(m.value) for m in at.metric]
    return "\n".join(parts)


def test_default_demo_firm_is_scored_and_page_has_required_text(guard):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Score a firm", "How it was checked", "What drives the model", "Limitations and data"]
    bands = [m.value for m in at.metric if m.label == "Risk band"]
    assert bands and bands[0] in ("Low", "Watch", "High")
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
        seen.add([m.value for m in at.metric if m.label == "Risk band"][0])
    assert len(seen) >= 2


def test_upload_mode_shows_policy_text_and_no_result_without_a_file():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio[0].set_value("Upload CSV").run()
    assert not at.exception
    text = page_text(at)
    assert "Attr21 blank is not scored" in text or "Attr21 blank" in text
    assert not [m for m in at.metric if m.label == "Risk band"]


def test_app_stops_with_an_error_when_frozen_files_do_not_verify(monkeypatch):
    st.cache_resource.clear()

    def bad(root=A.ROOT):
        raise A.FrozenArtifactError("frozen artifact hash mismatch, scoring is disabled: ['models/lightgbm.joblib']")

    monkeypatch.setattr(A, "verify_frozen", bad)
    at = AppTest.from_file(APP, default_timeout=120).run()
    st.cache_resource.clear()
    assert at.error and "do not match" in at.error[0].value
    assert not [m for m in at.metric if m.label == "Risk band"]
