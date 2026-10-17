"""Phase 8: app logic. Policy A, the blank rule, frozen-file checks, and explanations pinned to explain_firm."""

import hashlib
import io
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import app_logic as A
from src import calibrate as C
from src.preprocess import ROOT
from tests.test_no_test_access import guard  # noqa: F401  (pytest fixture)

needs_frozen = pytest.mark.skipif(
    not ((ROOT / "models" / "lightgbm.joblib").exists() and (ROOT / "app" / "demo_firms.json").exists()
         and (ROOT / "reports" / "phase8_app_numbers.json").exists()),
    reason="needs the committed model, demo firms and generated numbers",
)


@pytest.fixture(scope="module")
def frozen():
    return A.load_frozen()


def full_row(**over) -> pd.Series:
    row = pd.Series({a: 1.0 for a in A.ATTRS})
    for k, v in over.items():
        row[k] = v
    return row


# -------------------------------------------------------------------------- policy
def test_attr21_blank_is_refused_with_a_reason_and_no_score():
    out = A.assess(full_row(Attr21=np.nan))
    assert out["status"] == "refused" and "Attr21" in out["reason"] and "sales" in out["reason"]


def test_blank_limit_is_6_of_the_other_63_attributes():
    assert A.MAX_OTHER_BLANKS == 6
    six = {f"Attr{i}": np.nan for i in range(1, 7)}
    seven = {f"Attr{i}": np.nan for i in range(1, 8)}
    ok = A.assess(full_row(**six))
    assert ok["status"] == "scored" and ok["n_other_blank"] == 6
    for a in six:
        assert a in ok["warning"]
    assert "untested" in ok["warning"] or "not been tested" in ok["warning"]
    blocked = A.assess(full_row(**seven))
    assert blocked["status"] == "refused" and "limit" in blocked["reason"]


def test_no_blanks_means_no_warning_and_a_problem_blocks_scoring():
    assert A.assess(full_row())["warning"] == ""
    assert A.assess(full_row(), problem="Attr3 is not a number")["status"] == "refused"


def test_attr21_does_not_count_toward_the_other_blank_limit():
    out = A.assess(full_row(Attr21=np.nan, **{f"Attr{i}": np.nan for i in range(1, 4)}))
    assert out["n_other_blank"] == 3 and out["status"] == "refused"


# ------------------------------------------------------------------------- parsing
def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def good_frame(n=2) -> pd.DataFrame:
    return pd.DataFrame([{a: 1.5 + i for a in A.ATTRS} for i in range(n)])


def test_parse_accepts_attr_codes_any_case_and_dictionary_names():
    df = good_frame()
    out = A.parse_upload(csv_bytes(df))
    assert list(out["frame"].columns) == A.ATTRS and out["problems"] == {}
    lower = df.rename(columns={a: a.lower() for a in A.ATTRS})
    assert A.parse_upload(csv_bytes(lower))["frame"].notna().all().all()
    named = df.rename(columns=A.NAME_BY_ATTR)
    assert A.parse_upload(csv_bytes(named))["frame"].notna().all().all()


def test_parse_blank_markers_and_bad_cells():
    df = good_frame(4).astype(object)
    df.loc[0, "Attr5"] = ""
    df.loc[1, "Attr6"] = "?"
    df.loc[2, "Attr7"] = "abc"
    df.loc[3, "Attr8"] = "inf"
    out = A.parse_upload(csv_bytes(df))
    assert np.isnan(out["frame"].loc[0, "Attr5"]) and np.isnan(out["frame"].loc[1, "Attr6"])
    assert set(out["problems"]) == {2, 3}
    assert "Attr7" in out["problems"][2] and "finite" in out["problems"][3]
    res = A.assess(out["frame"].loc[2], out["problems"][2])
    assert res["status"] == "refused"


def test_parse_file_level_errors():
    with pytest.raises(ValueError, match="missing columns"):
        A.parse_upload(csv_bytes(good_frame().drop(columns=["Attr3", "Attr9"])))
    with pytest.raises(ValueError, match="at most"):
        A.parse_upload(csv_bytes(pd.concat([good_frame(1)] * (A.MAX_ROWS + 1), ignore_index=True)))
    with pytest.raises(ValueError, match="no rows"):
        A.parse_upload(csv_bytes(good_frame(1).iloc[0:0]))
    dup = good_frame(1)
    dup["attr3"] = 1.0
    with pytest.raises(ValueError, match="more than once"):
        A.parse_upload(csv_bytes(dup))
    extra = good_frame(1)
    extra["company_name"] = "x"
    assert A.parse_upload(csv_bytes(extra))["ignored_columns"] == ["company_name"]


# ----------------------------------------------------------------- frozen artifacts
@needs_frozen
def test_frozen_files_verify_and_tampering_is_detected(tmp_path):
    A.verify_frozen()
    for rel in A.FROZEN:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    shutil.copy(ROOT / "artifacts" / "phase5_protocol.json", tmp_path / "artifacts" / "phase5_protocol.json")
    A.verify_frozen(tmp_path)
    with open(tmp_path / "models" / "lightgbm.joblib", "ab") as fh:
        fh.write(b"0")
    with pytest.raises(A.FrozenArtifactError, match="mismatch"):
        A.verify_frozen(tmp_path)


def test_line_ending_tolerance_applies_to_json_text_files_only():
    text = b'{"a": 1}\r\n{"b": 2}\r\n'
    lf = text.replace(b"\r\n", b"\n")
    sha_crlf = hashlib.sha256(text).hexdigest()
    assert A.hash_matches("artifacts/calibration/bands.json", lf, sha_crlf)
    assert A.hash_matches("artifacts/calibration/bands.json", text, hashlib.sha256(lf).hexdigest())
    assert not A.hash_matches("models/lightgbm.joblib", lf, sha_crlf)
    assert not A.hash_matches("artifacts/calibration/bands.json", b"different", sha_crlf)


# ---------------------------------------------------------------- scoring and explanations
@needs_frozen
def test_demo_firms_follow_the_restricted_rule_and_reproduce_their_scores(frozen):
    demo = A.demo_firms()
    frame = A.demo_frame(demo)
    assert len(frame) == 3 and frame[A.KEY_ATTR].notna().all()
    res = A.score_frame(frozen, frame)
    assert (res["status"] == "scored").all()
    for f, (_, r) in zip(demo["firms"], res.iterrows()):
        assert f["would_be_scored"] and abs(f["model_probability"] - r["probability"]) < 1e-9
        assert f["model_band"] == r["band"]
    assert {f["role"] for f in demo["firms"]} == {"highest scoring true positive", "highest scoring false alarm",
                                                  "nearest miss below watch cut"}


@needs_frozen
def test_refused_firms_get_no_probability_and_no_band(frozen):
    frame = A.demo_frame(A.demo_firms()).iloc[[0, 1]].copy()
    frame.iloc[0, frame.columns.get_loc(A.KEY_ATTR)] = np.nan
    res = A.score_frame(frozen, frame)
    assert res.loc[0, "status"] == "refused" and np.isnan(res.loc[0, "probability"]) and res.loc[0, "band"] == ""
    assert res.loc[1, "status"] == "scored" and 0 < res.loc[1, "probability"] < 1


@needs_frozen
def test_native_explanation_equals_explain_firm(frozen):
    from src import explain as E

    frame = A.demo_frame(A.demo_firms())
    for i in range(len(frame)):
        row = frame.iloc[[i]]
        native = A.explain_native(frozen, row, top_k=6)
        ref = E.explain_firm(frozen["model"], frozen["calibrator"], row.rename(columns=A.NAME_BY_ATTR), frozen["bands"], top_k=6)
        assert native["band"] == ref["band"] and native["scale"] == ref["scale"]
        assert abs(native["probability"] - ref["probability"]) < 1e-12
        assert abs(native["base_probability"] - ref["base_probability"]) < 1e-9
        for a, b in zip(native["drivers"], ref["drivers"]):
            assert a["attr"] == b["attr"] and a["direction"] == b["direction"] and a["unverified"] == b["unverified"]
            assert abs(a["contribution"] - b["contribution"]) < 1e-9


@needs_frozen
def test_scoring_and_explaining_do_not_touch_the_test_split(guard, frozen):
    frame = A.demo_frame(A.demo_firms())
    A.score_frame(frozen, frame)
    A.explain_native(frozen, frame.iloc[[0]])
    assert guard == []


# ------------------------------------------------------------------ generated numbers
@needs_frozen
def test_numbers_are_consistent_with_earlier_phases():
    n = A.load_json("reports/phase8_app_numbers.json")
    phase4 = A.load_json("reports/phase4_calibration.json")["models"]["lightgbm"]["val"]["platt"]
    assert abs(n["numbers"]["val"]["all"]["pr_auc"] - phase4["pr_auc"]) < 1e-9
    for k in ("val", "train_oof"):
        d = n["numbers"][k]
        assert d["present"]["firms"] + d["attr21_missing_firms"] == d["all"]["firms"]
        assert d["present"]["bankrupt"] + d["attr21_missing_bankrupt"] == d["all"]["bankrupt"]
        assert sum(r["firms"] for r in d["present"]["band_table"]) == d["present"]["firms"]
    assert n["blank_limit"] == A.MAX_OTHER_BLANKS


@needs_frozen
def test_generated_sentences_use_the_generated_numbers():
    n = A.load_json("reports/phase8_app_numbers.json")
    s = A.flattered_sentence(n)
    assert str(n["numbers"]["train_oof"]["attr21_missing_firms"]) in s and "flattered" in s
    f = A.floor_statement(n)
    assert f"{n['numbers']['val']['present']['high']['precision']:.3f}" in f and f"{n['numbers']['train_oof']['present']['high']['precision']:.3f}" in f and "0.60" in f and "not changed" in f
    assert len(A.band_history(n, "High")) == 2
    eff = n["blank_effects"]
    assert "Attr27" in eff and "Attr37" in eff and "Attr21" not in eff
    lines = A.blank_history(n, ["Attr27", "Attr3"])
    assert len(lines) == 1 and lines[0].startswith("Attr27 blank") and "calibrated log-odds" in lines[0]


# ------------------------------------------------------------------- app source rules
APP_TEXT = lambda: (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8") + (ROOT / "src" / "app_logic.py").read_text(encoding="utf-8")  # noqa: E731


def test_app_text_rules():
    text = APP_TEXT()
    low = text.lower()
    assert "accura" not in low
    assert "—" not in text
    assert "is not written to disk by this app" in text
    assert "nothing stored" not in low and "nothing is stored" not in low
    for needle in ("CC BY 4.0", "10.24432/C5F600", "Tomczak", "not lending advice", "random split", "UNVERIFIED"):
        assert needle.lower() in low, needle


def test_app_never_references_the_test_split_file():
    for rel in ("app/streamlit_app.py", "src/app_logic.py", "scripts/run_phase8_prep.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "polish_5year" not in text and "allow_test" not in text, rel


def test_deploy_files():
    import lightgbm
    import sklearn
    import streamlit

    req = (ROOT / "app" / "requirements.txt").read_text().split()
    assert f"streamlit=={streamlit.__version__}" in req and f"lightgbm=={lightgbm.__version__}" in req
    assert f"scikit-learn=={sklearn.__version__}" in req
    assert not any(r.startswith(("torch", "shap", "numba")) for r in req)
    assert "*.joblib binary" in (ROOT / ".gitattributes").read_text()
    assert "!models/lightgbm.joblib" in (ROOT / ".gitignore").read_text()


# ------------------------------------------------------------- Phase 9: refused blanks
needs_rule = pytest.mark.skipif(not (ROOT / "reports" / "phase9_blank_rule.json").exists(), reason="needs the Phase 9 rule file")


@needs_rule
def test_blocked_set_comes_from_the_rule_file_and_matches_the_criteria():
    rule = A.load_json("reports/phase9_blank_rule.json")
    assert rule["applied"] and rule["ceiling_crossed"] and rule["ceiling_overridden"]
    assert rule["count_limit"] == A.MAX_OTHER_BLANKS == 6 and not rule["count_limit_condition_met"]
    meets = [a for a, c in rule["candidates"].items() if c["meets_all"]]
    assert rule["blocked"] == rule["meets_criteria"] == meets
    assert set(A.blocked_blanks()) == set(meets) == {"Attr24", "Attr27"}


@needs_rule
def test_refused_blanks_are_not_scored_and_the_reason_says_why():
    b = A.blocked_blanks()
    out27 = A.assess(full_row(Attr27=np.nan))
    assert out27["status"] == "refused" and out27["reason"].startswith("Attr27")
    assert "undefined when financial expenses are zero" in out27["reason"] and "UNVERIFIED" in out27["reason"]
    assert "higher" in out27["reason"] and "not the firm's finances" in out27["reason"]
    out24 = A.assess(full_row(Attr24=np.nan))
    assert out24["status"] == "refused" and "lower" in out24["reason"] and "UNVERIFIED" not in out24["reason"].split("Dictionary note")[0]
    both = A.assess(full_row(Attr24=np.nan, Attr27=np.nan))
    assert both["status"] == "refused" and "Attr24" in both["reason"] and "Attr27" in both["reason"]
    assert A.assess(full_row(Attr27=np.nan), blocked={})["status"] == "scored"      # explicit bypass for tests only
    assert A.assess(full_row(Attr3=np.nan))["status"] == "scored"                    # other blanks keep the warning
    assert A.assess(full_row(Attr21=np.nan, Attr27=np.nan))["reason"].startswith("Attr21")
    assert set(b) == {"Attr24", "Attr27"}


@needs_rule
def test_blocked_blanks_do_not_count_against_nothing_when_present_and_limit_is_unchanged():
    seven = {f"Attr{i}": np.nan for i in range(1, 8)}
    assert A.assess(full_row(**seven))["status"] == "refused" and "limit" in A.assess(full_row(**seven))["reason"]


def test_rule_file_missing_or_malformed_fails_closed(tmp_path):
    with pytest.raises(A.BlankRuleError):
        A.load_rule(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "phase9_blank_rule.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(A.BlankRuleError):
        A.load_rule(tmp_path)
    (tmp_path / "reports" / "phase9_blank_rule.json").write_text(json.dumps({"applied": True, "blocked": ["Attr27"], "blocked_detail": {}}), encoding="utf-8")
    with pytest.raises(A.BlankRuleError):
        A.load_rule(tmp_path)
    (tmp_path / "reports" / "phase9_blank_rule.json").write_text(json.dumps({"applied": False, "blocked": ["Attr27"]}), encoding="utf-8")
    assert A.load_rule(tmp_path)["blocked"] == {}                                     # not applied means nothing refused


@needs_frozen
@needs_rule
def test_demo_firms_are_swapped_not_exempted(frozen):
    from scripts import run_phase8_prep as P
    from src.train import load_split

    demo = A.demo_firms()
    blocked = A.blocked_blanks()
    frame = A.demo_frame(demo)
    for a in blocked:
        assert frame[a].notna().all(), f"a demo firm has {a} blank"
    X_va, y_va = load_split("val")
    p = frozen["calibrator"].transform(frozen["model"].predict_proba(X_va)[:, 1])
    again = P.pick_demo(X_va, y_va, p, frozen["bands"], blocked)
    assert [f["val_row"] for f in demo["firms"]] == list(again.values())
    old = P.pick_demo(X_va, y_va, p, frozen["bands"], {})                           # rule without the refusals
    refused_old = [r for r in old.values() if r not in again.values()]
    assert refused_old, "the Phase 8 pick should differ now"
    names = {f[1]: f[0] for f in A.FEATURES}
    for r in refused_old:
        row = X_va.iloc[r].rename(index=names)
        assert A.assess(row, blocked=blocked)["status"] == "refused"
