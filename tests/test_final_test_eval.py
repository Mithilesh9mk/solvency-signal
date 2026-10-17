"""Phase 5 one-shot enforcement. Nothing here reads the real test split."""

import json
import re
import shutil

import pandas as pd
import pytest

from scripts import final_test_eval as F
from src.polish_features import FEATURE_NAMES, TARGET
from src.preprocess import PROCESSED, ROOT
from tests.test_no_test_access import guard  # noqa: F401  (pytest fixture)

needs_frozen = pytest.mark.skipif(
    not (F.PROTOCOL.exists() and (ROOT / "models" / "lightgbm.joblib").exists()
         and (PROCESSED / "polish_5year_val.csv").exists()),
    reason="needs the frozen protocol, saved models and processed data",
)


def tmp_paths(tmp_path, **kw):
    return F.Paths(lock=tmp_path / "phase5.lock", results=tmp_path / "results.json",
                   report_md=tmp_path / "report.md", figure=tmp_path / "rel.png", **kw)


@needs_frozen
def test_second_run_is_refused_when_results_exist(guard, tmp_path):
    paths = tmp_paths(tmp_path)
    paths.results.write_text("{}")
    with pytest.raises(RuntimeError, match="already evaluated"):
        F.run(paths, split="test")
    assert guard == [] and not paths.lock.exists()


@needs_frozen
def test_second_run_is_refused_when_only_the_lock_exists(guard, tmp_path):
    paths = tmp_paths(tmp_path)
    paths.lock.write_text('{"status": "started"}')
    with pytest.raises(RuntimeError, match="previous attempt"):
        F.run(paths, split="test")
    assert guard == []


@needs_frozen
def test_hash_mismatch_aborts_before_any_read(guard, tmp_path):
    proto = json.loads(F.PROTOCOL.read_text())
    proto["files"]["artifacts/calibration/bands.json"]["sha256"] = "0" * 64
    bad = tmp_path / "protocol.json"
    bad.write_text(json.dumps(proto))
    paths = tmp_paths(tmp_path, protocol=bad)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        F.run(paths, split="test")
    assert guard == [] and not paths.lock.exists() and not paths.results.exists()


@needs_frozen
def test_dry_run_on_validation_writes_only_to_temp(guard, tmp_path):
    paths = tmp_paths(tmp_path)
    res = F.run(paths, split="val", dry=True, n_boot=20)
    assert guard == []
    assert res["split"] == "val" and res["dry_run"]
    assert paths.results.exists() and paths.report_md.exists() and paths.figure.exists()
    assert not paths.lock.exists()
    assert set(res["models"]) == {"lightgbm", "logistic_regression", "neural_net"}
    assert res["models"]["lightgbm"]["role"] == "headline"


@needs_frozen
def test_lock_is_written_before_the_read_and_second_run_then_fails(guard, tmp_path, monkeypatch):
    """Full non-dry path on a stand-in file that is not the test split."""
    df = pd.read_csv(PROCESSED / "polish_5year_val.csv")[FEATURE_NAMES + [TARGET]]
    stand_in = tmp_path / "holdout.csv"
    df.to_csv(stand_in, index=False)
    paths = tmp_paths(tmp_path, test_file=stand_in)
    seen = {}
    real_read = F.read_split

    def spy(split, p):
        seen["lock_present_at_read"] = p.lock.exists()
        return real_read(split, p)

    monkeypatch.setattr(F, "read_split", spy)
    res = F.run(paths, split="test", dry=False, n_boot=20)
    assert seen["lock_present_at_read"]
    lock = json.loads(paths.lock.read_text())
    assert lock["status"] == "complete" and lock["test_sha256"] == res["data_sha256"]
    assert lock["results_sha256"] == F.sha256_file(paths.results)
    with pytest.raises(RuntimeError, match="already evaluated"):
        F.run(paths, split="test", dry=False, n_boot=20)
    assert guard == []


@pytest.mark.skipif(not F.RESULTS.exists(), reason="Phase 5 has not been run yet")
def test_real_repo_refuses_a_rerun(guard):
    with pytest.raises(RuntimeError):
        F.run(F.Paths(), split="test")
    assert guard == []
    assert F.LOCK.exists() and json.loads(F.LOCK.read_text())["status"] == "complete"


def test_only_the_final_script_can_read_the_test_split():
    sources = [p for d in ("src", "scripts") for p in (ROOT / d).rglob("*.py") if p.name != "final_test_eval.py"]
    assert sources
    allowed_pattern_files = {"preprocess.py", "train.py", "run_phase4.py", "update_state.py"}
    seen = set()
    for p in sources:
        text = p.read_text(encoding="utf-8")
        rel = p.relative_to(ROOT).as_posix()
        assert not re.search(r"allow_test\s*=\s*True", text), rel
        assert "final_test_eval" not in text, rel
        for m in re.finditer(r"load_split\(\s*([\"'])(\w+)\1", text):
            assert m.group(2) in ("train", "val"), rel
        for m in re.finditer(r"load_split\(\s*([A-Za-z_])", text):
            assert p.name == "train.py", f"load_split called with a variable in {rel}"
        for line in text.splitlines():
            if "polish_5year_test" in line:
                assert p.name == "update_state.py" and not re.search(r"read_|open\(|load_", line), (rel, line)
        if "polish_5year_{" in text:
            seen.add(p.name)
    assert seen <= allowed_pattern_files, seen - allowed_pattern_files


def test_script_uses_stated_hash_and_lock_protocol():
    text = (ROOT / "scripts" / "final_test_eval.py").read_text(encoding="utf-8")
    assert text.index("paths.lock.write_text") < text.index("read_split(split, paths)")
    assert "refuse_if_used(paths)" in text and "verify_files(proto, paths)" in text
