"""Phase 4 must never touch the test split. Any open, stat or read of its file fails the test."""

import builtins
import io
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import train
from src.preprocess import PROCESSED, ROOT

LOCKED = "polish_5year_test"
needs_data = pytest.mark.skipif(
    not (PROCESSED / "polish_5year_train.csv").exists(), reason="run src.preprocess first"
)


@pytest.fixture
def guard(monkeypatch):
    """Raise if anything opens, stats or reads the locked test file. Records every attempt."""
    attempts = []

    def check(path):
        if LOCKED in os.fspath(path) if isinstance(path, (str, bytes, os.PathLike)) else False:
            attempts.append(os.fspath(path))
            raise AssertionError(f"test split touched: {path}")

    real_open, real_io_open, real_stat, real_read_csv = builtins.open, io.open, os.stat, pd.read_csv

    def guarded(real):
        def f(path, *a, **k):
            check(path)
            return real(path, *a, **k)
        return f

    monkeypatch.setattr(builtins, "open", guarded(real_open))
    monkeypatch.setattr(io, "open", guarded(real_io_open))
    monkeypatch.setattr(os, "stat", guarded(real_stat))
    monkeypatch.setattr(pd, "read_csv", guarded(real_read_csv))
    return attempts


def test_guard_itself_catches_access(guard, tmp_path):
    with pytest.raises(AssertionError):
        open(tmp_path / f"{LOCKED}.csv")
    with pytest.raises(AssertionError):
        pd.read_csv(tmp_path / f"{LOCKED}.csv")
    with pytest.raises(AssertionError):
        Path(tmp_path / f"{LOCKED}.csv").exists()
    assert len(guard) == 3


def test_load_split_refuses_test_by_default():
    with pytest.raises(PermissionError):
        train.load_split("test")


@needs_data
def test_phase4_pipeline_never_touches_test_split(guard, tmp_path):
    from scripts import run_phase4

    X_tr, y_tr = train.load_split("train")
    X_va, y_va = train.load_split("val")
    X_tr, y_tr = X_tr.iloc[:1500], y_tr.iloc[:1500]
    model = train.make_logreg().fit(X_tr, y_tr)
    res = run_phase4.run(X_tr, y_tr, X_va, y_va, {"logistic_regression": model}, ["logistic_regression"],
                         reports=tmp_path / "reports", artifacts=tmp_path / "artifacts", n_boot=20)
    assert guard == []
    assert (tmp_path / "artifacts" / "bands.json").exists()
    assert res["models"]["logistic_regression"]["bands"]["t_watch"] <= res["models"]["logistic_regression"]["bands"]["t_high"]


@needs_data
def test_update_state_never_touches_test_split(guard):
    from scripts import update_state

    summary = update_state.build_summary()
    assert guard == []
    assert summary["splits"]["test"]["status"].startswith("LOCKED")
    assert set(summary["splits"]) == {"train", "val", "test"}


def test_phase4_sources_never_request_the_test_split():
    for rel in ("src/calibrate.py", "scripts/run_phase4.py", "scripts/update_state.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"allow_test\s*=\s*True", text), rel
        assert not re.search(r"load_split\(\s*[\"']test", text), rel
        assert not re.search(r"read_csv\([^)]*test", text), rel
