"""Tests for src.train baselines: determinism and no test-split access."""

import numpy as np
import pytest

from src import train
from src.preprocess import PROCESSED

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "polish_5year_train.csv").exists(), reason="run src.preprocess first"
)


@pytest.mark.parametrize("name", list(train.BASELINES))
def test_baseline_is_deterministic(name):
    X, y = train.load_split("train")
    X, y = X.iloc[:1500], y.iloc[:1500]
    make = train.BASELINES[name]
    a = make().fit(X, y).predict_proba(X)[:, 1]
    b = make().fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)


def test_train_never_reads_test_split(monkeypatch, tmp_path):
    seen = []
    real = train.pd.read_csv
    monkeypatch.setattr(train.pd, "read_csv", lambda p, *a, **k: seen.append(str(p)) or real(p, *a, **k))
    monkeypatch.setattr(train, "cross_validate", lambda *a, **k: {})
    monkeypatch.setattr(train, "bootstrap_ci", lambda *a, **k: {})
    monkeypatch.setattr(train, "MODELS", tmp_path / "models")
    monkeypatch.setattr(train, "REPORTS", tmp_path)
    train.main()
    assert seen and not any("test" in p.rsplit("polish_5year_", 1)[-1] for p in seen)
