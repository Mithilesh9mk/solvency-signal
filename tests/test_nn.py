"""Tests for src.nn.TremorNet on synthetic data."""

import numpy as np
import pandas as pd

from src.nn import TremorNet


def _data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    X.iloc[::17, 3] = np.nan
    y = (X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.5, size=n) > 1.2).astype(int)
    return X, y


def test_learns_signal_and_outputs_probabilities():
    X, y = _data()
    m = TremorNet(max_epochs=60, patience=10).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(axis=1), 1)
    assert p[y == 1, 1].mean() > p[y == 0, 1].mean() + 0.2


def test_early_stopping_restores_best_epoch():
    X, y = _data()
    m = TremorNet(max_epochs=200, patience=5).fit(X, y)
    assert 1 <= m.best_epoch_ <= len(m.history_)
    assert m.history_[m.best_epoch_ - 1] == min(m.history_)
