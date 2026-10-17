"""Tests for src.evaluate metrics."""

import numpy as np

from src.evaluate import bootstrap_ci, recall_at_fpr, score


def test_perfect_ranking():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    m = score(y, s)
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    assert m["recall_at_fpr5"] == 1.0


def test_recall_at_fpr_respects_limit():
    rng = np.random.default_rng(42)
    y = np.r_[np.zeros(1000), np.ones(50)].astype(int)
    s = rng.random(1050)
    r = recall_at_fpr(y, s, 0.05)
    assert r["actual_fpr"] <= 0.05


def test_bootstrap_is_reproducible_and_brackets_point():
    rng = np.random.default_rng(42)
    y = np.r_[np.zeros(300), np.ones(30)].astype(int)
    s = (y * 0.5 + rng.random(330)) / 1.5
    a, b = bootstrap_ci(y, s, n=200), bootstrap_ci(y, s, n=200)
    assert a == b
    lo, hi = a["roc_auc_ci"]
    assert lo <= score(y, s)["roc_auc"] <= hi
