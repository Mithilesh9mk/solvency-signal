"""Tests for src.preprocess on the Polish 5year data."""

import hashlib

import numpy as np
import pytest

from src import preprocess as pp
from src.polish_features import FEATURES, FEATURE_NAMES, TARGET

pytestmark = pytest.mark.skipif(not pp.RAW_ZIP.exists(), reason="raw zip not present")


@pytest.fixture(scope="module")
def raw():
    return pp.load_polish(5)


@pytest.fixture(scope="module")
def parts(raw):
    return pp.split(pp.clean(raw))


def test_feature_map_is_complete():
    assert [a for a, *_ in FEATURES] == [f"Attr{i}" for i in range(1, 65)]
    assert len(set(FEATURE_NAMES)) == 64


def test_raw_shape_matches_uci(raw):
    assert raw.shape == (5910, 65)
    assert raw[TARGET].sum() == 410


def test_column_order_identity(raw):
    # assets_to_liabilities is the reciprocal of liabilities_to_assets. Two
    # rows break it: one rounding case and one probable data error (row 1992).
    m = raw[["assets_to_liabilities", "liabilities_to_assets"]].dropna()
    ok = np.isclose(m.iloc[:, 0] * m.iloc[:, 1], 1, rtol=1e-3)
    assert ok.mean() > 0.999


def test_raw_zip_not_modified():
    before = hashlib.sha256(pp.RAW_ZIP.read_bytes()).hexdigest()
    pp.load_polish(5)
    assert hashlib.sha256(pp.RAW_ZIP.read_bytes()).hexdigest() == before


def test_split_is_disjoint_stratified_and_complete(parts):
    idx = [set(p.index) for p in parts.values()]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert sum(len(p) for p in parts.values()) == 5850
    rates = [p[TARGET].mean() for p in parts.values()]
    assert max(rates) - min(rates) < 0.01


def test_split_is_reproducible(raw):
    a = pp.split(pp.clean(raw))
    b = pp.split(pp.clean(raw))
    assert all(a[k].index.equals(b[k].index) for k in a)


def test_no_duplicates_across_splits(parts):
    cols = FEATURE_NAMES + [TARGET]
    train = set(map(tuple, parts["train"][cols].fillna(-999).values))
    test = set(map(tuple, parts["test"][cols].fillna(-999).values))
    assert not train & test


def test_preprocessor_output_is_finite(parts):
    pre = pp.build_preprocessor().fit(parts["train"][FEATURE_NAMES])
    out = pre.transform(parts["test"][FEATURE_NAMES])
    assert np.isfinite(out).all()
    assert out.shape[1] > 64  # missing indicator columns appended
