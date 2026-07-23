"""Тесты дробного дифференцирования на синтетике с известным ответом."""

import numpy as np
import pandas as pd
import pytest

from trading.features.transforms import (
    adf_pvalue,
    frac_diff,
    frac_diff_weights,
    min_frac_diff_order,
)


def test_weights_d0_is_identity():
    # d=0: ряд не меняется, вес один и равен 1.
    w = frac_diff_weights(0.0, 100)
    assert len(w) == 1
    assert w[0] == pytest.approx(1.0)


def test_weights_shrink_window_with_higher_threshold():
    # Больше порог — короче окно (грубее приближение, но практичнее).
    assert len(frac_diff_weights(0.4, 10000, 1e-3)) < \
        len(frac_diff_weights(0.4, 10000, 1e-5))


def test_weights_d1_are_first_difference():
    # d=1: первая разность, веса [-1, 1] (старый, новый).
    w = frac_diff_weights(1.0, 100, threshold=1e-9)
    assert w[-1] == pytest.approx(1.0)
    assert w[-2] == pytest.approx(-1.0)


def test_d1_equals_plain_difference():
    s = pd.Series([1.0, 3.0, 6.0, 10.0, 15.0])
    fd = frac_diff(s, 1.0, threshold=1e-9)
    expected = s.diff()
    # Совпадает с обычной разностью там, где она определена.
    pd.testing.assert_series_equal(
        fd.dropna(), expected.dropna(), check_names=False
    )


def test_d0_returns_series_unchanged():
    s = pd.Series([1.0, 2.0, 1.5, 3.0])
    fd = frac_diff(s, 0.0)
    pd.testing.assert_series_equal(fd, s, check_names=False)


def test_fractional_order_keeps_more_memory_than_d1():
    """Дробная разность коррелирует с исходным рядом сильнее, чем d=1."""
    rng = np.random.default_rng(0)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 500)) + 100)
    frac = frac_diff(walk, 0.4).dropna()
    full = frac_diff(walk, 1.0).dropna()
    common = frac.index.intersection(full.index)
    corr_frac = np.corrcoef(walk.loc[frac.index], frac)[0, 1]
    corr_full = np.corrcoef(walk.loc[full.index], full)[0, 1]
    assert abs(corr_frac) > abs(corr_full)   # дробная сохраняет больше памяти


def test_random_walk_is_nonstationary_but_diff_is_stationary():
    rng = np.random.default_rng(1)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 500)) + 100)
    assert adf_pvalue(walk) > 0.05           # случайное блуждание нестационарно
    assert adf_pvalue(walk.diff()) < 0.05    # его разность — стационарна


def test_min_order_finds_stationarity():
    rng = np.random.default_rng(2)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 800)) + 100)
    result = min_frac_diff_order(walk)
    assert result["stationary"]
    assert 0 < result["d"] <= 1
    assert result["adf_pvalue"] < 0.05


def test_invalid_order_rejected():
    with pytest.raises(ValueError, match="в \\[0, 1\\]"):
        frac_diff(pd.Series([1.0, 2.0]), 1.5)
