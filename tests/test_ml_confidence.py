"""Тесты калиброванной модели и стратегии-гейта уверенности."""

import numpy as np
import pandas as pd
import pytest

from trading.ml.features import FEATURE_NAMES
from trading.ml.models.linear import CalibratedLogisticModel
from trading.strategy.examples.ml_confidence import MLConfidenceLong


def separable(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    f_good = y + rng.normal(0, 0.4, n)
    X = np.column_stack([f_good] + [rng.normal(0, 1, n) for _ in range(len(FEATURE_NAMES) - 1)])
    return X, y


class TestCalibratedModel:
    def test_fits_and_predicts(self):
        X, y = separable()
        m = CalibratedLogisticModel(FEATURE_NAMES)
        m.fit(X, y)
        p = m.predict_proba(X)[:, 1]
        assert ((p > 0.5).astype(int) == y).mean() > 0.75

    def test_explain_shows_calibrated_confidence(self):
        X, y = separable()
        m = CalibratedLogisticModel(FEATURE_NAMES)
        m.fit(X, y)
        text = m.explain(X[0])
        assert "калиброванная уверенность" in text
        assert "%" in text

    def test_calibration_is_reasonable(self):
        """Среди предсказаний с уверенностью ~высокой доля истинных высока."""
        X, y = separable(n=2000)
        m = CalibratedLogisticModel(FEATURE_NAMES)
        m.fit(X, y)
        p = m.predict_proba(X)[:, 1]
        high = p >= 0.8
        if high.sum() >= 20:                    # если такие вообще есть
            assert y[high].mean() > 0.6         # калибровка не абсурдна

    def test_handles_tiny_data_gracefully(self):
        X, y = separable(n=12)
        m = CalibratedLogisticModel(FEATURE_NAMES)
        m.fit(X, y)                              # не падает
        assert m.predict_proba(X).shape == (12, 2)


class FakeModel:
    """Уверенность = значение первого признака (mom_20), зажатое в [0,1]."""

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = np.clip(X[:, FEATURE_NAMES.index("mom_20")], 0, 1)
        return np.column_stack([1 - p, p])


def slice_with(mom_by_stock, n=200, last="2024-03-15"):
    dates = pd.bdate_range(end=last, periods=n)
    data = {}
    for secid, mom in mom_by_stock.items():
        # непрерывный рост: на конце mom_20 = daily^20 - 1 = mom
        daily = (1 + mom) ** (1 / 20)
        data[secid] = pd.DataFrame({"date": dates,
                                    "close": [100.0 * daily ** i for i in range(n)]})
    return data


class TestConfidenceGate:
    def test_buys_only_above_threshold(self):
        s = MLConfidenceLong(FakeModel(), threshold=0.8, max_positions=6)
        # mom_20: AAAA 0.9 (уверенность 0.9), BBBB 0.5, CCCC 0.1
        data = slice_with({"AAAA": 0.9, "BBBB": 0.5, "CCCC": 0.1})
        w = s.target_weights(data)
        assert set(w) == {"AAAA"}              # только уверенный ≥ 0.8

    def test_all_cash_when_none_confident(self):
        s = MLConfidenceLong(FakeModel(), threshold=0.8, max_positions=6)
        data = slice_with({"AAAA": 0.3, "BBBB": 0.2})
        assert s.target_weights(data) == {}    # весь портфель в кэше
        assert s.last_cash_share == pytest.approx(1.0)

    def test_weight_leaves_cash_when_few_qualify(self):
        s = MLConfidenceLong(FakeModel(), threshold=0.8, max_positions=6)
        data = slice_with({"AAAA": 0.95, "BBBB": 0.85})
        w = s.target_weights(data)
        assert set(w) == {"AAAA", "BBBB"}
        assert all(v == pytest.approx(1 / 6) for v in w.values())
        assert s.last_cash_share == pytest.approx(1 - 2 / 6)   # 4 слота в кэше

    def test_caps_at_max_positions_by_confidence(self):
        s = MLConfidenceLong(FakeModel(), threshold=0.8, max_positions=2)
        data = slice_with({"A": 0.99, "B": 0.95, "C": 0.90, "D": 0.85})
        w = s.target_weights(data)
        assert len(w) == 2
        assert set(w) == {"A", "B"}            # два самых уверенных

    def test_no_model_means_cash(self):
        s = MLConfidenceLong(None, threshold=0.8)
        assert s.target_weights(slice_with({"A": 0.9})) == {}
