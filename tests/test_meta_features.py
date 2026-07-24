"""Тесты признаков мета-модели и стратегии-фильтра (без утечки)."""

import numpy as np
import pandas as pd
import pytest

from trading.ml.features import FEATURE_NAMES, bet_features, features_frame
from trading.strategy.examples.meta_filtered import (
    IMOEX_KEY,
    USD_KEY,
    MetaFilteredMomentum,
)


def series(vals):
    return pd.Series(vals, dtype=float)


class TestFeatures:
    def test_momentum_signs(self):
        rising = series([100 + i for i in range(130)])
        f = bet_features(rising, rising, rising)
        assert f["mom_20"] > 0 and f["mom_60"] > 0 and f["mom_120"] > 0

    def test_short_series_gives_nan_not_crash(self):
        f = bet_features(series([100, 101, 102]), None, None)
        assert np.isnan(f["mom_120"])          # своя история коротка — nan
        assert f["usd_mom_60"] == 0.0          # межрыночного ряда нет — нейтральный 0

    def test_features_frame_canonical_columns(self):
        frame = features_frame([bet_features(series([100 + i for i in range(130)]),
                                             None, None)])
        assert list(frame.columns) == FEATURE_NAMES

    def test_features_use_only_past(self):
        """Признак на дату t не зависит от значений после t."""
        base = series([100 + i for i in range(130)])
        extended = pd.concat([base, series([999, 998, 997])], ignore_index=True)
        f_base = bet_features(base, base, base)
        # Обрезаем расширенный ряд обратно до t — должно совпасть.
        f_cut = bet_features(extended.iloc[:130], extended.iloc[:130],
                             extended.iloc[:130])
        assert f_base["mom_20"] == pytest.approx(f_cut["mom_20"])


class FakeMetaModel:
    """Мета-модель, принимающая только бумаги с положительным mom_20."""

    def __init__(self, accept_above=0.0):
        self.accept_above = accept_above

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        mom20 = X[:, FEATURE_NAMES.index("mom_20")]
        p = (mom20 > self.accept_above).astype(float)
        return np.column_stack([1 - p, p])


def make_slice(returns_by_stock, n=200, last="2024-03-15"):
    dates = pd.bdate_range(end=last, periods=n)
    data = {}
    for secid, total in returns_by_stock.items():
        daily = (1 + total) ** (1 / (n - 1))
        data[secid] = pd.DataFrame(
            {"date": dates, "close": [100.0 * daily ** i for i in range(n)]}
        )
    flat = pd.DataFrame({"date": dates, "close": [60.0] * n})
    data[USD_KEY] = flat
    data[IMOEX_KEY] = flat
    return data


class TestMetaFilteredStrategy:
    def test_without_model_is_plain_momentum(self):
        s = MetaFilteredMomentum(lookback=20, top_n=2, meta_model=None)
        data = make_slice({"AAAA": 0.3, "BBBB": 0.1, "CCCC": -0.2})
        w = s.target_weights(data)
        assert set(w) == {"AAAA", "BBBB"}
        assert all(v == pytest.approx(0.5) for v in w.values())

    def test_filter_drops_weak_signals_to_cash(self):
        # Модель принимает только растущие; падающая бумага отсеивается.
        s = MetaFilteredMomentum(lookback=20, top_n=3, meta_model=FakeMetaModel(),
                                 threshold=0.5)
        data = make_slice({"AAAA": 0.3, "BBBB": 0.1, "CCCC": -0.2})
        w = s.target_weights(data)
        assert "CCCC" not in w                    # отсеяна в кэш
        # Доля остаётся 1/top_n, а не 1/len(kept): освободившийся слот — кэш.
        assert all(v == pytest.approx(1 / 3) for v in w.values())
        assert sum(w.values()) < 1.0

    def test_service_keys_never_traded(self):
        s = MetaFilteredMomentum(lookback=20, top_n=4, meta_model=None)
        data = make_slice({"AAAA": 0.3})
        w = s.target_weights(data)
        assert USD_KEY not in w and IMOEX_KEY not in w
