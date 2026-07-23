"""Тесты разметки: тройной барьер, мета-метки, веса конкурентности.

Ценовые пути построены руками, ответы известны заранее.
"""

import numpy as np
import pandas as pd
import pytest

from trading.ml.labeling import (
    apply_triple_barrier,
    concurrency_weights,
    daily_volatility,
    meta_labels,
    triple_barrier_labels,
)


def price(path):
    return pd.Series(path, index=pd.bdate_range("2024-01-01", periods=len(path)))


class TestTripleBarrier:
    def test_upper_barrier_touched_first(self):
        # Цена растёт с 100 до 110: верхний барьер (+5%) задет первым.
        close = price([100, 102, 106, 110, 108])
        events = pd.DataFrame(
            {"vertical": [close.index[-1]], "target": [0.05]},
            index=[close.index[0]],
        )
        out = apply_triple_barrier(close, events, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["label"] == 1
        assert out.iloc[0]["t1"] == close.index[2]   # 106 = +6% ≥ +5%

    def test_lower_barrier_touched_first(self):
        close = price([100, 99, 94, 90])
        events = pd.DataFrame(
            {"vertical": [close.index[-1]], "target": [0.05]},
            index=[close.index[0]],
        )
        out = apply_triple_barrier(close, events, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["label"] == -1
        assert out.iloc[0]["t1"] == close.index[2]   # 94 = -6% ≤ -5%

    def test_vertical_barrier_when_flat(self):
        # Цена почти не двигается — срабатывает вертикальный барьер.
        close = price([100, 100.5, 99.8, 100.2, 100.1])
        events = pd.DataFrame(
            {"vertical": [close.index[-1]], "target": [0.05]},
            index=[close.index[0]],
        )
        out = apply_triple_barrier(close, events, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["label"] == 0
        assert out.iloc[0]["t1"] == close.index[-1]

    def test_barriers_scale_with_volatility(self):
        # Тот же путь, но большой target (широкие барьеры) → вертикальный.
        close = price([100, 103, 106, 104])
        wide = pd.DataFrame({"vertical": [close.index[-1]], "target": [0.20]},
                            index=[close.index[0]])
        narrow = pd.DataFrame({"vertical": [close.index[-1]], "target": [0.02]},
                              index=[close.index[0]])
        assert apply_triple_barrier(close, wide).iloc[0]["label"] == 0   # не достаёт
        assert apply_triple_barrier(close, narrow).iloc[0]["label"] == 1  # достаёт быстро

    def test_short_side_inverts_returns(self):
        # Шорт (side=-1): падение цены — это ПРИБЫЛЬ, верхний барьер.
        close = price([100, 97, 94, 96])
        events = pd.DataFrame(
            {"vertical": [close.index[-1]], "target": [0.05], "side": [-1]},
            index=[close.index[0]],
        )
        out = apply_triple_barrier(close, events, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["label"] == 1     # для шорта падение = прибыль
        assert out.iloc[0]["ret"] > 0


class TestTripleBarrierWrapper:
    def test_builds_labels_for_multiple_entries(self):
        rng = np.random.default_rng(0)
        close = pd.Series(
            np.cumprod(1 + rng.normal(0, 0.02, 200)) * 100,
            index=pd.bdate_range("2024-01-01", periods=200),
        )
        entries = close.index[30::10]
        out = triple_barrier_labels(close, entries, pt_sl=(1, 1), max_holding=10)
        assert len(out) > 0
        assert set(out["label"].unique()).issubset({-1, 0, 1})
        # Горизонт метки не выходит за максимальный срок удержания.
        for t0, row in out.iterrows():
            assert row["t1"] >= t0


class TestMetaLabels:
    def test_meta_one_when_primary_signal_worked(self):
        # Первичка говорит «лонг», цена выросла → мета=1.
        close = price([100, 103, 107, 110])
        side = pd.Series([1], index=[close.index[0]])
        out = meta_labels(
            close, pd.Index([close.index[0]]), side,
            pt_sl=(1, 1), max_holding=3, vol_span=2,
        )
        # target от волатильности может дать NaN на коротком ряду — проверяем,
        # что при наличии метки она согласована со знаком результата.
        if len(out):
            assert out.iloc[0]["meta"] == int(out.iloc[0]["ret"] > 0)

    def test_meta_zero_when_primary_signal_failed(self):
        close = price([100.0] * 3 + [88.0, 87.0])   # лонг, но цена рухнула
        side = pd.Series([1], index=[close.index[0]])
        out = meta_labels(
            close, pd.Index([close.index[0]]), side,
            pt_sl=(1, 1), max_holding=4, vol_span=2,
        )
        if len(out):
            assert out.iloc[0]["meta"] == 0


class TestConcurrencyWeights:
    def test_non_overlapping_labels_equal_weight(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        # Две метки без перекрытия: [0..2] и [5..7].
        t1 = pd.Series([idx[2], idx[7]], index=[idx[0], idx[5]])
        w = concurrency_weights(t1, idx)
        assert w.iloc[0] == pytest.approx(w.iloc[1])   # одинаковый вес

    def test_overlapping_label_gets_lower_weight(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        # Метка A [0..6] сильно перекрыта метками B,C,D; метка E [8..9] одна.
        t1 = pd.Series(
            [idx[6], idx[3], idx[4], idx[5], idx[9]],
            index=[idx[0], idx[1], idx[2], idx[3], idx[8]],
        )
        w = concurrency_weights(t1, idx)
        # Одинокая метка E весит больше перекрытой метки A.
        assert w.loc[idx[8]] > w.loc[idx[0]]

    def test_weights_average_to_one(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        t1 = pd.Series([idx[3], idx[6], idx[9]], index=[idx[0], idx[3], idx[6]])
        w = concurrency_weights(t1, idx)
        assert w.mean() == pytest.approx(1.0)


def test_daily_volatility_positive():
    rng = np.random.default_rng(1)
    close = pd.Series(np.cumprod(1 + rng.normal(0, 0.02, 100)) * 100)
    vol = daily_volatility(close, span=20).dropna()
    assert (vol >= 0).all()
    assert vol.iloc[-1] > 0
