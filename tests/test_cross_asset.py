"""Тесты межрыночной стратегии: сигнал по курсу, лаг, отсутствие утечки."""

import pandas as pd
import pytest

from trading.strategy.examples.cross_asset import (
    SIGNAL_KEY,
    CrossAssetExporters,
    CrossAssetExportersShort,
)


def stock_df(last="2024-03-15", n=300):
    dates = pd.bdate_range(end=last, periods=n)
    return pd.DataFrame({"date": dates, "close": [100.0] * n})


def fx_df(closes, last="2024-03-15"):
    dates = pd.bdate_range(end=last, periods=len(closes))
    return pd.DataFrame({"date": dates, "close": closes})


def test_holds_basket_when_ruble_weakened():
    s = CrossAssetExporters(lookback=20, lag=0)
    rising = [60.0 + i * 0.1 for i in range(300)]   # рубль слабеет (USD растёт)
    data = {"LKOH": stock_df(), "ROSN": stock_df(), SIGNAL_KEY: fx_df(rising)}
    weights = s.target_weights(data)
    assert set(weights) == {"LKOH", "ROSN"}
    assert weights["LKOH"] == pytest.approx(0.5)


def test_cash_when_ruble_strengthened():
    s = CrossAssetExporters(lookback=20, lag=0)
    falling = [90.0 - i * 0.1 for i in range(300)]  # рубль крепнет (USD падает)
    data = {"LKOH": stock_df(), SIGNAL_KEY: fx_df(falling)}
    assert s.target_weights(data) == {}


def test_lag_shifts_the_window():
    """При лаге сигнал берётся из прошлого: недавний разворот игнорируется."""
    # Курс рос 280 дней, затем резко падал 20 дней.
    closes = [60.0 + i * 0.1 for i in range(280)] + [88.0 - i for i in range(20)]
    data = {"LKOH": stock_df(), SIGNAL_KEY: fx_df(closes)}
    # Без лага видим свежее падение → кэш.
    assert CrossAssetExporters(lookback=10, lag=0).target_weights(data) == {}
    # С лагом 20 дней смотрим на давний рост → держим корзину.
    assert set(CrossAssetExporters(lookback=10, lag=20).target_weights(data)) == {"LKOH"}


def test_short_variant_inverts_decision():
    s = CrossAssetExportersShort(lookback=20, lag=0)
    falling = [90.0 - i * 0.1 for i in range(300)]  # рубль крепнет
    data = {"LKOH": stock_df(), SIGNAL_KEY: fx_df(falling)}
    weights = s.target_weights(data)
    assert weights["LKOH"] == pytest.approx(-1.0)   # шорт, доля отрицательная


def test_only_tradeable_exporters_included():
    s = CrossAssetExporters(lookback=20, lag=0)
    rising = [60.0 + i * 0.1 for i in range(300)]
    stale = stock_df()
    stale = stale.iloc[:-10]   # эта бумага перестала торговаться
    data = {"LKOH": stock_df(), "ROSN": stale, SIGNAL_KEY: fx_df(rising)}
    assert set(s.target_weights(data)) == {"LKOH"}


def test_non_exporter_never_selected():
    s = CrossAssetExporters(lookback=20, lag=0)
    rising = [60.0 + i * 0.1 for i in range(300)]
    data = {"SBER": stock_df(), SIGNAL_KEY: fx_df(rising)}  # SBER не экспортёр
    assert s.target_weights(data) == {}


def test_missing_signal_means_cash():
    s = CrossAssetExporters(lookback=20, lag=0)
    assert s.target_weights({"LKOH": stock_df()}) == {}


def test_explains_itself_in_russian():
    text = CrossAssetExporters(lookback=20, lag=21).explain_ru()
    assert "рубль" in text and "экспортёров" in text and "21" in text
    assert "≤" not in text  # просто нормальная строка


def test_params_within_limit():
    assert len(CrossAssetExporters(20, 5).params()) <= 3
