"""Тесты пересборки таймфреймов: OHLCV агрегируется правильно."""

import pandas as pd
import pytest

from trading.data.resample import resample_candles


def daily(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "value": [1e6] * n,
        "volume": [1000] * n,
    })


def test_weekly_ohlc_aggregation():
    # Одна полная торговая неделя (пн–пт), 5 дней.
    df = daily(5, "2024-01-01")  # 2024-01-01 — понедельник
    weekly = resample_candles(df, "week")
    assert len(weekly) == 1
    bar = weekly.iloc[0]
    assert bar["open"] == 100.0          # открытие понедельника
    assert bar["close"] == 104.5         # закрытие пятницы
    assert bar["high"] == 105.0          # максимум за неделю
    assert bar["low"] == 99.0            # минимум за неделю
    assert bar["volume"] == 5000         # сумма объёмов
    assert bar["value"] == pytest.approx(5e6)


def test_weekly_splits_two_weeks():
    weekly = resample_candles(daily(10, "2024-01-01"), "week")
    assert len(weekly) == 2


def test_bar_date_is_last_trading_day():
    df = daily(5, "2024-01-01")
    weekly = resample_candles(df, "week")
    # Дата бара — реальная последняя торговая дата (пятница 2024-01-05).
    assert weekly.iloc[0]["date"] == pd.Timestamp("2024-01-05")


def test_monthly_aggregation():
    df = daily(60, "2024-01-01")  # ~3 месяца рабочих дней
    monthly = resample_candles(df, "month")
    assert len(monthly) == 3
    assert monthly["date"].is_monotonic_increasing


def test_quarterly_aggregation():
    df = daily(180, "2024-01-01")
    quarterly = resample_candles(df, "quarter")
    assert 2 <= len(quarterly) <= 4


def test_all_securities_align_on_same_dates():
    """Две бумаги, ресемпленные одним правилом, дают одинаковые даты баров."""
    a = resample_candles(daily(20, "2024-01-01"), "week")
    b = resample_candles(daily(20, "2024-01-01"), "week")
    assert list(a["date"]) == list(b["date"])


def test_empty_input():
    assert resample_candles(pd.DataFrame(), "week").empty


def test_unknown_timeframe_rejected():
    with pytest.raises(ValueError, match="Неизвестный таймфрейм"):
        resample_candles(daily(5), "год")
