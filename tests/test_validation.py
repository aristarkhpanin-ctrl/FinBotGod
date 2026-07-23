"""Тесты валидации данных: каждая аномалия из ТЗ (раздел 4) ловится."""

import pandas as pd
import pytest

from trading.data.validation import (
    build_trading_calendar,
    validate_candles,
)


def make_candles(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
            ),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "value": [1e6] * 4,
            "volume": [1000] * 4,
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def test_clean_data_has_no_anomalies():
    report = validate_candles(make_candles(), "SBER")
    assert report.total_anomalies == 0
    assert not report.suspicious
    assert "аномалий нет" in report.describe_ru()


def test_duplicate_dates_detected():
    df = make_candles()
    df.loc[3, "date"] = df.loc[2, "date"]
    report = validate_candles(df, "SBER")
    assert report.duplicate_dates == 1


def test_bad_high_detected():
    # high ниже close — такого на бирже не бывает, это битые данные
    report = validate_candles(make_candles(high=[102.0, 101.5, 104.0, 105.0]), "SBER")
    assert report.bad_high == 1


def test_bad_low_detected():
    # low выше open — битые данные
    report = validate_candles(make_candles(low=[100.5, 100.0, 101.0, 102.0]), "SBER")
    assert report.bad_low == 1


def test_negative_volume_detected():
    report = validate_candles(make_candles(volume=[1000, -5, 1000, 1000]), "SBER")
    assert report.negative_volume == 1


def test_zero_price_detected():
    report = validate_candles(make_candles(low=[99.0, 0.0, 101.0, 102.0]), "SBER")
    assert report.zero_price == 1


def test_price_jump_marks_security_suspicious():
    """Скачок > 35% за день → бумага исключается из бэктеста (возможен сплит)."""
    df = make_candles(close=[101.0, 102.0, 160.0, 161.0])  # +57% за день
    report = validate_candles(df, "SBER", jump_threshold=0.35)
    assert report.suspicious
    assert len(report.price_jumps) == 1
    date, change = report.price_jumps[0]
    assert date == "2024-01-11"
    assert change == pytest.approx(160.0 / 102.0 - 1)
    assert "исключена из бэктеста" in report.describe_ru()


def test_jump_below_threshold_is_ok():
    df = make_candles(close=[101.0, 102.0, 130.0, 131.0])  # +27% — много, но ниже порога
    report = validate_candles(df, "SBER", jump_threshold=0.35)
    assert not report.suspicious


def test_missing_days_against_calendar():
    """Дыра в данных бумаги видна на фоне общего торгового календаря."""
    full = make_candles()
    with_hole = full.drop(index=2).reset_index(drop=True)  # пропал 2024-01-11
    calendar = build_trading_calendar({"SBER": full, "GAZP": with_hole})
    report = validate_candles(with_hole, "GAZP", trading_calendar=calendar)
    assert report.missing_days == 1


def test_empty_data_reported():
    report = validate_candles(pd.DataFrame(), "PLZL")
    assert report.n_rows == 0
    assert "данных нет" in report.describe_ru()
