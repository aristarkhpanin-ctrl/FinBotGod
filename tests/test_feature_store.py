"""Тесты point-in-time хранилища: признак не появляется раньше публикации."""

from datetime import timedelta

import pandas as pd
import pytest

from trading.features.registry import FeatureRegistry, FeatureSpec
from trading.features.store import FeatureStore, ForbiddenJoinError


def make_registry(name: str, lag_days: int) -> FeatureRegistry:
    reg = FeatureRegistry()
    reg.register(FeatureSpec(
        name=name,
        compute_fn=lambda df: df,
        availability_lag=timedelta(days=lag_days),
        source="тест",
        description_ru="Тестовый признак",
    ))
    return reg


def test_feature_available_only_after_lag():
    """Квартальная выручка за 31 марта с лагом 45 дней недоступна 1 апреля."""
    reg = make_registry("выручка", lag_days=45)
    store = FeatureStore(reg)
    # Событие относится к 31 марта, но публикуется 15 мая (45 дней спустя).
    store.add_values("выручка", pd.Series(
        [100.0], index=[pd.Timestamp("2024-03-31")]
    ))
    dates = pd.to_datetime(["2024-04-01", "2024-05-14", "2024-05-15", "2024-06-01"])
    joined = store.as_of(dates, "выручка")
    assert pd.isna(joined.loc["2024-04-01"])   # ещё не опубликовано
    assert pd.isna(joined.loc["2024-05-14"])   # накануне публикации
    assert joined.loc["2024-05-15"] == 100.0   # день публикации
    assert joined.loc["2024-06-01"] == 100.0   # позже — доступно


def test_zero_lag_feature_available_same_day():
    reg = make_registry("цена_закрытия", lag_days=0)
    store = FeatureStore(reg)
    store.add_values("цена_закрытия", pd.Series(
        [50.0, 51.0], index=pd.to_datetime(["2024-01-10", "2024-01-11"])
    ))
    dates = pd.to_datetime(["2024-01-10", "2024-01-11"])
    joined = store.as_of(dates, "цена_закрытия")
    assert joined.loc["2024-01-10"] == 50.0     # доступна в тот же день
    assert joined.loc["2024-01-11"] == 51.0


def test_backward_join_takes_last_available():
    reg = make_registry("ставка", lag_days=0)
    store = FeatureStore(reg)
    store.add_values("ставка", pd.Series(
        [0.16, 0.18], index=pd.to_datetime(["2024-01-01", "2024-06-01"])
    ))
    dates = pd.to_datetime(["2024-03-01", "2024-07-01"])
    joined = store.as_of(dates, "ставка")
    assert joined.loc["2024-03-01"] == 0.16     # действует прежнее значение
    assert joined.loc["2024-07-01"] == 0.18     # уже новое


def test_negative_lag_available_earlier():
    """Дивидендная отсечка известна заранее — отрицательный лаг."""
    reg = make_registry("отсечка", lag_days=-14)
    store = FeatureStore(reg)
    store.add_values("отсечка", pd.Series(
        [1.0], index=[pd.Timestamp("2024-05-15")]
    ))
    joined = store.as_of(pd.to_datetime(["2024-05-02", "2024-05-10"]), "отсечка")
    assert joined.loc["2024-05-02"] == 1.0      # известно за 13 дней до события


def test_unregistered_feature_rejected():
    store = FeatureStore(FeatureRegistry())
    with pytest.raises(KeyError, match="не зарегистрирован"):
        store.add_values("нет", pd.Series([1.0], index=[pd.Timestamp("2024-01-01")]))


def test_naive_merge_forbidden():
    with pytest.raises(ForbiddenJoinError, match="merge_asof"):
        FeatureStore.forbid_naive_merge()
