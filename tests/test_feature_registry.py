"""Тесты реестра признаков: без availability_lag признак не регистрируется."""

from datetime import timedelta

import pytest

from trading.features.registry import (
    FeatureRegistrationError,
    FeatureRegistry,
    FeatureSpec,
)


def make_spec(**overrides) -> FeatureSpec:
    defaults = dict(
        name="momentum_20",
        compute_fn=lambda df: df["close"].pct_change(20),
        availability_lag=timedelta(0),
        source="MOEX ISS, дневные свечи",
        description_ru="Изменение цены за 20 торговых дней",
    )
    defaults.update(overrides)
    return FeatureSpec(**defaults)


def test_valid_feature_registers():
    registry = FeatureRegistry()
    registry.register(make_spec())
    assert "momentum_20" in registry
    assert len(registry) == 1


def test_missing_lag_is_a_hard_error():
    """availability_lag без значения по умолчанию: не передал — TypeError."""
    with pytest.raises(TypeError):
        FeatureSpec(
            name="без_лага",
            compute_fn=lambda df: df,
            source="тест",
            description_ru="Признак без задержки",
        )


def test_wrong_lag_type_rejected():
    with pytest.raises(FeatureRegistrationError, match="timedelta"):
        make_spec(availability_lag=45)  # число вместо timedelta — ошибка


def test_negative_lag_allowed():
    """Дивидендная отсечка известна заранее — отрицательная задержка легальна."""
    spec = make_spec(name="div_cutoff", availability_lag=timedelta(days=-14))
    assert spec.availability_lag.days == -14


def test_quarterly_report_lag():
    spec = make_spec(name="выручка_квартал", availability_lag=timedelta(days=60))
    assert spec.availability_lag.days == 60


def test_missing_description_rejected():
    with pytest.raises(FeatureRegistrationError, match="русском"):
        make_spec(description_ru="")


def test_missing_source_rejected():
    with pytest.raises(FeatureRegistrationError, match="источник"):
        make_spec(source=" ")


def test_non_callable_compute_rejected():
    with pytest.raises(FeatureRegistrationError, match="функцией"):
        make_spec(compute_fn="не функция")


def test_duplicate_name_rejected():
    registry = FeatureRegistry()
    registry.register(make_spec())
    with pytest.raises(FeatureRegistrationError, match="уже зарегистрирован"):
        registry.register(make_spec())


def test_llm_derived_features_tracked_separately():
    """Признаки от LLM видны отдельно — для особых правил валидации."""
    registry = FeatureRegistry()
    registry.register(make_spec())
    registry.register(
        make_spec(name="news_sentiment", llm_derived=True,
                  availability_lag=timedelta(hours=1))
    )
    assert registry.llm_derived_names() == ["news_sentiment"]
