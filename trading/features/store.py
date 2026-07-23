"""Point-in-time хранилище признаков — ЧАСТЬ II (ТЗ, раздел 17).

Признак, посчитанный по данным, которых в тот момент не существовало, —
это утечка будущего в замаскированном виде. Правило: все соединения
признаков с ценовым рядом выполняются ТОЛЬКО через ``pandas.merge_asof``
с ``direction='backward'`` по ВРЕМЕНИ ДОСТУПНОСТИ (момент события +
availability_lag), никогда по времени события. Обычный merge по дате
в этом проекте запрещён.

Пример: квартальная выручка за I квартал относится к 31 марта, но
публикуется в мае (лаг ~45–90 дней). До даты публикации её знать нельзя,
поэтому в строку 1 апреля она попасть не должна — merge_asof по времени
доступности это гарантирует.
"""

from __future__ import annotations

import pandas as pd

from trading.features.registry import FeatureRegistry, FeatureSpec


class ForbiddenJoinError(Exception):
    """Попытка соединения признаков в обход as-of по времени доступности."""


class FeatureStore:
    def __init__(self, registry: FeatureRegistry | None = None):
        self.registry = registry or FeatureRegistry()
        self._values: dict[str, pd.Series] = {}

    def add_values(self, name: str, series: pd.Series) -> None:
        """Сохранить значения признака. Индекс series — ВРЕМЯ СОБЫТИЯ
        (момент, к которому относятся данные), не время публикации."""
        if name not in self.registry:
            raise KeyError(
                f"Признак «{name}» не зарегистрирован — сначала registry.register "
                f"с обязательным availability_lag."
            )
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
        self._values[name] = series

    def availability_index(self, name: str) -> pd.Series:
        """Значения признака, переиндексированные на ВРЕМЯ ДОСТУПНОСТИ
        (время события + лаг). Именно с этим временем признак реально
        существовал в мире."""
        spec: FeatureSpec = self.registry.get(name)
        series = self._values[name]
        avail = series.copy()
        avail.index = series.index + spec.availability_lag
        return avail.sort_index()

    def as_of(self, target_index: pd.DatetimeIndex, name: str) -> pd.Series:
        """As-of соединение признака с целевыми датами.

        Для каждой даты берётся последнее значение признака, СТАВШЕЕ
        ДОСТУПНЫМ не позже этой даты (merge_asof, direction='backward').
        Значения, которых в тот момент ещё не существовало, не подмешиваются.
        """
        avail = self.availability_index(name)
        left = pd.DataFrame({"date": pd.DatetimeIndex(target_index)}).sort_values("date")
        right = pd.DataFrame(
            {"date": avail.index, name: avail.values}
        ).sort_values("date")
        merged = pd.merge_asof(left, right, on="date", direction="backward")
        return pd.Series(merged[name].values, index=target_index, name=name)

    @staticmethod
    def forbid_naive_merge() -> None:
        """Явный запрет обычного merge по дате события (раздел 17)."""
        raise ForbiddenJoinError(
            "Обычный merge/join признаков по дате события в этом проекте "
            "запрещён: он подмешивает данные, которых на тот момент ещё не "
            "существовало. Используйте FeatureStore.as_of (merge_asof по "
            "времени доступности)."
        )
