"""Point-in-time хранилище признаков — ЧАСТЬ II (заглушка).

Все соединения признаков с ценовым рядом — исключительно через
``pandas.merge_asof`` с ``direction='backward'`` по времени ДОСТУПНОСТИ
(момент события + availability_lag), никогда по времени события.
Обычный merge по дате в этом проекте запрещён.
"""

from __future__ import annotations


class ForbiddenJoinError(Exception):
    """Попытка соединения признаков в обход as-of по времени доступности."""


class FeatureStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Хранилище признаков — Часть II, после фазы 8.")
