"""Реестр признаков с обязательной декларацией задержки доступности.

Признак, посчитанный по данным, которых в тот момент не существовало, —
утечка будущего в замаскированном виде. Поэтому ни один признак не
регистрируется без ``availability_lag`` — задержки между моментом,
к которому относятся данные, и моментом их реальной публичной доступности.

Примеры задержек:
* цена закрытия дня — 0 (доступна в момент закрытия);
* квартальная отчётность — 45–90 дней от конца квартала;
* данные Росстата — 20–40 дней;
* дивидендная отсечка — известна заранее (отрицательная задержка).

Отсутствие задержки — ошибка на этапе регистрации, а не предупреждение.
Признаки от LLM помечаются флагом ``llm_derived`` (ТЗ, раздел 21).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable


class FeatureRegistrationError(Exception):
    """Признак объявлен неправильно и в реестр не попадёт."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    compute_fn: Callable
    availability_lag: timedelta   # ОБЯЗАТЕЛЬНО, без значения по умолчанию
    source: str
    description_ru: str           # для человекочитаемого журнала
    llm_derived: bool = False     # признак получен от LLM (особые правила валидации)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise FeatureRegistrationError("У признака должно быть имя.")
        if not callable(self.compute_fn):
            raise FeatureRegistrationError(
                f"compute_fn признака «{self.name}» должен быть функцией."
            )
        if not isinstance(self.availability_lag, timedelta):
            raise FeatureRegistrationError(
                f"Признак «{self.name}»: availability_lag должен быть timedelta "
                f"(получено: {type(self.availability_lag).__name__}). Задержка "
                f"доступности данных обязана быть указана явно — это защита "
                f"от утечки будущего."
            )
        if not self.description_ru or not self.description_ru.strip():
            raise FeatureRegistrationError(
                f"Признак «{self.name}»: нужно описание на русском языке "
                f"(description_ru) для человекочитаемого журнала."
            )
        if not self.source or not self.source.strip():
            raise FeatureRegistrationError(
                f"Признак «{self.name}»: нужно указать источник данных (source)."
            )


class FeatureRegistry:
    def __init__(self):
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise FeatureRegistrationError(
                f"Признак «{spec.name}» уже зарегистрирован. Изменение "
                f"существующего признака — это новый признак с новым именем "
                f"(например, «{spec.name}_v2»), иначе старые прогоны в журнале "
                f"гипотез перестанут быть воспроизводимыми."
            )
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        if name not in self._specs:
            raise KeyError(f"Признак «{name}» не зарегистрирован.")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs)

    def llm_derived_names(self) -> list[str]:
        """Имена признаков от LLM — для особых правил валидации (раздел 21)."""
        return sorted(n for n, s in self._specs.items() if s.llm_derived)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs
