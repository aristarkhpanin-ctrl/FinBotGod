"""Реализации протокола Model — ЧАСТЬ II (заглушка до прохождения фазы 8).

Единый протокол (ТЗ, раздел 20): fit, predict_proba, feature_importance
и ОБЯЗАТЕЛЬНЫЙ метод explain — объяснение конкретного решения строкой
на русском языке. Модель без explain не допускается даже в
исследовательский контур.

Лестница сложности: 0 — правила без обучения; 1 — линейные модели;
2 — деревья и бустинг; 3 — последовательные модели; 4 — RL (интерфейс
закладывается, в план работ не ставится). Переход на ступень выше без
выполнения условий допуска запрещён.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Model(Protocol):
    def fit(self, X, y, sample_weight) -> None: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def feature_importance(self) -> dict[str, float]: ...
    def explain(self, x) -> str: ...   # объяснение на русском языке
