"""Ступень 1 лестницы сложности — линейные модели (ТЗ, раздел 20).

Логистическая регрессия как ВТОРИЧНАЯ (мета) модель: предсказывает
вероятность того, что сигнал первичной модели сработает. Открывается
только после ступени 0 (правила) и при PBO < 0,5.

Обязательный метод ``explain`` — объяснение конкретного решения строкой
на русском. Модель без него в контур не допускается даже в
исследовательский (прямое следствие раздела 0 ТЗ).
"""

from __future__ import annotations

import numpy as np


class LogisticMetaModel:
    """Мета-фильтр сигналов на логистической регрессии со стандартизацией.

    Стандартизатор обучается ВНУТРИ fit только на переданных обучающих
    данных — это не нарушает запрет раздела 19 (fit скейлера на полном
    датасете до разбиения), потому что fit получает лишь train-фолд.
    """

    def __init__(self, feature_names: list[str], C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.feature_names = list(feature_names)
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        self._fitted = False

    def fit(self, X, y, sample_weight=None) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        Xs = self._scaler.fit_transform(X)
        self._clf.fit(Xs, y, sample_weight=sample_weight)
        self._fitted = True

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        Xs = self._scaler.transform(np.asarray(X, dtype=float))
        return self._clf.predict_proba(Xs)

    def feature_importance(self) -> dict[str, float]:
        """Быстрая важность = модуль стандартизованного коэффициента.

        Это ПРОКСИ. Полноценная важность — MDA на purged-разбиении
        (ml.feature_importance.mda_importance), она не смещена и учитывает
        корреляции; см. раздел 20 ТЗ.
        """
        self._check_fitted()
        coefs = np.abs(self._clf.coef_[0])
        total = coefs.sum() or 1.0
        return dict(zip(self.feature_names, coefs / total))

    def explain(self, x) -> str:
        """Объяснение решения по одному наблюдению — строкой на русском."""
        self._check_fitted()
        x = np.asarray(x, dtype=float).reshape(1, -1)
        xs = self._scaler.transform(x)[0]
        contrib = self._clf.coef_[0] * xs           # вклад каждого признака
        proba = float(self.predict_proba(x)[0, 1])
        order = np.argsort(-np.abs(contrib))
        parts = []
        for i in order[:3]:
            sign = "за" if contrib[i] > 0 else "против"
            parts.append(f"{self.feature_names[i]} ({sign}, вклад {contrib[i]:+.2f})")
        return (
            f"Вероятность, что сигнал сработает: {proba:.0%}. "
            f"Главные факторы: {'; '.join(parts)}. "
            + ("Рекомендация: действовать по сигналу."
               if proba >= 0.5 else "Рекомендация: пропустить сигнал.")
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Модель не обучена — сначала fit().")
