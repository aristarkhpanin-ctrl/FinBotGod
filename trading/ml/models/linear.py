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


class CalibratedLogisticModel:
    """Логистическая модель с КАЛИБРОВКОЙ вероятностей (Platt scaling).

    Зачем: чтобы порог «уверенность ≥ 80%» имел смысл, предсказанная
    вероятность должна соответствовать реальной частоте. Сырые выходы
    логистической регрессии на шумных финансовых данных смещены. Калибровка
    делается на ОТЛОЖЕННОМ по времени хвосте обучающего окна (без утечки
    из теста): база учится на первых 80% обучения, сигмоида-калибратор — на
    последних 20%.

    Сохраняет explain() (через базовые коэффициенты) — требование ТЗ.
    """

    def __init__(self, feature_names: list[str], C: float = 1.0,
                 calib_fraction: float = 0.2):
        from sklearn.linear_model import LogisticRegression

        self.feature_names = list(feature_names)
        self._base = LogisticMetaModel(feature_names, C)
        self._calibrator = LogisticRegression(max_iter=1000)
        self._calib_fraction = calib_fraction
        self._fitted = False

    def fit(self, X, y, sample_weight=None) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n = len(X)
        k = max(int(n * (1 - self._calib_fraction)), 10)
        if k >= n or len(np.unique(y[:k])) < 2 or len(np.unique(y[k:])) < 2:
            # Данных мало для честной калибровки — учим базу на всём,
            # калибратор становится тождественным.
            self._base.fit(X, y, sample_weight)
            self._identity = True
            self._fitted = True
            return
        self._identity = False
        sw = None if sample_weight is None else np.asarray(sample_weight)[:k]
        self._base.fit(X[:k], y[:k], sw)
        raw = self._base.predict_proba(X[k:])[:, 1].reshape(-1, 1)
        self._calibrator.fit(raw, y[k:])
        self._fitted = True

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Модель не обучена — сначала fit().")
        raw = self._base.predict_proba(X)[:, 1]
        if getattr(self, "_identity", True):
            return np.column_stack([1 - raw, raw])
        cal = self._calibrator.predict_proba(raw.reshape(-1, 1))
        return cal

    def feature_importance(self) -> dict[str, float]:
        return self._base.feature_importance()

    def explain(self, x) -> str:
        base_text = self._base.explain(x)
        proba = float(self.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0, 1])
        return f"[калиброванная уверенность {proba:.0%}] " + base_text
