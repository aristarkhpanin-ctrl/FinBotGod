"""Важность признаков — ЧАСТЬ II (ТЗ, раздел 20).

Стандартная важность в деревьях (MDI) смещена в пользу признаков с высокой
кардинальностью. Здесь — среднее падение точности (MDA) на purged-разбиении:
для каждого признака его значения в тесте перемешиваются, и измеряется,
насколько просела точность. Признак, без которого модель не теряет
точность, бесполезен независимо от того, что говорит MDI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def mda_importance(model_factory, X: pd.DataFrame, y: pd.Series, cv,
                   sample_weight: pd.Series | None = None,
                   random_state: int = 0) -> pd.DataFrame:
    """Среднее падение точности при перемешивании каждого признака.

    ``model_factory`` — функция без аргументов, создающая свежую модель
    (fit/predict_proba). ``cv`` — итератор (train_idx, test_idx), обычно
    PurgedKFold. Возвращает DataFrame со средней важностью и стандартным
    отклонением по фолдам.
    """
    rng = np.random.default_rng(random_state)
    cols = list(X.columns)
    scores = {c: [] for c in cols}
    Xv, yv = X.to_numpy(), y.to_numpy().astype(int)

    for train_idx, test_idx in cv.split(X):
        if len(test_idx) == 0 or len(np.unique(yv[train_idx])) < 2:
            continue
        model = model_factory()
        sw = None if sample_weight is None else sample_weight.to_numpy()[train_idx]
        model.fit(Xv[train_idx], yv[train_idx], sample_weight=sw)
        base = accuracy_score(yv[test_idx], model.predict_proba(Xv[test_idx])[:, 1] > 0.5)
        for j, col in enumerate(cols):
            Xt = Xv[test_idx].copy()
            Xt[:, j] = rng.permutation(Xt[:, j])          # ломаем признак
            shuffled = accuracy_score(yv[test_idx], model.predict_proba(Xt)[:, 1] > 0.5)
            scores[col].append(base - shuffled)           # падение точности

    rows = []
    for col in cols:
        vals = scores[col] or [0.0]
        rows.append({"feature": col, "mda_mean": float(np.mean(vals)),
                     "mda_std": float(np.std(vals))})
    return pd.DataFrame(rows).sort_values("mda_mean", ascending=False).reset_index(drop=True)
