"""Мониторинг моделей и режимов — ЧАСТЬ II (ТЗ, раздел 22).

Модель конечна: обученная на одном режиме, она перестаёт работать при
смене режима — это нормальное свойство, а не отказ. Здесь — детекторы:

* CUSUM — накопительная сумма отклонений, ловит структурные сдвиги ряда;
* PSI (Population Stability Index) — дрейф распределения признака между
  обучающей выборкой и текущими данными (порог 0,25 — сильный дрейф);
* KS-тест — тот же вопрос статистически строго.

Реализация своя (кроме KS из scipy), с тестами на синтетике.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_events(series: pd.Series, threshold: float) -> list[tuple]:
    """Симметричный CUSUM-фильтр структурных сдвигов.

    Накапливает отклонения приращений; когда накопленная сумма превышает
    порог, фиксирует событие и обнуляется. Возвращает список
    (метка_времени, направление ±1).
    """
    if threshold <= 0:
        raise ValueError("порог CUSUM должен быть положительным")
    events = []
    s_pos = s_neg = 0.0
    diff = series.diff().dropna()
    for t, d in diff.items():
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_pos > threshold:
            s_pos = 0.0
            events.append((t, 1))
        elif s_neg < -threshold:
            s_neg = 0.0
            events.append((t, -1))
    return events


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """PSI между обучающим (expected) и текущим (actual) распределениями.

    Границы бинов — по квантилям обучающего распределения. PSI < 0,1 —
    сдвига нет; 0,1–0,25 — умеренный; > 0,25 — сильный дрейф, модель под
    подозрением (ТЗ, раздел 22).
    """
    expected = np.asarray(expected, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = np.asarray(actual, dtype=float)
    actual = actual[~np.isnan(actual)]
    if len(expected) < bins or len(actual) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    eps = 1e-6
    e_pct = e_counts / e_counts.sum() + eps
    a_pct = a_counts / a_counts.sum() + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_drift(expected, actual) -> dict:
    """Двухвыборочный тест Колмогорова — Смирнова на дрейф распределения."""
    from scipy.stats import ks_2samp

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 3 or len(actual) < 3:
        return {"statistic": float("nan"), "pvalue": float("nan"), "drift": False}
    stat, p = ks_2samp(expected, actual)
    return {"statistic": float(stat), "pvalue": float(p), "drift": bool(p < 0.05)}


def feature_drift_psi(train_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    """PSI по каждому признаку между обучающей и текущей выборками."""
    common = [c for c in train_df.columns if c in live_df.columns]
    rows = [
        {"feature": c,
         "psi": population_stability_index(train_df[c], live_df[c]),
         "strong_drift": population_stability_index(train_df[c], live_df[c]) > 0.25}
        for c in common
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
