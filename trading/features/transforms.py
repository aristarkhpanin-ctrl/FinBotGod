"""Преобразования признаков — ЧАСТЬ II (ТЗ, раздел 17).

Дробное дифференцирование (fractional differentiation): ценовые ряды
нестационарны, а большинство моделей это предполагают. Простые разности
(d=1) убивают память ряда. Дробный порядок d ∈ (0, 1) делает ряд
стационарным, сохраняя максимум предсказательной информации.

Подбирается минимальный d, при котором тест Дики — Фуллера проходит.
Реализация своя, с тестом на синтетике с известным ответом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def frac_diff_weights(d: float, size: int, threshold: float = 1e-3) -> np.ndarray:
    """Веса биномиального ряда дробной разности порядка d.

    w[0]=1, w[k] = -w[k-1] * (d - k + 1) / k. Обрезаются, когда становятся
    меньше threshold по модулю (окно фиксированной ширины).
    """
    w = [1.0]
    k = 1
    while k < size:
        next_w = -w[-1] * (d - k + 1) / k
        if abs(next_w) < threshold:
            break
        w.append(next_w)
        k += 1
    return np.array(w[::-1])   # от старых к новым


def frac_diff(series: pd.Series, d: float, threshold: float = 1e-3) -> pd.Series:
    """Дробная разность ряда порядка d (метод фиксированного окна).

    Каждое значение = свёртка последних len(weights) точек с весами.
    Первые точки, где окна не хватает, отбрасываются (NaN).
    """
    if not (0 <= d <= 1):
        raise ValueError(f"порядок d должен быть в [0, 1], получено {d}")
    s = series.astype(float)
    weights = frac_diff_weights(d, len(s), threshold)
    width = len(weights)
    out = pd.Series(index=s.index, dtype=float)
    values = s.to_numpy()
    for i in range(width - 1, len(s)):
        window = values[i - width + 1: i + 1]
        if np.isnan(window).any():
            continue
        out.iloc[i] = float(np.dot(weights, window))
    return out


def adf_pvalue(series: pd.Series) -> float:
    """p-значение теста Дики — Фуллера. Меньше 0,05 → ряд стационарен."""
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 20:
        return float("nan")
    return float(adfuller(clean, maxlag=1, regression="c", autolag=None)[1])


def min_frac_diff_order(
    series: pd.Series, pvalue_threshold: float = 0.05,
    grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> dict:
    """Минимальный порядок d, при котором ряд проходит тест Дики — Фуллера.

    Возвращает найденный d, p-значение и сам дифференцированный ряд.
    Так ряд становится стационарным, теряя минимум памяти.
    """
    for d in grid:
        diffed = frac_diff(series, d)
        p = adf_pvalue(diffed)
        if not np.isnan(p) and p < pvalue_threshold:
            return {"d": d, "adf_pvalue": p, "series": diffed, "stationary": True}
    # Даже d=1 не дал стационарности — вернём последнее.
    diffed = frac_diff(series, grid[-1])
    return {
        "d": grid[-1], "adf_pvalue": adf_pvalue(diffed),
        "series": diffed, "stationary": False,
    }


def fractional_differentiation(series: pd.Series, d: float | None = None):
    """Совместимость с реестром: если d задан — дробная разность,
    иначе подбор минимального стационарного порядка."""
    if d is None:
        return min_frac_diff_order(series)
    return frac_diff(series, d)
