"""Валидация ML без утечки будущего — ЧАСТЬ II (ТЗ, раздел 19).

Обычная кросс-валидация на финансовых рядах врёт: метки перекрываются во
времени, и стандартный KFold сажает в обучение наблюдения, пересекающиеся
с тестом. Здесь реализованы:

* PurgedKFold — разбиение по времени с ОЧИСТКОЙ (из обучения убираются
  наблюдения, чьи метки перекрываются с тестом) и ЭМБАРГО (буфер после
  теста, из-за автокорреляции);
* CSCV → PBO — вероятность переподгонки бэктеста.

Запрещённые практики (раздел 19) вызывают ошибку, а не предупреждение:
это поведение библиотек по умолчанию, и его нужно блокировать явно.

Реализация своя, с тестами на синтетике с известным ответом: сторонние
часто содержат те самые утечки, от которых должны защищать.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForbiddenPracticeError(Exception):
    """Использована практика, ведущая к утечке будущего."""


def forbid_shuffle_split(shuffle: bool) -> None:
    if shuffle:
        raise ForbiddenPracticeError(
            "shuffle=True на временном ряду перемешивает будущее с прошлым — "
            "прямая утечка. На временных рядах разбиение только по времени."
        )


def forbid_plain_kfold() -> None:
    raise ForbiddenPracticeError(
        "Стандартный KFold/StratifiedKFold без очистки и эмбарго завышает "
        "качество из-за перекрытия меток. Используйте PurgedKFold."
    )


class PurgedKFold:
    """K-Fold по времени с очисткой и эмбарго.

    ``t1`` — Series: индекс = время наблюдения (начало метки), значение =
    время закрытия метки (конец её горизонта). Наблюдение занимает интервал
    [index, t1[index]]. Из обучения удаляются наблюдения, чей интервал
    пересекает тестовый отрезок, плюс эмбарго-буфер после теста.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits должен быть не меньше 2")
        if t1 is None:
            raise ValueError("t1 обязателен: без горизонтов меток очистка "
                             "невозможна (это и есть защита от утечки).")
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 должен быть отсортирован по времени наблюдения.")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame | np.ndarray | None = None):
        n = len(self.t1)
        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)
        fold_bounds = [(fold[0], fold[-1] + 1)
                       for fold in np.array_split(indices, self.n_splits)]
        times = self.t1.index

        for start, end in fold_bounds:
            test_indices = indices[start:end]
            test_t0 = times[start]                      # начало теста
            test_t1 = self.t1.iloc[start:end].max()     # конец меток теста

            # Обучение слева: наблюдения, чья метка закрылась ДО начала теста.
            train_left = indices[self.t1.values < test_t0]
            # Обучение справа: наблюдения, начавшиеся ПОСЛЕ конца меток теста
            # плюс эмбарго-буфер.
            right_start = times.searchsorted(test_t1, side="right")
            right_start = min(right_start + embargo, n)
            train_right = indices[right_start:]

            train_indices = np.concatenate([train_left, train_right])
            yield train_indices, test_indices


def cscv_pbo(performance: pd.DataFrame, n_partitions: int = 16) -> dict:
    """Вероятность переподгонки бэктеста (PBO) через CSCV.

    ``performance`` — матрица (T периодов × N конфигураций) поэлементной
    доходности за период. Метод: время делится на n_partitions частей;
    для каждого сочетания половины частей как «обучение» лучшая по обучению
    конфигурация проверяется на дополняющей половине. PBO — доля случаев,
    когда лучшая на обучении оказалась ниже медианы на проверке.

    PBO > 0,5 означает, что процедура отбора не лучше подбрасывания монеты
    (ТЗ, раздел 19): такая стратегия к боевому запуску не допускается.
    """
    from itertools import combinations

    if n_partitions % 2 != 0:
        raise ValueError("n_partitions должно быть чётным (делим пополам).")
    T, N = performance.shape
    if N < 2:
        raise ValueError("Нужно минимум 2 конфигурации для оценки PBO.")
    if T < n_partitions:
        raise ValueError(f"Периодов ({T}) меньше числа частей ({n_partitions}).")

    parts = np.array_split(np.arange(T), n_partitions)
    M = performance.to_numpy()
    logits = []
    half = n_partitions // 2
    for is_parts in combinations(range(n_partitions), half):
        is_rows = np.concatenate([parts[i] for i in is_parts])
        oos_rows = np.concatenate([parts[i] for i in range(n_partitions)
                                   if i not in is_parts])
        is_perf = M[is_rows].mean(axis=0)      # средняя доходность на обучении
        oos_perf = M[oos_rows].mean(axis=0)    # на проверке
        best = int(np.argmax(is_perf))
        # Ранг лучшей конфигурации на проверке (доля, 0..1).
        rank = (oos_perf < oos_perf[best]).sum() / (N - 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))

    logits = np.array(logits)
    pbo = float((logits <= 0).mean())   # доля случаев «ниже медианы на OOS»
    return {
        "pbo": pbo,
        "n_combinations": len(logits),
        "median_logit": float(np.median(logits)),
        "verdict_ru": (
            f"PBO = {pbo:.2f}. "
            + ("ВЫШЕ 0,5 — отбор не лучше монетки, к запуску не допускается."
               if pbo > 0.5 else
               "Ниже 0,5 — процедура отбора устойчивее случайной.")
        ),
    }


def purged_kfold(*args, **kwargs):
    """Совместимость с реестром модулей: PurgedKFold — основной класс."""
    return PurgedKFold(*args, **kwargs)
