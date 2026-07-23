"""Валидация ML — ЧАСТЬ II (заглушка до прохождения фазы 8).

Планируется (ТЗ, раздел 19): Purged K-Fold с эмбарго, комбинаторно-
симметричная кросс-валидация (CSCV), вероятность переподгонки бэктеста
(PBO), дефлированный коэффициент Шарпа.

ЗАПРЕЩЁННЫЕ ПРАКТИКИ (вызывают ошибку, не предупреждение):
train_test_split с shuffle=True на временных рядах; KFold без очистки;
fit скейлера или отбор признаков на полном датасете до разбиения;
заполнение пропусков средним по всей истории.
"""

from __future__ import annotations


class ForbiddenPracticeError(Exception):
    """Использована практика, ведущая к утечке будущего."""


def purged_kfold(*args, **kwargs):
    raise NotImplementedError("Purged K-Fold — Часть II, после фазы 8.")


def cscv_pbo(*args, **kwargs):
    raise NotImplementedError("CSCV / PBO — Часть II, после фазы 8.")


def deflated_sharpe(*args, **kwargs):
    raise NotImplementedError("Дефлированный Шарп — Часть II, после фазы 8.")
