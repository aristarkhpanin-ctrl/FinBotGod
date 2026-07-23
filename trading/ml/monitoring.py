"""Мониторинг моделей — ЧАСТЬ II (заглушка до прохождения фазы 8).

Планируется (ТЗ, раздел 22): CUSUM для структурных сдвигов, детекция
рыночного режима, дрейф признаков (PSI, KS-тест), автоматический вывод
модели из эксплуатации по заранее зафиксированным критериям.
"""

from __future__ import annotations


def cusum_breaks(*args, **kwargs):
    raise NotImplementedError("CUSUM — Часть II, после фазы 8.")


def feature_drift_psi(*args, **kwargs):
    raise NotImplementedError("Дрейф признаков (PSI) — Часть II, после фазы 8.")
