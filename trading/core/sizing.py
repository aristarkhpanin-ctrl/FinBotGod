"""Расчёт размера позиции с учётом лотности — ФАЗА 3 (заглушка).

Будущая логика (ТЗ, раздел 6):

    target_rubles = portfolio_value * target_weight
    lots = floor(target_rubles / (price * lot_size))
    actual_shares = lots * lot_size

Обязательные проверки: lots == 0 → пропуск сделки с причиной в лог;
превышение max_position_pct → обрезание до лимита; достаточность денег
с учётом комиссии. Плюс отчёт о разрешающей способности капитала.
"""

from __future__ import annotations


def size_position(*args, **kwargs):
    raise NotImplementedError("Сайзинг — фаза 3. Модуль ещё не реализован.")


def resolution_report(*args, **kwargs):
    raise NotImplementedError(
        "Отчёт о разрешающей способности — фаза 3. Модуль ещё не реализован."
    )
