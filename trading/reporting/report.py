"""Отчёты. Полные отчёты бэктеста — фаза 4.

Уже сейчас реализована обязательная шапка каждого отчёта: порог
безубыточности стратегии. ТЗ (раздел 1): этот факт не должен быть
спрятан, система обязана показывать его в каждом отчёте.
"""

from __future__ import annotations

from trading.formatting import fmt_rub
from trading.settings import Settings


def breakeven_header_ru(settings: Settings) -> str:
    """Обязательная шапка каждого отчёта: экономика проекта честно."""
    capital = settings.capital.start_amount
    infra = settings.benchmark.infra_cost_rub_year
    rf = settings.benchmark.risk_free_rate
    infra_share = infra / capital
    breakeven = settings.breakeven_rate()
    return (
        f"ЭКОНОМИКА ПРОЕКТА\n"
        f"Капитал: {fmt_rub(capital, 0)} ₽\n"
        f"Безрисковая альтернатива (фонд денежного рынка): {rf:.2%} годовых\n"
        f"Инфраструктура: {fmt_rub(infra, 0)} ₽/год = {infra_share:.1%} капитала\n"
        f"ПОРОГ БЕЗУБЫТОЧНОСТИ СТРАТЕГИИ: ~{breakeven:.0%} годовых.\n"
        f"Стратегия, зарабатывающая меньше, проигрывает вкладу под ключевую "
        f"ставку и не окупает инфраструктуру."
    )


def full_report(*args, **kwargs):
    raise NotImplementedError("Полный отчёт бэктеста — фаза 4. Модуль ещё не реализован.")
