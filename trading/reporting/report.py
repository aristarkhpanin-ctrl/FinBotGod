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


def _pct(x, digits: int = 1) -> str:
    """Процент или «н/д», если метрика не определена."""
    import math

    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "н/д"
    return f"{x:.{digits}%}"


def _num(x, digits: int = 2) -> str:
    import math

    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "н/д"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{digits}f}"


def full_report_ru(settings: Settings, result) -> str:
    """Полный отчёт бэктеста (ТЗ, разделы 7 и 13).

    Обязательно: три бенчмарка; если стратегия проигрывает фонду
    денежного рынка — это выводится ПЕРВЫМ, до всех остальных цифр.
    """
    m = result.metrics
    rf = settings.benchmark.risk_free_rate
    lines: list[str] = [breakeven_header_ru(settings), ""]
    lines += [result.resolution_report, ""]

    strategy_annual = m.get("annual_return")
    if strategy_annual is not None and strategy_annual < rf:
        lines += [
            "=" * 64,
            f"!!! СТРАТЕГИЯ ПРОИГРЫВАЕТ ФОНДУ ДЕНЕЖНОГО РЫНКА !!!",
            f"!!! {_pct(strategy_annual)} годовых против {_pct(rf, 2)} без риска !!!",
            "=" * 64,
            "",
        ]

    imoex = result.benchmarks.get("IMOEX")
    mm = result.benchmarks["денежный_рынок"]
    lines += [
        "СРАВНЕНИЕ (годовых):",
        f"  1. Стратегия:              {_pct(strategy_annual)} "
        f"(итог {fmt_rub(m.get('end_equity', 0), 0)} ₽)",
        f"  2. Купил и держи IMOEX:    "
        + (_pct(imoex["annual_return"]) if imoex else "н/д — данных индекса нет"),
        f"  3. Фонд денежного рынка:   {_pct(mm['annual_return'], 2)} "
        f"(итог {fmt_rub(mm['end_equity'], 0)} ₽)",
        "",
        "МЕТРИКИ:",
        f"  Доходность за период:      {_pct(m.get('total_return'))}",
        f"  CAGR:                      {_pct(strategy_annual)}",
        f"  Макс. просадка:            {_pct(m.get('max_drawdown'))} "
        f"(длительность {m.get('max_drawdown_days', 'н/д')} дн.)",
        f"  Шарп (безрисковая {rf:.2%}): {_num(m.get('sharpe'))}",
        f"  Сортино:                   {_num(m.get('sortino'))}",
        f"  Кальмар:                   {_num(m.get('calmar'))}",
        f"  Сделок: {m.get('n_trades', 0)} | Оборот: {fmt_rub(m.get('turnover', 0), 0)} ₽",
        f"  Доля прибыльных продаж:    {_pct(m.get('win_rate'))} | "
        f"Профит-фактор: {_num(m.get('profit_factor'))}",
        "",
        "ИЗДЕРЖКИ:",
        f"  Всего: {fmt_rub(m.get('total_costs', 0))} ₽ "
        f"({_pct(m.get('costs_pct_of_gross'))} от валовой прибыли)",
    ]

    total_ndfl = sum(result.ndfl_by_year.values())
    if result.ndfl_by_year:
        by_year = ", ".join(
            f"{y}: {fmt_rub(v, 0)} ₽" for y, v in sorted(result.ndfl_by_year.items())
        )
        lines.append(f"  НДФЛ 13% с зафиксированной прибыли: {fmt_rub(total_ndfl, 0)} ₽ ({by_year})")
    else:
        lines.append("  НДФЛ: прибыль не фиксировалась, налога нет")

    if settings.costs.stress_multiplier > 1:
        lines += [
            "",
            f"ВНИМАНИЕ: это СТРЕСС-ПРОГОН с издержками "
            f"× {settings.costs.stress_multiplier:g}.",
        ]
    if result.limitations:
        lines += [""] + [f"ОГРАНИЧЕНИЕ: {t}" for t in result.limitations]
    lines += ["", f"Идентификатор прогона в журнале гипотез: {result.run_hash}"]
    return "\n".join(lines)
