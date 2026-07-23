"""Расчёт размера позиции с учётом лотности — ФАЗА 3.

Формула (ТЗ, раздел 6):

    target_rubles = portfolio_value * target_weight
    lots = floor(target_rubles / (price * lot_size))
    actual_shares = lots * lot_size

Обязательные проверки:
1. lots == 0 → сделка не совершается, причина пишется по-русски:
   «Пропуск SBER: целевая сумма 4 200 ₽ меньше стоимости одного лота 4 850 ₽».
2. Превышение max_position_pct → обрезание до лимита, пересчёт лотов вниз.
3. Достаточность денег с учётом комиссии, а не только цены.

Каждое решение объясняется одной строкой, проверяемой калькулятором.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading.core.costs import CostModel, TradeCosts, UnfillableOrderError
from trading.formatting import fmt_rub
from trading.logging_setup import get_logger

log = get_logger("sizing")


@dataclass(frozen=True)
class SizingDecision:
    secid: str
    lots: int
    shares: int
    order_value: float           # рублей, без издержек
    costs: TradeCosts | None     # издержки сделки; None при пропуске
    reason_ru: str               # объяснение решения одной строкой

    @property
    def skipped(self) -> bool:
        return self.lots == 0


def size_position(
    secid: str,
    price: float,
    lot_size: int,
    target_weight: float,
    portfolio_value: float,
    cash_available: float,
    max_position_pct: float,
    cost_model: CostModel,
    adv_20: float,
) -> SizingDecision:
    """Считает покупку одной бумаги. Возвращает решение с объяснением.

    Никогда не бросает исключений о ликвидности — нереализуемая заявка
    превращается в пропуск с причиной, понятной заказчику.
    """
    if price <= 0 or lot_size <= 0:
        return _skip(secid, f"Пропуск {secid}: некорректные цена ({price}) "
                            f"или лот ({lot_size}) — битые данные.")
    if target_weight <= 0:
        return _skip(secid, f"Пропуск {secid}: целевая доля {target_weight:.0%}.")

    lot_cost = price * lot_size
    target_rubles = portfolio_value * target_weight

    # Проверка 2: обрезание до лимита на позицию.
    limit_rubles = portfolio_value * max_position_pct
    trimmed = target_rubles > limit_rubles
    effective_target = min(target_rubles, limit_rubles)

    lots = math.floor(effective_target / lot_cost)

    # Проверка 1: целевая сумма меньше стоимости одного лота.
    if lots == 0:
        reason = (
            f"Пропуск {secid}: целевая сумма {fmt_rub(effective_target, 0)} ₽ "
            f"меньше стоимости одного лота {fmt_rub(lot_cost, 0)} ₽"
        )
        if trimmed:
            reason += f" (доля обрезана лимитом {max_position_pct:.0%} на позицию)"
        log.info(reason)
        return _skip(secid, reason)

    # Проверка 3: хватает ли денег на заявку вместе с издержками.
    # Уменьшаем лоты, пока заявка + комиссия не влезут в свободные деньги.
    while lots > 0:
        order_value = lots * lot_cost
        try:
            costs = cost_model.trade_costs(order_value, adv_20)
        except UnfillableOrderError as e:
            reason = f"Пропуск {secid}: {e}"
            log.warning(reason)
            return _skip(secid, reason)
        if order_value + costs.total <= cash_available:
            break
        lots -= 1
    else:
        one_lot_with_costs = lot_cost + cost_model.trade_costs(lot_cost, adv_20).total
        reason = (
            f"Пропуск {secid}: свободно {fmt_rub(cash_available, 0)} ₽, "
            f"а один лот с издержками стоит {fmt_rub(one_lot_with_costs, 0)} ₽"
        )
        log.info(reason)
        return _skip(secid, reason)

    shares = lots * lot_size
    reason = (
        f"{secid}: {lots} лот(ов) = {shares} шт по {fmt_rub(price)} ₽ "
        f"= {fmt_rub(order_value)} ₽ + издержки {fmt_rub(costs.total)} ₽"
    )
    if trimmed:
        reason += (
            f" (целевая сумма {fmt_rub(target_rubles, 0)} ₽ обрезана лимитом "
            f"{max_position_pct:.0%} = {fmt_rub(limit_rubles, 0)} ₽)"
        )
    return SizingDecision(
        secid=secid, lots=lots, shares=shares,
        order_value=order_value, costs=costs, reason_ru=reason,
    )


def _skip(secid: str, reason: str) -> SizingDecision:
    return SizingDecision(
        secid=secid, lots=0, shares=0, order_value=0.0, costs=None,
        reason_ru=reason,
    )


def resolution_report_ru(
    capital: float,
    lot_costs: dict[str, float],
    max_positions: int,
    max_position_pct: float,
    classic_risk_per_trade: float = 0.02,
) -> str:
    """Отчёт о разрешающей способности капитала (ТЗ, раздел 6).

    Заказчик видит его ДО кривой доходности: что физически может
    и чего не может этот размер счёта.

    ``lot_costs`` — стоимость одного лота по каждой бумаге универсума
    (цена * размер лота), в рублях.
    """
    if not lot_costs:
        return "Отчёт о разрешающей способности: универсум пуст, считать нечего."
    avg_lot = sum(lot_costs.values()) / len(lot_costs)
    affordable = math.floor(capital / avg_lot) if avg_lot > 0 else 0
    positions = min(max_positions, affordable)
    min_step = avg_lot / capital
    too_expensive = sorted(
        s for s, c in lot_costs.items() if c > capital * max_position_pct
    )

    lines = [
        f"РАЗРЕШАЮЩАЯ СПОСОБНОСТЬ КАПИТАЛА",
        f"Капитал: {fmt_rub(capital, 0)} ₽",
        f"Средняя стоимость лота в универсуме: {fmt_rub(avg_lot, 0)} ₽",
        f"Максимум одновременных позиций: {positions} "
        f"(лимит конфига {max_positions}, по деньгам помещается {affordable})",
        f"Минимальный шаг позиции: {min_step:.0%} портфеля",
    ]
    if min_step > classic_risk_per_trade:
        lines.append(
            f"ВНИМАНИЕ: риск-менеджмент с ограничением "
            f"{classic_risk_per_trade:.0%} на сделку нереализуем — минимальный "
            f"шаг позиции {min_step:.0%} больше этого лимита."
        )
    if too_expensive:
        lines.append(
            f"Недоступны при лимите {max_position_pct:.0%} на позицию "
            f"(один лот дороже {fmt_rub(capital * max_position_pct, 0)} ₽): "
            + ", ".join(too_expensive)
        )
    return "\n".join(lines)
