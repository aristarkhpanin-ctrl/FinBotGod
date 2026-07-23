"""Симулятор исполнения для бэктеста и бумажной торговли — ФАЗА 4.

Исполнение по переданной цене (движок подаёт цену ОТКРЫТИЯ дня T+1 —
оптимистичное исполнение по цене сигнала запрещено). Все издержки,
включая проскальзывание, списываются деньгами через модель издержек.

Идентификатор заявки детерминирован: дата + тикер + направление —
задел под идемпотентность боевого контура (ТЗ, раздел 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

from trading.core.costs import CostModel, TradeCosts
from trading.core.portfolio import Portfolio


@dataclass(frozen=True)
class Fill:
    client_order_id: str
    day: Date
    secid: str
    side: str                 # "buy" | "sell"
    shares: int
    price: float              # цена исполнения (open дня T+1)
    order_value: float
    costs: TradeCosts
    realized_pnl: float | None   # только для продаж


class SimulatedExecution:
    """Единственное место, где сделки меняют портфель в бэктесте."""

    def __init__(self, portfolio: Portfolio, cost_model: CostModel):
        self.portfolio = portfolio
        self.cost_model = cost_model
        self.fills: list[Fill] = []

    def execute(self, day: Date, secid: str, side: str, shares: int,
                price: float, adv_20: float) -> Fill:
        """Исполняет заявку. Ликвидность проверяет модель издержек:
        нереализуемая заявка бросает UnfillableOrderError ДО изменения
        портфеля."""
        order_value = shares * price
        costs = self.cost_model.trade_costs(order_value, adv_20)
        realized = None
        if side == "buy":
            self.portfolio.buy(secid, shares, price, costs, day)
        elif side == "sell":
            realized = self.portfolio.sell(secid, shares, price, costs, day)
        else:
            raise ValueError(f"Неизвестное направление заявки: {side}")
        fill = Fill(
            client_order_id=f"{day.isoformat()}-{secid}-{side}",
            day=day, secid=secid, side=side, shares=shares, price=price,
            order_value=order_value, costs=costs, realized_pnl=realized,
        )
        self.fills.append(fill)
        return fill
