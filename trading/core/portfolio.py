"""Состояние портфеля: позиции, кэш, зафиксированная прибыль по годам.

Правила учёта:
* средняя цена входа (avg_cost) — по методу средней стоимости;
* зафиксированная прибыль для НДФЛ считается БЕЗ вычета комиссий из
  налоговой базы — консервативно: расчётный налог не меньше реального;
* издержки уменьшают кэш в момент сделки и тем самым честно попадают
  в кривую стоимости портфеля;
* короткие продажи запрещены: продать можно только то, что есть.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

from trading.core.costs import TradeCosts


class PortfolioError(Exception):
    """Нарушение инварианта портфеля — признак бага в вызывающем коде."""


@dataclass
class Position:
    shares: int
    avg_cost: float   # средняя цена входа за штуку, без издержек


class Portfolio:
    def __init__(self, cash: float):
        if cash <= 0:
            raise PortfolioError(f"Стартовый капитал должен быть больше нуля: {cash}")
        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.realized_pnl_by_year: dict[int, float] = {}
        self.total_costs_paid = 0.0

    def shares_of(self, secid: str) -> int:
        pos = self.positions.get(secid)
        return pos.shares if pos else 0

    def buy(self, secid: str, shares: int, price: float, costs: TradeCosts,
            day: Date) -> None:
        if shares <= 0:
            raise PortfolioError(f"Покупка {secid}: число акций должно быть > 0")
        total = shares * price + costs.total
        if total > self.cash + 1e-9:
            raise PortfolioError(
                f"Покупка {secid} на {total:.2f} ₽ при кэше {self.cash:.2f} ₽ — "
                f"проверка достаточности денег должна была сработать раньше."
            )
        pos = self.positions.get(secid)
        if pos is None:
            self.positions[secid] = Position(shares=shares, avg_cost=price)
        else:
            new_shares = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + shares * price) / new_shares
            pos.shares = new_shares
        self.cash -= total
        self.total_costs_paid += costs.total

    def sell(self, secid: str, shares: int, price: float, costs: TradeCosts,
             day: Date) -> float:
        """Продажа. Возвращает зафиксированную прибыль сделки (для журнала)."""
        pos = self.positions.get(secid)
        if pos is None or shares <= 0 or shares > pos.shares:
            raise PortfolioError(
                f"Продажа {secid}: запрошено {shares} шт, в портфеле "
                f"{pos.shares if pos else 0} шт. Короткие продажи запрещены."
            )
        realized = (price - pos.avg_cost) * shares
        self.realized_pnl_by_year[day.year] = (
            self.realized_pnl_by_year.get(day.year, 0.0) + realized
        )
        pos.shares -= shares
        if pos.shares == 0:
            del self.positions[secid]
        self.cash += shares * price - costs.total
        self.total_costs_paid += costs.total
        return realized

    def market_value(self, prices: dict[str, float]) -> float:
        value = 0.0
        for secid, pos in self.positions.items():
            if secid not in prices:
                raise PortfolioError(
                    f"Нет цены для оценки позиции {secid} — данные неполные."
                )
            value += pos.shares * prices[secid]
        return value

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)
