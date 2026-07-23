"""Состояние портфеля: позиции, кэш, зафиксированная прибыль по годам.

Правила учёта:
* средняя цена входа (avg_cost) — по методу средней стоимости;
* зафиксированная прибыль для НДФЛ считается БЕЗ вычета комиссий из
  налоговой базы — консервативно: расчётный налог не меньше реального;
* издержки уменьшают кэш в момент сделки и тем самым честно попадают
  в кривую стоимости портфеля;
* короткие продажи разрешены ТОЛЬКО при allow_short=True: позиция
  становится отрицательной, выручка от продажи зачисляется в кэш,
  avg_cost хранит среднюю цену открытия шорта. Плата за заём бумаг
  начисляется движком ежедневно через charge_borrow_fee.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

from trading.core.costs import TradeCosts


class PortfolioError(Exception):
    """Нарушение инварианта портфеля — признак бага в вызывающем коде."""


@dataclass
class Position:
    shares: int       # положительное — лонг, отрицательное — шорт
    avg_cost: float   # средняя цена входа за штуку, без издержек


class Portfolio:
    def __init__(self, cash: float, allow_short: bool = False):
        if cash <= 0:
            raise PortfolioError(f"Стартовый капитал должен быть больше нуля: {cash}")
        self.cash = cash
        self.allow_short = allow_short
        self.positions: dict[str, Position] = {}
        self.realized_pnl_by_year: dict[int, float] = {}
        self.total_costs_paid = 0.0
        self.total_borrow_fees = 0.0

    def shares_of(self, secid: str) -> int:
        pos = self.positions.get(secid)
        return pos.shares if pos else 0

    def short_value(self, prices: dict[str, float]) -> float:
        """Суммарная стоимость коротких позиций (положительное число)."""
        return sum(
            -pos.shares * prices[secid]
            for secid, pos in self.positions.items()
            if pos.shares < 0 and secid in prices
        )

    def charge_borrow_fee(self, fee: float) -> None:
        """Ежедневная плата за заём бумаг по коротким позициям."""
        if fee < 0:
            raise PortfolioError(f"Плата за заём не может быть отрицательной: {fee}")
        self.cash -= fee
        self.total_borrow_fees += fee

    def _add_realized(self, realized: float, day: Date) -> None:
        self.realized_pnl_by_year[day.year] = (
            self.realized_pnl_by_year.get(day.year, 0.0) + realized
        )

    def buy(self, secid: str, shares: int, price: float, costs: TradeCosts,
            day: Date) -> float | None:
        """Покупка: открытие/наращивание лонга или закрытие шорта.

        Возвращает зафиксированную прибыль, если покупка закрывала шорт."""
        if shares <= 0:
            raise PortfolioError(f"Покупка {secid}: число акций должно быть > 0")
        total = shares * price + costs.total
        if total > self.cash + 1e-9:
            raise PortfolioError(
                f"Покупка {secid} на {total:.2f} ₽ при кэше {self.cash:.2f} ₽ — "
                f"проверка достаточности денег должна была сработать раньше."
            )
        realized = None
        pos = self.positions.get(secid)
        if pos is not None and pos.shares < 0:
            # Закрытие шорта (возможно частичное), остаток — в лонг.
            covered = min(shares, -pos.shares)
            realized = (pos.avg_cost - price) * covered
            self._add_realized(realized, day)
            pos.shares += covered
            remaining = shares - covered
            if pos.shares == 0:
                del self.positions[secid]
            if remaining > 0:
                self.positions[secid] = Position(shares=remaining, avg_cost=price)
        elif pos is None:
            self.positions[secid] = Position(shares=shares, avg_cost=price)
        else:
            new_shares = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + shares * price) / new_shares
            pos.shares = new_shares
        self.cash -= total
        self.total_costs_paid += costs.total
        return realized

    def sell(self, secid: str, shares: int, price: float, costs: TradeCosts,
             day: Date) -> float | None:
        """Продажа: закрытие лонга и/или открытие/наращивание шорта.

        Возвращает зафиксированную прибыль, если продажа закрывала лонг."""
        if shares <= 0:
            raise PortfolioError(f"Продажа {secid}: число акций должно быть > 0")
        pos = self.positions.get(secid)
        held = pos.shares if pos else 0
        short_part = shares - max(held, 0)
        if short_part > 0 and not self.allow_short:
            raise PortfolioError(
                f"Продажа {secid}: запрошено {shares} шт, в портфеле {held} шт. "
                f"Короткие продажи запрещены (allow_short=False)."
            )
        realized = None
        close_part = min(shares, held) if held > 0 else 0
        if close_part:
            realized = (price - pos.avg_cost) * close_part
            self._add_realized(realized, day)
            pos.shares -= close_part
            if pos.shares == 0:
                del self.positions[secid]
        if short_part > 0:
            pos = self.positions.get(secid)
            if pos is None:
                self.positions[secid] = Position(shares=-short_part, avg_cost=price)
            else:  # pos.shares < 0 — наращиваем шорт, усредняем цену открытия
                total_short = -pos.shares + short_part
                pos.avg_cost = (
                    (-pos.shares) * pos.avg_cost + short_part * price
                ) / total_short
                pos.shares = -total_short
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
