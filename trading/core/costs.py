"""МОДЕЛЬ ИЗДЕРЖЕК — ядро системы.

Бэктест без честной модели издержек покажет прибыльной любую стратегию.
Каждая сделка платит:

* комиссию брокера (% от оборота, но не меньше минимума в рублях);
* комиссию биржи (% от оборота);
* половину спреда (% от оборота, на каждой стороне сделки);
* проскальзывание, растущее с размером заявки относительно ликвидности:

      slippage_pct = base_slippage + impact_coefficient * (order_value / adv_20)

  где adv_20 — средний дневной оборот бумаги в рублях за 20 дней.

Заявка больше 1% от adv_20 — предупреждение в лог. Больше 5% — отклонение:
бэктест, который «покупает» недоступный на рынке объём, бесполезен.

Стресс-режим: множитель >= 1 умножает все издержки. Стратегия, умирающая
при удвоении издержек, не имеет запаса прочности.

НДФЛ 13% с зафиксированной прибыли считается в годовом разрезе отдельно.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading.formatting import fmt_rub
from trading.logging_setup import get_logger
from trading.settings import CostConfig

log = get_logger("costs")


class UnfillableOrderError(Exception):
    """Заявка нереализуема: объём превышает допустимую долю дневного оборота."""


@dataclass(frozen=True)
class TradeCosts:
    """Разбивка издержек одной сделки, все суммы в рублях."""

    order_value: float
    broker_commission: float
    exchange_fee: float
    half_spread: float
    slippage: float
    slippage_pct: float          # проскальзывание в % — для журнала
    adv_share: float             # доля заявки от среднего дневного оборота

    @property
    def total(self) -> float:
        return (
            self.broker_commission + self.exchange_fee
            + self.half_spread + self.slippage
        )

    def explain_ru(self) -> str:
        """Одна строка на русском — проверяется калькулятором за 30 секунд."""
        return (
            f"Сделка {fmt_rub(self.order_value)} ₽: комиссия брокера "
            f"{fmt_rub(self.broker_commission)} ₽ + биржа {fmt_rub(self.exchange_fee)} ₽ "
            f"+ полспреда {fmt_rub(self.half_spread)} ₽ + проскальзывание "
            f"{fmt_rub(self.slippage)} ₽ ({self.slippage_pct:.3f}%) "
            f"= итого {fmt_rub(self.total)} ₽"
        )


class CostModel:
    def __init__(self, config: CostConfig):
        self.cfg = config

    @property
    def stress(self) -> float:
        return self.cfg.stress_multiplier

    def slippage_pct(self, order_value: float, adv_20: float) -> float:
        """Проскальзывание в процентах от оборота (до стресс-множителя)."""
        if adv_20 <= 0:
            raise UnfillableOrderError(
                "Средний дневной оборот бумаги неизвестен или равен нулю — "
                "оценить проскальзывание невозможно, сделка отклонена."
            )
        return (
            self.cfg.base_slippage_pct
            + self.cfg.impact_coefficient * (order_value / adv_20)
        )

    def trade_costs(self, order_value: float, adv_20: float) -> TradeCosts:
        """Полные издержки сделки. Отклоняет нереализуемые заявки.

        ``order_value`` — сумма сделки в рублях,
        ``adv_20`` — средний дневной оборот бумаги в рублях за 20 дней.
        """
        if order_value <= 0:
            raise ValueError(f"Сумма сделки должна быть больше нуля: {order_value}")
        if adv_20 <= 0:
            raise UnfillableOrderError(
                "Средний дневной оборот бумаги неизвестен или равен нулю — "
                "сделка отклонена."
            )

        adv_share = order_value / adv_20
        if adv_share > self.cfg.adv_reject_share:
            raise UnfillableOrderError(
                f"Заявка {order_value:.0f} ₽ — это {adv_share:.1%} дневного "
                f"оборота бумаги ({adv_20:.0f} ₽). Порог отказа "
                f"{self.cfg.adv_reject_share:.0%}: такой объём рынок не даст "
                f"купить по расчётной цене, сделка отклонена."
            )
        if adv_share > self.cfg.adv_warning_share:
            log.warning(
                "крупная заявка относительно ликвидности",
                доля_оборота=f"{adv_share:.2%}",
                сумма_заявки=round(order_value, 2),
                дневной_оборот=round(adv_20, 2),
            )

        s = self.stress
        broker = max(
            order_value * self.cfg.broker_commission_pct / 100,
            self.cfg.broker_commission_min,
        ) * s
        exchange = order_value * self.cfg.exchange_fee_pct / 100 * s
        half_spread = order_value * self.cfg.half_spread_pct / 100 * s
        slip_pct = self.slippage_pct(order_value, adv_20) * s
        slippage = order_value * slip_pct / 100

        return TradeCosts(
            order_value=order_value,
            broker_commission=broker,
            exchange_fee=exchange,
            half_spread=half_spread,
            slippage=slippage,
            slippage_pct=slip_pct,
            adv_share=adv_share,
        )

    def ndfl(self, realized_profit_by_year: dict[int, float]) -> dict[int, float]:
        """НДФЛ по годам: 13% с положительной зафиксированной прибыли.

        Убыточный год налога не создаёт (перенос убытков между годами
        здесь консервативно не учитывается — реальный налог может быть
        только меньше расчётного, не больше).
        """
        return {
            year: max(profit, 0.0) * self.cfg.ndfl_rate
            for year, profit in realized_profit_by_year.items()
        }
