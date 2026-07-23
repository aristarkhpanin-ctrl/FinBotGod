"""НЕЗАВИСИМЫЙ риск-слой — ФАЗА 6 (ТЗ, раздел 9).

Последняя линия обороны. Не зависит от логики стратегии и работает,
даже если стратегия содержит грубые ошибки.

Предохранители:

| Предохранитель            | Действие                                     |
|---------------------------|----------------------------------------------|
| max_position_pct          | Обрезать заявку                              |
| max_positions             | Отклонить новую покупку                      |
| whitelist                 | Отклонить, записать в лог как ИНЦИДЕНТ       |
| max_order_value           | ОСТАНОВИТЬ СИСТЕМУ (баг в расчёте единиц)    |
| max_orders_per_day        | ОСТАНОВИТЬ СИСТЕМУ (зацикливание)            |
| daily_loss_limit          | Закрыть позиции, стоп до следующего дня      |
| max_drawdown_from_peak    | Закрыть всё, остановиться навсегда           |

Два последних работают по ФАКТИЧЕСКОЙ стоимости портфеля, переданной
извне (у брокера в боевом контуре, у симулятора в бэктесте), а не по
внутреннему расчёту стратегии — внутренний расчёт может быть неверным,
в этом весь смысл.

Идемпотентность заявок: детерминированные client_order_id создаёт слой
исполнения; сверка состояния с брокером при переподключении — фаза 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

from trading.formatting import fmt_rub
from trading.logging_setup import get_logger
from trading.settings import RiskConfig

log = get_logger("risk")


class SystemHalted(Exception):
    """Срабатывание предохранителя, требующее полной остановки системы."""


@dataclass(frozen=True)
class OrderCheck:
    allowed: bool
    max_value: float | None      # если заявку нужно обрезать — потолок в рублях
    reason_ru: str


class RiskGuards:
    def __init__(self, config: RiskConfig, whitelist: set[str]):
        self.cfg = config
        self.whitelist = set(whitelist)
        self._current_day: Date | None = None
        self._orders_today = 0
        self._peak_equity: float | None = None
        self._prev_close_equity: float | None = None
        self._liquidation_pending: str | None = None   # "день" | "навсегда"
        self._halted_forever_reason: str | None = None
        self._buys_blocked_on: Date | None = None

    # ---------- Состояние ----------

    @property
    def halted_forever(self) -> str | None:
        return self._halted_forever_reason

    @property
    def liquidation_pending(self) -> str | None:
        return self._liquidation_pending

    def liquidation_done(self, day: Date) -> None:
        if self._liquidation_pending == "день":
            self._buys_blocked_on = day     # день ликвидации — без покупок
        self._liquidation_pending = None

    def new_day(self, day: Date) -> None:
        if self._current_day != day:
            self._current_day = day
            self._orders_today = 0

    # ---------- Проверка заявки ----------

    def check_order(
        self,
        day: Date,
        secid: str,
        side: str,                    # "buy" | "sell"
        order_value: float,
        portfolio_value: float,
        position_value: float,        # |стоимость| позиции по этой бумаге
        n_positions: int,
        is_new_position: bool,
        increases_risk: bool | None = None,
    ) -> OrderCheck:
        """Проверяет одну заявку.

        ``increases_risk`` — увеличивает ли заявка |позицию| (покупка лонга
        или ОТКРЫТИЕ ШОРТА). Заявки, уменьшающие риск (продажа лонга,
        закрытие шорта), ограничены только «стоп-системой»: выйти из
        позиции можно всегда, кроме полного отказа системы. Если параметр
        не задан, риск-увеличение выводится из направления: buy = да."""
        self.new_day(day)
        if increases_risk is None:
            increases_risk = side == "buy"

        if self._halted_forever_reason:
            final_liquidation = (
                not increases_risk and self._liquidation_pending == "навсегда"
            )
            if not final_liquidation:
                return OrderCheck(False, None, (
                    f"Система остановлена навсегда: {self._halted_forever_reason} "
                    f"Требуется ручной перезапуск."
                ))

        # Потолок суммы заявки: превышение = баг в расчёте единиц.
        if order_value > self.cfg.max_order_value:
            raise SystemHalted(
                f"Заявка {secid} на {fmt_rub(order_value, 0)} ₽ превышает "
                f"потолок {fmt_rub(self.cfg.max_order_value, 0)} ₽. Это признак "
                f"бага в расчёте единиц (рубли/копейки/штуки/лоты). "
                f"СИСТЕМА ОСТАНОВЛЕНА."
            )

        # Потолок числа заявок в день: превышение = зацикливание.
        if self._orders_today + 1 > self.cfg.max_orders_per_day:
            raise SystemHalted(
                f"Попытка отправить заявку №{self._orders_today + 1} за день "
                f"при лимите {self.cfg.max_orders_per_day}. Это признак "
                f"зацикливания. СИСТЕМА ОСТАНОВЛЕНА."
            )

        if not increases_risk:
            self._orders_today += 1
            return OrderCheck(True, None, "заявка на снижение риска разрешена")

        # --- Дальше только заявки, УВЕЛИЧИВАЮЩИЕ риск (лонг или шорт) ---

        if self._buys_blocked_on == day:
            return OrderCheck(False, None, (
                "Новые позиции запрещены: сегодня день остановки после "
                "срабатывания дневного лимита убытка."
            ))

        if secid not in self.whitelist:
            reason = (
                f"ИНЦИДЕНТ: попытка открыть позицию {secid} вне белого списка. "
                f"Заявка отклонена. Стратегия не должна была даже предлагать "
                f"эту бумагу."
            )
            log.error(reason)
            return OrderCheck(False, None, reason)

        if is_new_position and n_positions >= self.cfg.max_positions:
            return OrderCheck(False, None, (
                f"Отклонено: лимит числа позиций "
                f"({self.cfg.max_positions}) исчерпан."
            ))

        # Доля одной позиции: обрезание, не отказ.
        position_limit = portfolio_value * self.cfg.max_position_pct
        allowed_value = position_limit - position_value
        if allowed_value <= 0:
            return OrderCheck(False, None, (
                f"Отклонено: позиция {secid} уже занимает лимит "
                f"{self.cfg.max_position_pct:.0%} портфеля."
            ))
        if order_value > allowed_value + 1e-9:
            self._orders_today += 1
            return OrderCheck(True, allowed_value, (
                f"Заявка {secid} обрезана риск-слоем: {fmt_rub(order_value, 0)} ₽ "
                f"→ {fmt_rub(allowed_value, 0)} ₽ (лимит "
                f"{self.cfg.max_position_pct:.0%} на позицию)."
            ))

        self._orders_today += 1
        return OrderCheck(True, None, "покупка разрешена")

    # ---------- Проверка портфеля (по фактической стоимости) ----------

    def check_equity(self, day: Date, actual_equity: float) -> str | None:
        """Вызывается после закрытия каждого дня с ФАКТИЧЕСКОЙ стоимостью
        портфеля. Возвращает текст сработавшего предохранителя или None."""
        if self._halted_forever_reason:
            return None

        if self._peak_equity is None or actual_equity > self._peak_equity:
            self._peak_equity = actual_equity

        drawdown = actual_equity / self._peak_equity - 1
        if drawdown < -self.cfg.max_drawdown_from_peak:
            reason = (
                f"ПРЕДОХРАНИТЕЛЬ max_drawdown_from_peak: просадка {drawdown:.1%} "
                f"от пика {fmt_rub(self._peak_equity, 0)} ₽ превысила лимит "
                f"{self.cfg.max_drawdown_from_peak:.0%}. Закрыть всё, уйти в кэш, "
                f"ОСТАНОВИТЬСЯ НАВСЕГДА до ручного перезапуска."
            )
            self._liquidation_pending = "навсегда"
            self._halted_forever_reason = reason
            self._prev_close_equity = actual_equity
            log.error(reason)
            return reason

        if self._prev_close_equity is not None:
            daily_change = actual_equity / self._prev_close_equity - 1
            if daily_change < -self.cfg.daily_loss_limit:
                reason = (
                    f"ПРЕДОХРАНИТЕЛЬ daily_loss_limit: убыток {daily_change:.1%} "
                    f"за день превысил лимит {self.cfg.daily_loss_limit:.0%}. "
                    f"Закрыть позиции, остановиться до следующего дня."
                )
                self._liquidation_pending = "день"
                self._prev_close_equity = actual_equity
                log.warning(reason)
                return reason

        self._prev_close_equity = actual_equity
        return None
