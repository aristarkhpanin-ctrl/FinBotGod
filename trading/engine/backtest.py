"""Движок бэктеста — ФАЗА 4.

Дисциплина времени (ТЗ, раздел 7) — жёсткое правило:
* сигнал рассчитывается по данным, доступным на закрытии дня T;
* исполнение — по цене ОТКРЫТИЯ дня T+1 с издержками и проскальзыванием;
* стратегия получает срез данных по день T включительно — доступа к
  будущему у неё нет архитектурно (движок не передаёт полный DataFrame).

Параметр ``signal_shift_days`` сдвигает сигнал на N дней назад
относительно исполнения — он существует ТОЛЬКО для обязательного теста
на заглядывание в будущее: при сдвиге доходность обязана заметно
ухудшаться, иначе в системе утечка будущего.

Каждый прогон автоматически записывается в журнал гипотез. Прогона
в обход журнала не существует: без журнала движок не создаётся.

Известные ограничения (печатаются в отчёте):
* лотность берётся сегодняшняя — исторические изменения лотов ISS не отдаёт;
* лимит числа позиций проверяется здесь же; независимый риск-слой — фаза 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.core.costs import CostModel, UnfillableOrderError
from trading.core.portfolio import Portfolio
from trading.core.sizing import resolution_report_ru
from trading.execution.simulated import Fill, SimulatedExecution
from trading.formatting import fmt_rub
from trading.logging_setup import get_logger
from trading.reporting.metrics import (
    buy_and_hold_benchmark,
    compute_metrics,
    money_market_benchmark,
)
from trading.research.ledger import HypothesisLedger
from trading.risk.guards import RiskGuards, SystemHalted
from trading.settings import Settings

log = get_logger("backtest")


@dataclass
class BacktestResult:
    equity: pd.Series            # стоимость портфеля по дням (на закрытии)
    fills: list[Fill]
    decisions: list[str]         # человекочитаемые решения
    metrics: dict
    benchmarks: dict
    ndfl_by_year: dict[int, float]
    resolution_report: str
    limitations: list[str]
    run_hash: str
    guard_events: list[str]      # сработавшие предохранители
    halted_reason: str | None    # система остановлена предохранителем
    report_ru: str = ""
    monthly_report_ru: str = ""


class BacktestEngine:
    def __init__(
        self,
        candles: dict[str, pd.DataFrame],
        lot_sizes: dict[str, int],
        settings: Settings,
        strategy,
        ledger: HypothesisLedger,
        data_hash: str,
        imoex_close: pd.Series | None = None,
        signal_shift_days: int = 0,
        source: str = "человек",
        whitelist: list[str] | None = None,
        journal=None,               # DecisionJournal — файлы decisions.log/events.jsonl
        trade_from: str | None = None,
        # trade_from: сделки только с этой даты; данные ДО неё стратегия
        # видит (это прошлое — для расчёта сигналов), но не торгует.
        # Метрики считаются с trade_from: разогрев не разбавляет доходность.
        allow_short: bool = False,
        # allow_short: разрешить короткие позиции. Плата за заём бумаг
        # (издержки.ставка_займа_шорт_годовых) начисляется ежедневно.
    ):
        if ledger is None:
            raise ValueError(
                "Бэктест без журнала гипотез запрещён (ТЗ, раздел 16)."
            )
        self.settings = settings
        self.strategy = strategy
        self.ledger = ledger
        self.data_hash = data_hash
        self.imoex_close = imoex_close
        self.signal_shift = int(signal_shift_days)
        self.source = source
        self.journal = journal
        self.trade_from = pd.Timestamp(trade_from) if trade_from else None
        self.allow_short = allow_short
        self.lot_sizes = lot_sizes
        self.cost_model = CostModel(settings.costs)

        self.candles: dict[str, pd.DataFrame] = {}
        for secid, df in candles.items():
            if df is None or df.empty:
                continue
            d = df.sort_values("date").reset_index(drop=True).copy()
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
            d["adv20"] = d["value"].rolling(20, min_periods=1).mean()
            self.candles[secid] = d
        if not self.candles:
            raise ValueError("Бэктест без данных невозможен: нет ни одной свечи.")

        # Независимый риск-слой: белый список по умолчанию — все бумаги,
        # по которым есть данные (в боевом контуре — universe.yaml).
        self.guards = RiskGuards(
            settings.risk,
            set(whitelist) if whitelist is not None else set(self.candles),
        )

        all_dates = sorted(set().union(*(set(d["date"]) for d in self.candles.values())))
        self.dates = pd.DatetimeIndex(all_dates)
        self._date_arrays = {s: d["date"].values for s, d in self.candles.items()}
        self.opens = pd.DataFrame(
            {s: d.set_index("date")["open"] for s, d in self.candles.items()},
            index=self.dates,
        )
        closes = pd.DataFrame(
            {s: d.set_index("date")["close"] for s, d in self.candles.items()},
            index=self.dates,
        )
        self.closes_ffill = closes.ffill()
        self.adv20 = pd.DataFrame(
            {s: d.set_index("date")["adv20"] for s, d in self.candles.items()},
            index=self.dates,
        ).ffill()

    # ---------- Вспомогательное ----------

    def _slices_until(self, t: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Срезы данных по день T включительно — всё, что видит стратегия."""
        out = {}
        for secid, d in self.candles.items():
            n = int(np.searchsorted(self._date_arrays[secid], t.to_datetime64(), side="right"))
            if n > 0:
                out[secid] = d.iloc[:n]
        return out

    def _valuation_prices(self, exec_idx: int, opens_row: pd.Series) -> dict[str, float]:
        """Цены для оценки портфеля при сайзинге: open дня исполнения,
        а для бумаг без торгов — последний известный close ДО этого дня
        (не сегодняшний close: он ещё не известен на открытии)."""
        prev = self.closes_ffill.iloc[exec_idx - 1] if exec_idx > 0 else opens_row
        prices = {}
        for secid in self.candles:
            o = opens_row.get(secid)
            if o is not None and not math.isnan(o):
                prices[secid] = float(o)
            else:
                p = prev.get(secid)
                if p is not None and not math.isnan(p):
                    prices[secid] = float(p)
        return prices

    # ---------- Основной цикл ----------

    def run(self) -> BacktestResult:
        s = self.settings
        portfolio = Portfolio(s.capital.start_amount, allow_short=self.allow_short)
        execution = SimulatedExecution(portfolio, self.cost_model)
        decisions: list[str] = []
        limitations = [
            "Лотность взята сегодняшняя: исторические изменения размеров "
            "лотов ISS не отдаёт, до 2020 года возможна погрешность сайзинга.",
        ]

        lot_costs = {}
        for secid, d in self.candles.items():
            lot = self.lot_sizes.get(secid)
            if lot and lot > 0:
                lot_costs[secid] = float(d["close"].iloc[0]) * lot
        resolution = resolution_report_ru(
            capital=s.capital.start_amount,
            lot_costs=lot_costs,
            max_positions=s.risk.max_positions,
            max_position_pct=s.risk.max_position_pct,
        )

        guard_events: list[str] = []
        halted_reason: str | None = None
        equity_values: list[float] = []
        decisions_flushed = 0          # сколько решений уже ушло в журнал
        costs_flushed = 0.0            # издержки, учтённые в прошлых сводках
        for i, day in enumerate(self.dates):
            trading_allowed = (
                halted_reason is None
                and not (self.guards.halted_forever and
                         self.guards.liquidation_pending is None)
            )
            if self.trade_from is not None and day < self.trade_from:
                trading_allowed = False   # разогрев: сигналы есть, сделок нет
            if i >= 1 and trading_allowed:
                self.guards.new_day(day.date())
                try:
                    if self.guards.liquidation_pending:
                        self._liquidate_all(i, day, portfolio, execution, decisions)
                    else:
                        signal_idx = i - 1 - self.signal_shift
                        if signal_idx >= 0:
                            t = self.dates[signal_idx]
                            weights = self.strategy.target_weights(
                                self._slices_until(t)
                            )
                            self._execute_day(
                                i, day, t, weights, portfolio, execution, decisions
                            )
                except SystemHalted as e:
                    halted_reason = str(e)
                    guard_events.append(f"{day.date()}  {halted_reason}")
                    decisions.append(f"{day.date().isoformat()}  ОСТАНОВКА  {halted_reason}")
            row = self.closes_ffill.iloc[i]
            prices = {sec: float(row[sec]) for sec in portfolio.positions}
            # Плата за заём бумаг по коротким позициям — каждый торговый день.
            if self.allow_short:
                short_value = portfolio.short_value(prices)
                if short_value > 0:
                    portfolio.charge_borrow_fee(
                        short_value * s.costs.short_borrow_rate / 252
                    )
            equity_value = portfolio.equity(prices)
            equity_values.append(equity_value)
            # Проверка предохранителей по фактической стоимости портфеля.
            if halted_reason is None:
                verdict = self.guards.check_equity(day.date(), equity_value)
                if verdict:
                    guard_events.append(f"{day.date()}  {verdict}")
                    decisions.append(f"{day.date().isoformat()}  {verdict}")
            # Журнал: решения дня + обязательная ежедневная сводка.
            if self.journal is not None:
                for text in decisions[decisions_flushed:]:
                    self.journal.log_decision(text, тип="решение")
                decisions_flushed = len(decisions)
                prev_equity = (
                    equity_values[-2] if len(equity_values) > 1
                    else s.capital.start_amount
                )
                self.journal.daily_summary(
                    day=day.date(),
                    equity=equity_value,
                    prev_equity=prev_equity,
                    costs_today=portfolio.total_costs_paid - costs_flushed,
                    costs_total=portfolio.total_costs_paid,
                    start_capital=s.capital.start_amount,
                    risk_free_rate=s.benchmark.risk_free_rate,
                    n_positions=len(portfolio.positions),
                    max_positions=s.risk.max_positions,
                )
                costs_flushed = portfolio.total_costs_paid

        equity = pd.Series(equity_values, index=self.dates, name="equity")
        if self.trade_from is not None:
            equity = equity[equity.index >= self.trade_from]
        metrics = compute_metrics(
            equity, s.benchmark.risk_free_rate, fills=execution.fills
        )
        if portfolio.total_borrow_fees > 0:
            # Плата за заём — такие же издержки, как комиссии и спред.
            metrics["borrow_fees"] = portfolio.total_borrow_fees
            metrics["total_costs"] = (
                metrics.get("total_costs", 0.0) + portfolio.total_borrow_fees
            )
        days_span = (equity.index[-1] - equity.index[0]).days
        benchmarks = {
            "денежный_рынок": money_market_benchmark(
                s.capital.start_amount, days_span, s.benchmark.risk_free_rate
            ),
        }
        if self.imoex_close is not None and len(self.imoex_close) >= 2:
            benchmarks["IMOEX"] = buy_and_hold_benchmark(self.imoex_close)
        else:
            benchmarks["IMOEX"] = None
            limitations.append(
                "Данные индекса IMOEX за период недоступны — бенчмарк "
                "«купил и держи индекс» не рассчитан."
            )
        ndfl = self.cost_model.ndfl(portfolio.realized_pnl_by_year)

        run_hash = self.ledger.record(
            hypothesis=self.strategy.explain_ru(),
            source=self.source,
            params={
                **self.strategy.params(),
                "signal_shift_days": self.signal_shift,
                "stress_multiplier": s.costs.stress_multiplier,
                "capital": s.capital.start_amount,
                "allow_short": self.allow_short,
            },
            data_hash=self.data_hash,
            metrics_is={
                k: metrics.get(k)
                for k in ("annual_return", "total_return", "sharpe",
                          "max_drawdown", "n_trades", "total_costs")
            },
        )

        if self.guards.halted_forever and halted_reason is None:
            halted_reason = self.guards.halted_forever
        result = BacktestResult(
            equity=equity, fills=execution.fills, decisions=decisions,
            metrics=metrics, benchmarks=benchmarks, ndfl_by_year=ndfl,
            resolution_report=resolution, limitations=limitations,
            run_hash=run_hash, guard_events=guard_events,
            halted_reason=halted_reason,
        )
        from trading.reporting.human_log import monthly_report
        from trading.reporting.report import full_report_ru

        result.report_ru = full_report_ru(s, result)
        monthly_text, monthly_table = monthly_report(
            equity, execution.fills, s.benchmark.risk_free_rate, guard_events
        )
        result.monthly_report_ru = monthly_text
        if self.journal is not None:
            self.journal.write_monthly_csv(monthly_table)
        return result

    # ---------- Ликвидация по предохранителю ----------

    def _liquidate_all(self, exec_idx, day, portfolio, execution, decisions):
        """Продажа всех позиций по open — после срабатывания дневного
        лимита убытка или лимита просадки. Позиции без торгов в этот день
        остаются до следующего дня (только лимитные заявки, чудес нет)."""
        kind = self.guards.liquidation_pending
        opens_row = self.opens.iloc[exec_idx]
        t = self.dates[exec_idx - 1]
        day_str = day.date().isoformat()
        prices = self._valuation_prices(exec_idx, opens_row)
        pv = portfolio.equity({sec: prices[sec] for sec in portfolio.positions})
        for secid in sorted(portfolio.positions):
            o = opens_row.get(secid)
            if o is None or math.isnan(o):
                decisions.append(
                    f"{day_str}  ЛИКВИДАЦИЯ ОТЛОЖЕНА  {secid}: нет торгов"
                )
                continue
            price = float(o)
            held = portfolio.shares_of(secid)
            side = "sell" if held > 0 else "buy"   # шорт закрывается покупкой
            shares = abs(held)
            lot = self.lot_sizes.get(secid, 1)
            if side == "buy":
                # Закрытие шорта требует денег — при нехватке частично.
                lots = shares // lot
                while lots > 0:
                    order_value = lots * lot * price
                    try:
                        cost_est = self.cost_model.trade_costs(
                            order_value, float(self.adv20.loc[t, secid])
                        ).total
                    except UnfillableOrderError:
                        break
                    if order_value + cost_est <= portfolio.cash:
                        break
                    lots -= 1
                if lots == 0:
                    decisions.append(
                        f"{day_str}  ЛИКВИДАЦИЯ ОТЛОЖЕНА  {secid}: не хватает "
                        f"денег на закрытие шорта"
                    )
                    continue
                shares = lots * lot
            check = self.guards.check_order(
                day.date(), secid, side, shares * price, pv,
                position_value=abs(held) * price,
                n_positions=len(portfolio.positions),
                is_new_position=False,
                increases_risk=False,
            )
            if not check.allowed:
                decisions.append(f"{day_str}  ОТКАЗ  {secid}: {check.reason_ru}")
                continue
            adv = float(self.adv20.loc[t, secid])
            try:
                fill = execution.execute(day.date(), secid, side, shares, price, adv)
            except UnfillableOrderError as e:
                decisions.append(
                    f"{day_str}  ЛИКВИДАЦИЯ ОТЛОЖЕНА  {secid}: {e}"
                )
                continue
            decisions.append(
                f"{day_str}  ЛИКВИДАЦИЯ ({kind})  {secid}  {shares} шт "
                f"по {fmt_rub(price)} ₽ = {fmt_rub(fill.order_value)} ₽ | "
                f"издержки {fmt_rub(fill.costs.total)} ₽"
            )
        if not portfolio.positions:
            self.guards.liquidation_done(day.date())

    # ---------- Исполнение одного дня ----------

    def _execute_day(self, exec_idx, day, t, weights, portfolio, execution, decisions):
        s = self.settings
        opens_row = self.opens.iloc[exec_idx]
        prices = self._valuation_prices(exec_idx, opens_row)
        pv = portfolio.equity({sec: prices[sec] for sec in portfolio.positions})
        day_str = day.date().isoformat()

        targets = {sec: w for sec, w in (weights or {}).items() if w != 0}
        involved = sorted(set(targets) | set(portfolio.positions))
        plans = []
        for secid in involved:
            lot = self.lot_sizes.get(secid)
            if not lot or lot <= 0:
                decisions.append(f"{day_str}  ПРОПУСК  {secid}: лотность неизвестна")
                continue
            o = opens_row.get(secid)
            held_shares = portfolio.shares_of(secid)
            weight = targets.get(secid, 0.0)
            if o is None or math.isnan(o):
                if held_shares != 0 or weight != 0:
                    decisions.append(
                        f"{day_str}  ОТЛОЖЕНО  {secid}: нет торгов в этот день"
                    )
                continue
            if weight < 0 and not self.allow_short:
                decisions.append(
                    f"{day_str}  ОТКАЗ  {secid}: стратегия просит шорт, "
                    f"но короткие позиции выключены (allow_short=False)"
                )
                weight = 0.0
            price = float(o)
            weight_capped = (
                math.copysign(min(abs(weight), s.risk.max_position_pct), weight)
                if weight else 0.0
            )
            lot_cost = price * lot
            held_lots = int(held_shares / lot)   # знак сохраняется (шорт < 0)
            raw = pv * weight_capped / lot_cost if weight_capped else 0.0
            # Новая позиция — усечение к нулю, т.е. floor по модулю (ТЗ, раздел 6).
            # Корректировка существующей — к ближайшему лоту: иначе издержки чуть
            # уменьшают портфель и движок бесконечно гоняет один лот туда-сюда.
            if held_lots == 0:
                desired = int(raw)
            else:
                desired = int(raw + math.copysign(0.5, raw)) if raw else 0
            if weight_capped != 0 and desired == 0 and held_lots == 0:
                decisions.append(
                    f"{day_str}  ПРОПУСК  {secid}: целевая сумма "
                    f"{fmt_rub(abs(pv * weight_capped), 0)} ₽ меньше стоимости "
                    f"одного лота {fmt_rub(lot_cost, 0)} ₽"
                )
                continue
            delta = desired - held_lots
            if delta != 0:
                plans.append((secid, held_lots, desired, delta, lot, price, weight))

        # Порядок исполнения: сначала заявки, снижающие риск (продажа лонга,
        # закрытие шорта) — они освобождают деньги и лимиты; затем заявки,
        # увеличивающие риск. Внутри групп продажи раньше покупок (дают кэш).
        def order_key(plan):
            secid, held_lots, desired, delta, *_ = plan
            increases = abs(desired) > abs(held_lots)
            return (increases, delta > 0, secid)

        for secid, held_lots, desired, delta, lot, price, weight in sorted(
            plans, key=order_key
        ):
            increases = abs(desired) > abs(held_lots)
            adv = float(self.adv20.loc[t, secid])
            lot_cost = lot * price
            held_value = abs(portfolio.shares_of(secid)) * price
            if delta < 0:
                # Продажа: снижение лонга и/или открытие (наращивание) шорта.
                lots = -delta
                exposure_note = False
                if increases:
                    # Совокупная шорт-экспозиция не больше стоимости портфеля.
                    room = pv - portfolio.short_value(prices)
                    lots_by_room = max(int(room / lot_cost), 0)
                    if lots > lots_by_room:
                        lots, exposure_note = lots_by_room, True
                    if lots == 0:
                        decisions.append(
                            f"{day_str}  ПРОПУСК  {secid}: лимит совокупной "
                            f"шорт-экспозиции (100% портфеля) исчерпан"
                        )
                        continue
                check = self.guards.check_order(
                    day.date(), secid, "sell", lots * lot_cost, pv,
                    position_value=held_value,
                    n_positions=len(portfolio.positions),
                    is_new_position=held_lots == 0,
                    increases_risk=increases,
                )
                if not check.allowed:
                    decisions.append(f"{day_str}  ОТКАЗ  {secid}: {check.reason_ru}")
                    continue
                if check.max_value is not None:
                    lots = min(lots, int(check.max_value / lot_cost))
                    decisions.append(f"{day_str}  РИСК-СЛОЙ  {check.reason_ru}")
                    if lots == 0:
                        continue
                shares = lots * lot
                try:
                    fill = execution.execute(
                        day.date(), secid, "sell", shares, price, adv
                    )
                except UnfillableOrderError as e:
                    decisions.append(f"{day_str}  ОТКАЗ  {secid} (продажа): {e}")
                    continue
                action = "ШОРТ" if increases else "ПРОДАЖА"
                realized_note = (
                    f" | зафиксировано {fill.realized_pnl:+.2f} ₽"
                    if fill.realized_pnl is not None else ""
                )
                decisions.append(
                    f"{day_str}  {action}  {secid}  {lots} лот = {shares} шт "
                    f"по {fmt_rub(price)} ₽ = {fmt_rub(fill.order_value)} ₽ | "
                    f"издержки {fmt_rub(fill.costs.total)} ₽{realized_note} | "
                    f"причина: целевая доля {weight:.0%}"
                    + (" | обрезано лимитом шорт-экспозиции" if exposure_note else "")
                )
            else:
                # Покупка: закрытие шорта и/или открытие (наращивание) лонга.
                lots = delta
                liquidity_note = money_note = False
                while lots > 0:
                    order_value = lots * lot_cost
                    try:
                        costs = self.cost_model.trade_costs(order_value, adv)
                    except UnfillableOrderError:
                        liquidity_note = True
                        lots -= 1
                        continue
                    if order_value + costs.total <= portfolio.cash:
                        break
                    money_note = True
                    lots -= 1
                if lots == 0:
                    why = ("ликвидности не хватает" if liquidity_note
                           else "не хватает денег с учётом издержек")
                    decisions.append(
                        f"{day_str}  ПРОПУСК  {secid}: {why} "
                        f"(свободно {fmt_rub(portfolio.cash, 0)} ₽)"
                    )
                    continue
                check = self.guards.check_order(
                    day.date(), secid, "buy", lots * lot_cost, pv,
                    position_value=held_value,
                    n_positions=len(portfolio.positions),
                    is_new_position=held_lots == 0,
                    increases_risk=increases,
                )
                if not check.allowed:
                    decisions.append(f"{day_str}  ОТКАЗ  {secid}: {check.reason_ru}")
                    continue
                if check.max_value is not None:
                    lots = min(lots, int(check.max_value / lot_cost))
                    decisions.append(f"{day_str}  РИСК-СЛОЙ  {check.reason_ru}")
                    if lots == 0:
                        continue
                shares = lots * lot
                fill = execution.execute(day.date(), secid, "buy", shares, price, adv)
                note = ""
                if liquidity_note:
                    note = " | заявка обрезана по ликвидности"
                elif money_note:
                    note = " | заявка обрезана по деньгам"
                action = "ЗАКРЫТИЕ ШОРТА" if held_lots < 0 else "ПОКУПКА"
                realized_note = (
                    f" | зафиксировано {fill.realized_pnl:+.2f} ₽"
                    if fill.realized_pnl is not None else ""
                )
                decisions.append(
                    f"{day_str}  {action}  {secid}  {lots} лот = {shares} шт "
                    f"по {fmt_rub(price)} ₽ = {fmt_rub(fill.order_value)} ₽ | "
                    f"издержки {fmt_rub(fill.costs.total)} ₽{realized_note} | "
                    f"кэш после: {fmt_rub(portfolio.cash, 0)} ₽ | "
                    f"причина: целевая доля {weight:.0%}{note}"
                )
