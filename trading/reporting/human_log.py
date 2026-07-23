"""Человекочитаемый журнал решений — ФАЗА 7 (ТЗ, раздел 10).

Два параллельных потока:

1. ``decisions.log`` — по одной записи на решение, на русском языке,
   проверяется калькулятором;
2. ``events.jsonl`` — структурированный JSON для анализа.

Ежедневная сводка ИТОГИ ДНЯ всегда заканчивается сравнением с фондом
денежного рынка: заказчик видит, обгоняет ли он безрисковую
альтернативу, без дополнительных запросов.

Ежемесячный отчёт — автоматически, текстом и CSV: доходность, издержки,
число сделок, сравнение с бенчмарком, сработавшие предохранители.
"""

from __future__ import annotations

import json
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading.formatting import fmt_rub


class DecisionJournal:
    def __init__(self, log_dir: str | Path = "logs"):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._decisions = open(self.dir / "decisions.log", "a", encoding="utf-8")
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")

    def close(self) -> None:
        self._decisions.close()
        self._events.close()

    def log_decision(self, text_ru: str, **fields) -> None:
        """Одно решение: строка в decisions.log + JSON в events.jsonl."""
        self._decisions.write(text_ru + "\n")
        self._decisions.flush()
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "text": text_ru,
            **fields,
        }
        self._events.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._events.flush()

    def daily_summary(
        self,
        day: Date,
        equity: float,
        prev_equity: float,
        costs_today: float,
        costs_total: float,
        start_capital: float,
        risk_free_rate: float,
        n_positions: int,
        max_positions: int,
    ) -> str:
        """ИТОГИ ДНЯ. Последняя строка — сравнение с фондом денежного
        рынка, она обязательна в каждой ежедневной сводке."""
        day_change = equity / prev_equity - 1 if prev_equity > 0 else 0.0
        rf_daily = (1 + risk_free_rate) ** (1 / 365.25) - 1
        total_change = equity / start_capital - 1
        text = (
            f"{day.isoformat()}  ИТОГИ ДНЯ\n"
            f"            Портфель: {fmt_rub(equity, 0)} ₽ "
            f"({day_change:+.2%} за день, {total_change:+.2%} с начала)\n"
            f"            Позиций: {n_positions} из {max_positions}\n"
            f"            Издержки за день: {fmt_rub(costs_today)} ₽ "
            f"({costs_today / start_capital:.2%} капитала)\n"
            f"            Издержки с начала: {fmt_rub(costs_total)} ₽ "
            f"({costs_total / start_capital:.2%} капитала)\n"
            f"            Бенчмарк за тот же период: фонд ДР дал "
            f"{rf_daily:+.3%} за день"
        )
        self.log_decision(
            text,
            тип="итоги_дня",
            день=day.isoformat(),
            портфель=round(equity, 2),
            изменение_за_день=round(day_change, 6),
            издержки_за_день=round(costs_today, 2),
            издержки_всего=round(costs_total, 2),
            позиций=n_positions,
        )
        return text

    def write_monthly_csv(self, table: pd.DataFrame, name: str = "monthly.csv") -> Path:
        path = self.dir / name
        table.to_csv(path, index=False, encoding="utf-8")
        return path


def monthly_report(
    equity: pd.Series,
    fills: list,
    risk_free_rate: float,
    guard_events: list[str] | None = None,
) -> tuple[str, pd.DataFrame]:
    """Ежемесячный отчёт: текст + таблица (для CSV).

    Колонки: месяц, доходность, фонд ДР за месяц, издержки, число сделок,
    сработавшие предохранители.
    """
    guard_events = guard_events or []
    month_ends = equity.resample("ME").last()
    rows = []
    prev_value = float(equity.iloc[0])
    for month_end, value in month_ends.items():
        month_key = f"{month_end.year}-{month_end.month:02d}"
        month_fills = [f for f in fills if f.day.strftime("%Y-%m") == month_key]
        month_guards = [g for g in guard_events if g.startswith(month_key)]
        month_start = month_end.replace(day=1)
        days_in_month = (month_end - month_start).days + 1
        rf_month = (1 + risk_free_rate) ** (days_in_month / 365.25) - 1
        rows.append(
            {
                "месяц": month_key,
                "портфель_руб": round(float(value), 2),
                "доходность_пct": round((float(value) / prev_value - 1) * 100, 2),
                "фонд_др_пct": round(rf_month * 100, 2),
                "издержки_руб": round(sum(f.costs.total for f in month_fills), 2),
                "сделок": len(month_fills),
                "предохранители": "; ".join(month_guards) if month_guards else "",
            }
        )
        prev_value = float(value)

    table = pd.DataFrame(rows)
    lines = ["ЕЖЕМЕСЯЧНЫЙ ОТЧЁТ"]
    for r in rows:
        beat = "обогнал фонд ДР" if r["доходность_пct"] > r["фонд_др_пct"] \
            else "ПРОИГРАЛ фонду ДР"
        lines.append(
            f"  {r['месяц']}: {r['доходность_пct']:+.2f}% "
            f"(фонд ДР {r['фонд_др_пct']:+.2f}%, {beat}) | "
            f"портфель {fmt_rub(r['портфель_руб'], 0)} ₽ | "
            f"издержки {fmt_rub(r['издержки_руб'])} ₽ | сделок {r['сделок']}"
            + (f" | ПРЕДОХРАНИТЕЛИ: {r['предохранители']}"
               if r["предохранители"] else "")
        )
    months_beaten = sum(
        1 for r in rows if r["доходность_пct"] > r["фонд_др_пct"]
    )
    lines.append(
        f"  Итого: обогнал фонд денежного рынка в {months_beaten} из "
        f"{len(rows)} месяцев."
    )
    return "\n".join(lines), table
