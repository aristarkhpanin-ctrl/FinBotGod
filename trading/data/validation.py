"""Валидация ценовых рядов при загрузке.

Проверки из ТЗ (раздел 4): дубликаты дат, пропущенные торговые дни,
согласованность OHLC, неотрицательный объём, ненулевая цена, детектор
подозрительных скачков цены (возможные сплиты/консолидации).

Любая аномалия попадает в отчёт и в лог с предупреждением — не молча.
Скачок цены больше порога помечает бумагу как подозрительную: такие
бумаги исключаются из бэктеста с явным сообщением, автоматическая
корректировка не выполняется.

Исключение — подтверждённые рыночные события: даты реальных обвалов,
которые человек проверил и внёс в конфиг (например, 24.02.2022 — начало
СВО). Скачок в такую дату — не сплит и не битые данные, бумага остаётся
в бэктесте. Событие в любую другую дату по-прежнему исключает бумагу.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    secid: str
    n_rows: int = 0
    duplicate_dates: int = 0
    missing_days: int = 0
    bad_high: int = 0            # high < max(open, close)
    bad_low: int = 0             # low > min(open, close)
    negative_volume: int = 0
    zero_price: int = 0
    price_jumps: list[tuple[str, float]] = field(default_factory=list)  # (дата, изменение)
    confirmed_jumps: list[tuple[str, float, str]] = field(default_factory=list)
    #                ^ (дата, изменение, причина) — подтверждённые события
    suspicious: bool = False     # исключить из бэктеста

    @property
    def total_anomalies(self) -> int:
        return (
            self.duplicate_dates + self.missing_days + self.bad_high
            + self.bad_low + self.negative_volume + self.zero_price
            + len(self.price_jumps)
        )

    def describe_ru(self) -> str:
        """Одна строка отчёта по бумаге, понятная без чтения кода."""
        if self.n_rows == 0:
            return f"{self.secid}: данных нет"
        parts = []
        if self.duplicate_dates:
            parts.append(f"дубликатов дат: {self.duplicate_dates}")
        if self.missing_days:
            parts.append(f"пропущенных торговых дней: {self.missing_days}")
        if self.bad_high:
            parts.append(f"строк с high ниже open/close: {self.bad_high}")
        if self.bad_low:
            parts.append(f"строк с low выше open/close: {self.bad_low}")
        if self.negative_volume:
            parts.append(f"строк с отрицательным объёмом: {self.negative_volume}")
        if self.zero_price:
            parts.append(f"строк с нулевой ценой: {self.zero_price}")
        if self.price_jumps:
            jumps = ", ".join(f"{d}: {chg:+.0%}" for d, chg in self.price_jumps[:5])
            parts.append(f"скачков цены: {len(self.price_jumps)} ({jumps})")
        if self.confirmed_jumps:
            events = ", ".join(
                f"{d}: {chg:+.0%} — {reason}"
                for d, chg, reason in self.confirmed_jumps[:3]
            )
            parts.append(f"подтверждённых событий: {len(self.confirmed_jumps)} ({events})")
        status = "ПОДОЗРИТЕЛЬНА, исключена из бэктеста" if self.suspicious else "ок"
        detail = "; ".join(parts) if parts else "аномалий нет"
        return f"{self.secid}: {self.n_rows} дней, {detail} — {status}"


def _confirmed_reason(day_iso: str, secid: str, confirmed_events) -> str | None:
    """Причина подтверждённого события на дату, если оно покрывает бумагу."""
    for event in confirmed_events or []:
        if event.date == day_iso and (not event.tickers or secid in event.tickers):
            return event.reason
    return None


def validate_candles(
    df: pd.DataFrame,
    secid: str,
    jump_threshold: float = 0.35,
    trading_calendar: set | None = None,
    confirmed_events=None,
) -> ValidationReport:
    """Проверяет дневные свечи одной бумаги.

    ``trading_calendar`` — множество дат, в которые рынок точно торговался
    (объединение дат по всем бумагам универсума). Если не передан,
    пропущенные дни не считаются: у самой бумаги нет способа отличить
    выходной от дыры в данных.

    ``confirmed_events`` — список подтверждённых человеком рыночных событий
    (объекты с полями date, reason, tickers — см. settings.ConfirmedEvent).
    """
    report = ValidationReport(secid=secid, n_rows=len(df))
    if df.empty:
        return report

    dates = pd.to_datetime(df["date"])
    report.duplicate_dates = int(dates.duplicated().sum())

    if trading_calendar:
        own = set(dates.dt.normalize())
        span = {
            d for d in trading_calendar
            if dates.min().normalize() <= d <= dates.max().normalize()
        }
        report.missing_days = len(span - own)

    report.bad_high = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
    report.bad_low = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
    report.negative_volume = int((df["volume"] < 0).sum())
    report.zero_price = int(
        ((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()
    )

    # Детектор сплитов/консолидаций: скачок close-to-close больше порога.
    # Скачок в дату подтверждённого события — реальный обвал, не аномалия.
    ordered = df.sort_values("date")
    changes = ordered["close"].pct_change()
    for idx in changes.index[changes.abs() > jump_threshold]:
        day = pd.to_datetime(ordered.loc[idx, "date"]).date().isoformat()
        change = float(changes.loc[idx])
        reason = _confirmed_reason(day, secid, confirmed_events)
        if reason:
            report.confirmed_jumps.append((day, change, reason))
        else:
            report.price_jumps.append((day, change))
    if report.price_jumps:
        report.suspicious = True

    return report


def build_trading_calendar(frames: dict[str, pd.DataFrame]) -> set:
    """Объединение торговых дат по всем бумагам — опорный календарь."""
    calendar: set = set()
    for df in frames.values():
        if not df.empty:
            calendar |= set(pd.to_datetime(df["date"]).dt.normalize())
    return calendar
