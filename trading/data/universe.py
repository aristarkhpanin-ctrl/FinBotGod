"""Состав торгового универсума на дату — защита от ошибки выживаемости.

Если тестировать стратегию на бумагах, которые торгуются сегодня,
обанкротившиеся и делистингованные компании выпадут из выборки и
результат будет завышен. Поэтому универсум на дату D — это исторический
состав индекса Мосбиржи на дату D, пересечённый с белым списком.

Если исторический состав на дату получить не удалось, это ограничение
фиксируется явным текстом в отчёте — не молча (требование ТЗ, раздел 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading.data.moex_client import MarketData


@dataclass
class UniverseSnapshot:
    date: str
    tickers: list[str]           # состав индекса на дату ∩ белый список
    from_index: bool             # True — исторический состав получен
    limitation_ru: str | None    # текст ограничения для отчёта, если есть


def index_schedule(market: MarketData, start: str, end: str,
                   index_id: str = "IMOEX", max_probe_days: int = 7) -> pd.Series:
    """Помесячный график состава индекса: дата среза → множество тикеров.

    Срез берётся в первый торговый день месяца: если на календарную дату
    биржа вернула пустой состав (выходной/праздник), пробуем следующие дни.
    Состав IMOEX пересматривается раз в квартал, помесячный шаг ловит
    каждое изменение с точностью до месяца.
    """
    snapshots: dict[pd.Timestamp, frozenset] = {}
    for month_start in pd.date_range(start, end, freq="MS"):
        for shift in range(max_probe_days):
            day = month_start + pd.Timedelta(days=shift)
            comp = market.index_composition(str(day.date()), index_id)
            if comp is not None and not comp.empty:
                col = "ticker" if "ticker" in comp.columns else "secids"
                snapshots[day] = frozenset(str(t).upper() for t in comp[col].dropna())
                break
    if not snapshots:
        return pd.Series(dtype=object)
    return pd.Series(snapshots).sort_index()


def universe_on_date(
    market: MarketData, date: str, whitelist: list[str]
) -> UniverseSnapshot:
    """Универсум на дату: исторический состав IMOEX ∩ белый список.

    При недоступности исторического состава возвращает белый список
    целиком с явным текстом ограничения.
    """
    try:
        composition = market.index_composition(date)
    except Exception as e:  # сетевая ошибка или отсутствие данных
        return UniverseSnapshot(
            date=date,
            tickers=sorted(whitelist),
            from_index=False,
            limitation_ru=(
                f"ОГРАНИЧЕНИЕ: исторический состав индекса на {date} получить "
                f"не удалось ({e}). Использован сегодняшний белый список — "
                f"результат бэктеста может быть завышен из-за ошибки выживаемости."
            ),
        )
    if composition.empty:
        return UniverseSnapshot(
            date=date,
            tickers=sorted(whitelist),
            from_index=False,
            limitation_ru=(
                f"ОГРАНИЧЕНИЕ: ISS не вернул состав индекса на {date}. "
                f"Использован сегодняшний белый список — результат бэктеста "
                f"может быть завышен из-за ошибки выживаемости."
            ),
        )
    ticker_col = "ticker" if "ticker" in composition.columns else "secids"
    index_tickers = {str(t).upper() for t in composition[ticker_col].dropna()}
    selected = sorted(index_tickers & set(whitelist))
    return UniverseSnapshot(
        date=date, tickers=selected, from_index=True, limitation_ru=None
    )
