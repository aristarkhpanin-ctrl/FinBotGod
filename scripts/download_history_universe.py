"""Загрузка ИСТОРИЧЕСКОГО универсума: все, кто когда-либо был в IMOEX.

Запуск:
    python scripts/download_history_universe.py

Исправление ошибки выживаемости. Раньше исследования шли по 35 бумагам,
живым сегодня. Здесь скачивается:
* помесячный состав IMOEX 2015–2026 (кто был в индексе на каждую дату);
* дневные свечи всех бумаг, хоть раз входивших в индекс (включая ушедших
  с биржи — Polymetal, QIWI, «старый» Яндекс, Уралкалий и др.);
* история дивидендов по каждой бумаге;
* индекс полной доходности MCFTR (с дивидендами) — честный бенчмарк.

Всё кэшируется в Parquet; повторный запуск сеть не трогает.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.dividends import DividendsData, ex_dividend_dates
from trading.data.moex_client import MarketData, MoexClient
from trading.data.universe import index_schedule
from trading.data.validation import build_trading_calendar, validate_candles
from trading.logging_setup import setup_logging
from trading.settings import load_settings

DATE_FROM, DATE_TILL = "2015-01-01", "2026-07-22"


def main() -> int:
    setup_logging()
    settings = load_settings()
    cache = ParquetCache(settings.data.cache_dir)
    client = MoexClient(requests_per_second=settings.data.requests_per_second,
                        timeout_seconds=settings.data.timeout_seconds,
                        retries=settings.data.retries)
    market = MarketData(client, cache)

    print("1/4 Состав IMOEX помесячно…")
    schedule = index_schedule(market, DATE_FROM, DATE_TILL)
    members = sorted(set().union(*schedule.values))
    sizes = schedule.map(len)
    print(f"    срезов: {len(schedule)}, размер индекса {sizes.min()}–{sizes.max()}, "
          f"разных участников за период: {len(members)}")

    print("2/4 Дневные свечи всех участников…")
    frames = {}
    for i, t in enumerate(members, 1):
        frames[t] = market.daily_candles(t, DATE_FROM, DATE_TILL)
        if i % 10 == 0:
            print(f"    {i}/{len(members)}")
    calendar = build_trading_calendar(frames)
    last_day = max(calendar)
    delisted = sorted(t for t, df in frames.items()
                      if not df.empty and pd.Timestamp(df["date"].max()) < last_day - pd.Timedelta(days=30))

    print("3/4 Дивиденды…")
    dd = DividendsData(cache)
    div_counts = {}
    cal = pd.DatetimeIndex(sorted(calendar))
    for t in members:
        divs = dd.history(t)
        df = frames[t]
        ex = ex_dividend_dates(divs, cal)
        if not df.empty:
            lo, hi = pd.Timestamp(df["date"].min()), pd.Timestamp(df["date"].max())
            ex = ex[(ex["ex_date"] >= lo) & (ex["ex_date"] <= hi)]
        div_counts[t] = len(ex)

    print("4/4 Индекс полной доходности MCFTR…")
    mcftr = market.index_candles(DATE_FROM, DATE_TILL, "MCFTR")

    lots = market.lot_sizes()
    no_lot = sorted(t for t in members if t not in lots)

    print("\n" + "=" * 72)
    print("ОТЧЁТ: ИСТОРИЧЕСКИЙ УНИВЕРСУМ")
    print("=" * 72)
    print(f"Участников IMOEX за {DATE_FROM[:4]}–{DATE_TILL[:4]}: {len(members)} "
          f"(было в исследованиях: 35)")
    print(f"Ушли с торгов раньше конца периода: {len(delisted)} — {', '.join(delisted)}")
    print(f"Без сегодняшнего размера лота (берётся 1): {len(no_lot)} — {', '.join(no_lot)}")
    with_divs = [t for t, n in div_counts.items() if n > 0]
    print(f"Дивиденды найдены: {len(with_divs)} бумаг, выплат в периоде: "
          f"{sum(div_counts.values())}")
    print(f"Без найденных дивидендов: {', '.join(sorted(t for t, n in div_counts.items() if n == 0))}")
    print(f"MCFTR: {len(mcftr)} дней, с {pd.Timestamp(mcftr['date'].min()).date()}")

    flagged = []
    for t, df in sorted(frames.items()):
        rep = validate_candles(df, t, jump_threshold=settings.data.price_jump_threshold,
                               trading_calendar=calendar,
                               confirmed_events=settings.data.confirmed_events)
        if rep.suspicious:
            flagged.append(f"{t} {[(d, f'{c:+.0%}') for d, c in rep.price_jumps]}")
    print(f"\nСкачки цены >{settings.data.price_jump_threshold:.0%} (кроме 24.02.2022): "
          f"{len(flagged)}")
    for line in flagged:
        print(f"    {line}")
    print(f"\nХеш версии данных: {cache.data_version_hash()}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
