"""Скачивание дневных свечей по бумагам универсума с отчётом валидации.

Запуск:
    python scripts/download_universe.py [--till 2026-07-23]

Что делает:
1. Загружает список бумаг TQBR с размерами лотов (LOTSIZE).
2. Скачивает дневные свечи по каждому тикеру из universe.yaml
   с начала истории (settings.yaml → данные.начало_истории) по --till.
   Всё скачанное кэшируется в Parquet: повторный запуск не делает
   ни одного сетевого запроса.
3. Пытается получить исторический состав индекса IMOEX (антивыживаемость);
   при неудаче печатает ограничение явным текстом.
4. Валидирует каждую бумагу и печатает отчёт: аномалии, подозрительные
   бумаги (возможные сплиты), пропуски данных.

Стратегию не запускает, заявок не создаёт — это только слой данных.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexClient
from trading.data.universe import universe_on_date
from trading.data.validation import build_trading_calendar, validate_candles
from trading.logging_setup import get_logger, setup_logging
from trading.reporting.report import breakeven_header_ru
from trading.settings import ConfigError, load_settings, load_universe

log = get_logger("download")


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка данных Мосбиржи")
    parser.add_argument(
        "--till", default=date.today().isoformat(),
        help="Конечная дата диапазона (по умолчанию — сегодня). "
             "При повторном запуске с той же датой сеть не используется.",
    )
    parser.add_argument("--settings", default=None, help="Путь к settings.yaml")
    args = parser.parse_args()

    setup_logging()
    try:
        settings = load_settings(args.settings) if args.settings else load_settings()
        tickers = load_universe()
    except ConfigError as e:
        print(f"\nОШИБКА КОНФИГУРАЦИИ\n{e}", file=sys.stderr)
        return 2

    print()
    print(breakeven_header_ru(settings))
    print()

    cache = ParquetCache(settings.data.cache_dir)
    client = MoexClient(
        requests_per_second=settings.data.requests_per_second,
        timeout_seconds=settings.data.timeout_seconds,
        retries=settings.data.retries,
    )
    market = MarketData(client, cache)
    date_from = settings.data.history_start
    date_till = args.till

    # --- 1. Список бумаг и лотность -------------------------------------
    print("Загрузка списка бумаг TQBR (лотность)…")
    try:
        lot_sizes = market.lot_sizes()
    except Exception as e:
        print(
            f"\nСЕТЬ НЕДОСТУПНА: не удалось связаться с iss.moex.com.\n"
            f"Причина: {e}\n"
            f"Кэш не пострадал. Проверьте интернет (или сетевую политику "
            f"окружения) и запустите скрипт ещё раз — уже скачанное "
            f"повторно качаться не будет.",
            file=sys.stderr,
        )
        client.close()
        return 1
    print(f"Получено бумаг с LOTSIZE: {len(lot_sizes)}")

    # --- 2. Свечи по универсуму -----------------------------------------
    print(f"\nЗагрузка дневных свечей {date_from} … {date_till}, "
          f"{len(tickers)} бумаг (кэш: {settings.data.cache_dir}/)")
    frames = {}
    failures: dict[str, str] = {}
    for i, secid in enumerate(tickers, 1):
        try:
            frames[secid] = market.daily_candles(secid, date_from, date_till)
            n = len(frames[secid])
            print(f"  [{i:>2}/{len(tickers)}] {secid:<6} {n:>5} дней"
                  + ("  (нет данных!)" if n == 0 else ""))
        except Exception as e:
            failures[secid] = str(e)
            print(f"  [{i:>2}/{len(tickers)}] {secid:<6} ОШИБКА: {e}")

    # --- 3. Исторический состав индекса (антивыживаемость) --------------
    print("\nПроверка доступности исторического состава индекса IMOEX…")
    probe_dates = ["2015-06-01", "2020-06-01", date_till]
    limitations: list[str] = []
    for d in probe_dates:
        snap = universe_on_date(market, d, tickers)
        if snap.from_index:
            print(f"  {d}: состав получен, {len(snap.tickers)} бумаг из белого списка в индексе")
        else:
            print(f"  {d}: НЕДОСТУПЕН")
            limitations.append(snap.limitation_ru)

    # --- 4. Валидация ----------------------------------------------------
    print("\n" + "=" * 72)
    print("ОТЧЁТ ВАЛИДАЦИИ ДАННЫХ")
    print("=" * 72)
    calendar = build_trading_calendar(frames)
    print(f"Опорный торговый календарь: {len(calendar)} дней "
          f"(объединение дат по всем бумагам)")
    if settings.data.confirmed_events:
        print("Подтверждённые рыночные события (скачки в эти даты — не аномалии):")
        for event in settings.data.confirmed_events:
            scope = ", ".join(event.tickers) if event.tickers else "все бумаги"
            print(f"  {event.date} ({scope}): {event.reason}")
    print()
    suspicious: list[str] = []
    total_anomalies = 0
    for secid, df in sorted(frames.items()):
        report = validate_candles(
            df, secid,
            jump_threshold=settings.data.price_jump_threshold,
            trading_calendar=calendar,
            confirmed_events=settings.data.confirmed_events,
        )
        print(report.describe_ru())
        total_anomalies += report.total_anomalies
        if report.suspicious:
            suspicious.append(secid)

    print(f"\nИтого аномалий: {total_anomalies}")
    if suspicious:
        print(
            f"\nПОДОЗРИТЕЛЬНЫЕ БУМАГИ (скачок цены "
            f"> {settings.data.price_jump_threshold:.0%} за день — возможен сплит "
            f"или консолидация): {', '.join(suspicious)}\n"
            f"Эти бумаги ИСКЛЮЧАЮТСЯ из бэктеста. Автоматическая корректировка "
            f"не выполняется (требование ТЗ)."
        )
    if failures:
        print(f"\nНЕ СКАЧАЛИСЬ: {', '.join(sorted(failures))}")
    for text in limitations:
        print(f"\n{text}")

    # --- 5. Лотность по универсуму ---------------------------------------
    missing_lots = [t for t in tickers if t not in lot_sizes]
    print(f"\nЛотность известна для {len(tickers) - len(missing_lots)} "
          f"из {len(tickers)} бумаг универсума.")
    if missing_lots:
        print(f"БЕЗ ЛОТНОСТИ (нет в сегодняшнем списке TQBR — возможен "
              f"делистинг): {', '.join(missing_lots)}")

    print(f"\nХеш версии данных: {cache.data_version_hash()}")
    print("Готово. Повторный запуск с теми же датами не тронет сеть.")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
