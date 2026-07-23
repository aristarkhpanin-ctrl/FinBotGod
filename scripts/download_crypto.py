"""Загрузка дневных свечей криптовалют с Bitfinex и отчёт валидации.

Запуск:
    python scripts/download_crypto.py

Скачивает пары из crypto_universe.yaml, кэширует в Parquet, печатает
отчёт валидации. Повторный запуск не делает сетевых запросов.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.data.cache import ParquetCache
from trading.data.crypto_client import BitfinexClient, CryptoData
from trading.data.validation import build_trading_calendar, validate_candles
from trading.logging_setup import setup_logging
from trading.reporting.report import breakeven_header_ru
from trading.settings import DEFAULT_SETTINGS_PATH, load_settings, load_universe


def main() -> int:
    setup_logging()
    settings = load_settings(DEFAULT_SETTINGS_PATH.parent / "settings_crypto.yaml")
    tickers = load_universe(DEFAULT_SETTINGS_PATH.parent / "crypto_universe.yaml",
                            uppercase=False)

    print()
    print(breakeven_header_ru(settings))
    print()

    cache = ParquetCache(settings.data.cache_dir)
    client = BitfinexClient(requests_per_second=settings.data.requests_per_second,
                            timeout_seconds=settings.data.timeout_seconds,
                            retries=settings.data.retries)
    market = CryptoData(client, cache)

    print(f"Загрузка {len(tickers)} пар с Bitfinex…")
    frames = {}
    for i, pair in enumerate(tickers, 1):
        try:
            df = market.daily_candles(pair)
            frames[pair] = df
            rng = f"{df['date'].min().date()}…{df['date'].max().date()}" if len(df) else "нет"
            print(f"  [{i:>2}/{len(tickers)}] {pair:<12} {len(df):>5} дней  {rng}")
        except Exception as e:
            print(f"  [{i:>2}/{len(tickers)}] {pair:<12} ОШИБКА: {e}")

    print("\n" + "=" * 72)
    print("ОТЧЁТ ВАЛИДАЦИИ ДАННЫХ (крипта)")
    print("=" * 72)
    calendar = build_trading_calendar(frames)
    print(f"Опорный календарь: {len(calendar)} дней (крипта торгуется 24/7)\n")
    suspicious, total = [], 0
    for pair, df in sorted(frames.items()):
        rep = validate_candles(df, pair, jump_threshold=settings.data.price_jump_threshold,
                               trading_calendar=calendar)
        print(rep.describe_ru())
        total += rep.total_anomalies
        if rep.suspicious:
            suspicious.append(pair)
    print(f"\nИтого аномалий: {total}")
    if suspicious:
        print(f"Подозрительные (скачок > {settings.data.price_jump_threshold:.0%}): "
              f"{', '.join(suspicious)}")
    print("\nОГРАНИЧЕНИЕ ВЫЖИВАЕМОСТИ: это монеты, торгуемые на Bitfinex в 2026. "
          "Обнулившиеся и делистингованные монеты в выборку не попали — "
          "результат завышен.")
    print(f"\nХеш версии данных: {cache.data_version_hash()}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
