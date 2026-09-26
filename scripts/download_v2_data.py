"""Данные для архитектуры v2: длинная история трёх «фондов».

Запуск:
    python scripts/download_v2_data.py

* MCFTRR — индекс МосБиржи полной доходности «нетто» (дивиденды
  реинвестируются после налога): прообраз фонда на индекс акций;
* MCFTR, IMOEX — для сравнения (брутто и только цены);
* RGBITR — индекс гособлигаций полной доходности: прообраз фонда ОФЗ;
* ключевая ставка ЦБ — доходность фонда денежного рынка.

Индексы берутся из архива итогов торгов ISS: там история с 2003 года,
включая кризис 2008 — модель риска должна увидеть настоящий обвал.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexClient
from trading.data.rates import RatesData
from trading.logging_setup import setup_logging
from trading.settings import load_settings

DATE_FROM, DATE_TILL = "2003-01-01", "2026-09-25"
RATES_FROM, RATES_TILL = "2013-09-13", "2026-09-26"
INDICES = ["MCFTRR", "MCFTR", "IMOEX", "RGBITR"]


def main() -> int:
    setup_logging()
    settings = load_settings()
    cache = ParquetCache(settings.data.cache_dir)
    client = MoexClient(requests_per_second=settings.data.requests_per_second,
                        timeout_seconds=settings.data.timeout_seconds,
                        retries=settings.data.retries)
    market = MarketData(client, cache)
    for name in INDICES:
        df = market.index_history(name, DATE_FROM, DATE_TILL)
        gaps = df["date"].diff().dt.days
        print(f"{name}: {len(df)} дней, {df['date'].min().date()} … "
              f"{df['date'].max().date()}, макс. разрыв {int(gaps.max())} дн., "
              f"последнее значение {df['close'].iloc[-1]:.2f}")
        jumps = df.set_index("date")["close"].pct_change().abs()
        big = jumps[jumps > 0.15]
        if not big.empty:
            print(f"    дневные изменения >15%: "
                  + ", ".join(f"{d.date()} {v:.0%}" for d, v in big.items()))
    kr = RatesData(cache).key_rate(RATES_FROM, RATES_TILL)
    print(f"Ключевая ставка: {len(kr)} точек, последняя {kr['rate_pct'].iloc[-1]}% "
          f"на {pd.Timestamp(kr['date'].max()).date()}")
    print(f"Хеш версии данных: {cache.data_version_hash()}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
