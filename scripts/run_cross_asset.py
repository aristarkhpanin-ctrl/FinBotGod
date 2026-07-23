"""Межрыночная гипотеза: курс USD/RUB как ведущий сигнал для экспортёров.

Запуск:
    python scripts/run_cross_asset.py

Проверяет идею «движение одного актива предсказывает другой с лагом» на
самом сильном доступном сигнале Мосбиржи — курсе доллара. Лонг-версия:
держим корзину нефтяников и металлургов, когда рубль слабел; иначе кэш.
Перебираются окно наблюдения курса и лаг (0, месяц, квартал) — прямая
проверка гипотезы «курс двинулся раньше, акции реагируют позже».

Всё через walk-forward: метрики и вердикт — только out-of-sample.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexClient
from trading.data.validation import build_trading_calendar, validate_candles
from trading.logging_setup import setup_logging
from trading.reporting.metrics import buy_and_hold_benchmark
from trading.reporting.report import breakeven_header_ru
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import WalkForwardRunner
from trading.settings import load_settings, load_universe
from trading.strategy.examples.cross_asset import (
    EXPORTERS,
    SIGNAL_KEY,
    CrossAssetExporters,
)


def main() -> int:
    setup_logging()
    settings = load_settings()
    cache = ParquetCache(settings.data.cache_dir)
    client = MoexClient(
        requests_per_second=settings.data.requests_per_second,
        timeout_seconds=settings.data.timeout_seconds,
        retries=settings.data.retries,
    )
    market = MarketData(client, cache)
    dfrom, dtill = settings.data.history_start, "2026-07-22"

    daily = {s: market.daily_candles(s, dfrom, dtill) for s in load_universe()}
    calendar = build_trading_calendar(daily)
    clean = {}
    for secid, df in daily.items():
        report = validate_candles(
            df, secid, jump_threshold=settings.data.price_jump_threshold,
            trading_calendar=calendar, confirmed_events=settings.data.confirmed_events,
        )
        if not report.suspicious and not df.empty:
            clean[secid] = df

    # Курс USD/RUB, выровненный по торговому календарю акций.
    fx = market.fx_candles(dfrom, dtill)
    stock_dates = pd.Index(sorted(calendar))
    fx = fx.set_index("date").reindex(stock_dates).ffill().dropna(subset=["close"])
    fx = fx.reset_index().rename(columns={"index": "date"})
    clean[SIGNAL_KEY] = fx[["date", "open", "high", "low", "close", "value", "volume"]]

    lot_sizes = market.lot_sizes()
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")
    imoex = market.index_candles(dfrom, dtill)

    exporters_present = [s for s in EXPORTERS if s in clean]
    grid = {"lookback": [20, 60, 120], "lag": [0, 21, 63]}  # лаг: 0, ~месяц, ~квартал

    started = time.monotonic()
    result = WalkForwardRunner(
        candles=clean, lot_sizes=lot_sizes, settings=settings,
        strategy_factory=CrossAssetExporters, param_grid=grid,
        ledger=ledger, data_hash=f"{data_hash}|cross_usd", source="llm",
    ).run()
    elapsed = time.monotonic() - started

    oos_start, oos_end = result.oos_equity.index[0], result.oos_equity.index[-1]
    imoex_oos = imoex[(imoex["date"] >= oos_start) & (imoex["date"] <= oos_end)]
    imoex_bm = buy_and_hold_benchmark(imoex_oos.set_index("date")["close"])

    print(result.report_ru)
    lines = [
        "# Межрыночная гипотеза: USD/RUB → экспортёры",
        "",
        f"Дата прогона: {pd.Timestamp.now():%Y-%m-%d %H:%M}. Хеш данных: `{data_hash}`.",
        "",
        "```", breakeven_header_ru(settings), "```",
        "",
        f"Сигнал: курс USD/RUB (2015–2026). Корзина экспортёров "
        f"({len(exporters_present)} бумаг): {', '.join(exporters_present)}.",
        "",
        "Логика: держим корзину, когда рубль слабел за окно наблюдения "
        "(со сдвигом lag), иначе — кэш. Перебор: окно [20, 60, 120] дней, "
        "лаг [0, 21, 63] дня (одновременно / месяц / квартал).",
        "",
        "```", result.report_ru, "```",
        "",
        f"На том же OOS-периоде ({oos_start:%Y-%m-%d} … {oos_end:%Y-%m-%d}):",
        f"- Купил и держи IMOEX: {imoex_bm['annual_return']:.1%} годовых (без дивидендов)",
        f"- Фонд денежного рынка: {settings.benchmark.risk_free_rate:.2%} годовых",
        "",
        "## Ограничения",
        "",
        "- Прямого ряда Brent на Мосбирже нет (только фьючерсы, требующие "
        "сшивки контрактов) — использован курс USD/RUB как ближайший чистый "
        "межрыночный сигнал для экспортёров.",
        "- Дивиденды не учтены. Причинность «курс → акции» здесь проверяется "
        "статистически на ценах, без фундаментальной модели.",
        "",
        f"Время прогона: {elapsed:.0f} с.",
    ]
    out = Path("docs/research_cross_asset.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nВердикт: {result.verdict_ru}")
    print(f"Полный отчёт: {out}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
