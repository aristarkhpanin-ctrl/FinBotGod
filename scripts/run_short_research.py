"""Walk-forward прогон ШОРТ-гипотез: зарабатываем на падениях.

Запуск:
    python scripts/run_short_research.py

Отличия от лонгового исследования (scripts/run_research.py):
* движок с allow_short=True — стратегии продают без покрытия;
* ежедневная плата за заём бумаг (ставка_займа_шорт_годовых, 18%)
  честно списывается из кэша;
* ВАЖНО: дивиденды в данных отсутствуют. Лонги это ЗАНИЖАЕТ, а шорты —
  ЗАВЫШАЕТ: в реальности шортист платит дивиденды из своего кармана.
  На российском рынке это 8–12% годовых не в пользу шорта. Отчёт
  предупреждает об этом явно.
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
from trading.reporting.report import breakeven_header_ru
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import WalkForwardRunner
from trading.settings import load_settings, load_universe
from trading.strategy.examples.monthly_ranked import ShortMomentum, ShortOverbought

HYPOTHESES = [
    (
        "Шорт-гипотеза 1 — моментум вниз",
        "Падающие бумаги продолжают падать: шортим худших.",
        ShortMomentum,
        {"lookback": [60, 120, 250], "top_n": [2, 3, 4]},
    ),
    (
        "Шорт-гипотеза 2 — откат перегретых",
        "Лидеры недавнего роста откатываются: шортим лучших.",
        ShortOverbought,
        {"lookback": [5, 10, 21], "top_n": [2, 3, 4]},
    ),
]


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

    frames = {
        secid: market.daily_candles(secid, settings.data.history_start, "2026-07-22")
        for secid in load_universe()
    }
    calendar = build_trading_calendar(frames)
    clean = {}
    for secid, df in frames.items():
        report = validate_candles(
            df, secid, jump_threshold=settings.data.price_jump_threshold,
            trading_calendar=calendar,
            confirmed_events=settings.data.confirmed_events,
        )
        if not report.suspicious and not df.empty:
            clean[secid] = df

    lot_sizes = market.lot_sizes()
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")

    lines = [
        "# Отчёт исследования: шорт-гипотезы (заработок на падениях)",
        "",
        f"Дата прогона: {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
        f"Хеш версии данных: `{data_hash}`.",
        "",
        "```",
        breakeven_header_ru(settings),
        "```",
        "",
        f"Плата за заём бумаг: {settings.costs.short_borrow_rate:.0%} годовых, "
        "начисляется ежедневно на стоимость короткой позиции.",
        "",
        "**ВАЖНОЕ ЗАВЫШЕНИЕ РЕЗУЛЬТАТА ШОРТОВ.** Дивиденды в данных "
        "отсутствуют. Шортист обязан выплачивать дивиденды по занятым "
        "бумагам из своего кармана (на российском рынке 8–12% годовых). "
        "Здесь эта выплата НЕ учтена, поэтому реальные результаты шортов "
        "были бы ЗАМЕТНО ХУЖЕ показанных.",
        "",
        "Источник гипотез: LLM (Claude).",
        "",
    ]
    print(f"Бумаг в исследовании: {len(clean)}")

    verdicts = []
    for title, idea, cls, grid in HYPOTHESES:
        print(f"\n=== {title} ===\nИдея: {idea}\nСетка: {grid}")
        started = time.monotonic()
        result = WalkForwardRunner(
            candles=clean,
            lot_sizes=lot_sizes,
            settings=settings,
            strategy_factory=cls,
            param_grid=grid,
            ledger=ledger,
            data_hash=f"{data_hash}|{cls.__name__}",
            source="llm",
            engine_kwargs={"allow_short": True},
        ).run()
        print(result.report_ru)
        print(f"(время прогона: {time.monotonic() - started:.0f} с)")
        verdicts.append((title, result))
        lines += [f"## {title}", "", f"*Идея: {idea}*", "",
                  "```", result.report_ru, "```", ""]

    lines += ["## Вердикты", ""]
    print("\nВЕРДИКТЫ:")
    for title, result in verdicts:
        print(f"\n{title}:\n  {result.verdict_ru}")
        lines += [f"**{title}**", "", result.verdict_ru, ""]

    out = Path("docs/research_short_2026-07.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nПолный отчёт: {out}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
