"""Прогон гипотез через walk-forward на реальных данных Мосбиржи.

Запуск:
    python scripts/run_research.py [--till 2026-07-22]

Что делает:
1. Берёт свечи из кэша (сеть — только если чего-то нет в кэше).
2. Исключает подозрительные бумаги (сплиты без подтверждения).
3. Прогоняет каждую гипотезу через walk-forward: параметры подбираются
   на обучающих окнах, метрики — только по склейке проверочных.
4. Каждая конфигурация автоматически пишется в журнал гипотез
   (journal/hypotheses.sqlite) с источником «llm».
5. Пишет полный отчёт в docs/ и печатает краткие вердикты.

Гипотезы сгенерированы LLM. Их немного и они простые — это осознанно:
чем больше вариантов проверено, тем менее значим лучший результат.
"""

from __future__ import annotations

import argparse
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
from trading.strategy.examples.monthly_ranked import (
    AbsoluteMomentum,
    CrossSectionalMomentum,
    MeanReversion,
)

HYPOTHESES = [
    (
        "Гипотеза 1 — кросс-секционный моментум",
        "Лидеры роста последних месяцев продолжают расти.",
        CrossSectionalMomentum,
        {"lookback": [60, 120, 250], "top_n": [2, 3, 4]},
    ),
    (
        "Гипотеза 2 — моментум с фильтром падения",
        "Как гипотеза 1, но падающие бумаги не покупаются вовсе — лучше кэш.",
        AbsoluteMomentum,
        {"lookback": [120, 250], "top_n": [2, 4]},
    ),
    (
        "Гипотеза 3 — возврат к среднему",
        "Сильно упавшие за последние дни бумаги отскакивают.",
        MeanReversion,
        {"lookback": [5, 10, 21], "top_n": [2, 3, 4]},
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--till", default="2026-07-22")
    args = parser.parse_args()

    setup_logging()
    settings = load_settings()
    tickers = load_universe()
    cache = ParquetCache(settings.data.cache_dir)
    client = MoexClient(
        requests_per_second=settings.data.requests_per_second,
        timeout_seconds=settings.data.timeout_seconds,
        retries=settings.data.retries,
    )
    market = MarketData(client, cache)
    date_from = settings.data.history_start

    # --- Данные и отбор чистых бумаг ------------------------------------
    frames = {
        secid: market.daily_candles(secid, date_from, args.till)
        for secid in tickers
    }
    calendar = build_trading_calendar(frames)
    clean, excluded = {}, []
    for secid, df in frames.items():
        report = validate_candles(
            df, secid,
            jump_threshold=settings.data.price_jump_threshold,
            trading_calendar=calendar,
            confirmed_events=settings.data.confirmed_events,
        )
        if report.suspicious:
            excluded.append(f"{secid} ({report.price_jumps})")
        elif not df.empty:
            clean[secid] = df

    imoex = market.index_candles(date_from, args.till)
    lot_sizes = market.lot_sizes()
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")

    lines: list[str] = [
        "# Отчёт исследования: три гипотезы на данных Мосбиржи 2015–2026",
        "",
        f"Дата прогона: {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
        f"Хеш версии данных: `{data_hash}`.",
        "",
        "```",
        breakeven_header_ru(settings),
        "```",
        "",
        f"Универсум: {len(clean)} бумаг (исключены как подозрительные: "
        f"{', '.join(excluded) if excluded else 'нет'}).",
        "",
        "Источник гипотез: LLM (Claude) — так и помечено в журнале гипотез.",
        "",
    ]
    print(f"Бумаг в исследовании: {len(clean)}, исключено: {excluded}")

    verdicts = []
    for title, idea, cls, grid in HYPOTHESES:
        print(f"\n=== {title} ===\nИдея: {idea}\nСетка параметров: {grid}")
        started = time.monotonic()
        runner = WalkForwardRunner(
            candles=clean,
            lot_sizes=lot_sizes,
            settings=settings,
            strategy_factory=cls,
            param_grid=grid,
            ledger=ledger,
            data_hash=f"{data_hash}|{cls.__name__}",
            source="llm",
        )
        result = runner.run()
        elapsed = time.monotonic() - started

        # Бенчмарки на том же OOS-периоде.
        oos_start, oos_end = result.oos_equity.index[0], result.oos_equity.index[-1]
        imoex_oos = imoex[(imoex["date"] >= oos_start) & (imoex["date"] <= oos_end)]
        imoex_bm = buy_and_hold_benchmark(imoex_oos.set_index("date")["close"])

        print(result.report_ru)
        print(f"(время прогона: {elapsed:.0f} с)")
        verdicts.append((title, result))

        lines += [
            f"## {title}",
            "",
            f"*Идея: {idea}*",
            "",
            "```",
            result.report_ru,
            "```",
            "",
            f"Для сравнения на том же OOS-периоде "
            f"({oos_start:%Y-%m-%d} … {oos_end:%Y-%m-%d}):",
            f"- Купил и держи индекс IMOEX: "
            f"{imoex_bm['annual_return']:.1%} годовых (без издержек, без дивидендов)",
            f"- Фонд денежного рынка: "
            f"{settings.benchmark.risk_free_rate:.2%} годовых",
            "",
        ]

    # --- Сводка журнала гипотез -----------------------------------------
    summary = ledger.summary_ru()
    lines += [
        "## Журнал гипотез (все испытания, накопительно)",
        "",
        "```",
        summary,
        "```",
        "",
        "## Ограничения",
        "",
        "- Дивиденды не учтены ни в стратегиях, ни в бенчмарке IMOEX "
        "(у обоих занижение, у дивидендных стратегий — сильнее).",
        "- Лотность взята сегодняшняя; исторические изменения лотов "
        "ISS не отдаёт.",
        "- UPRO делистингована в 2025: позиции в ней после остановки торгов "
        "замораживаются по последней цене до конца прогона.",
        "- AFKS исключена: скачок −37% 03.05.2017 не подтверждён заказчиком "
        "как рыночное событие.",
        "",
        "## Вердикты",
        "",
    ]
    print("\n" + "=" * 72)
    print("СВОДКА ЖУРНАЛА ГИПОТЕЗ")
    print(summary)
    print("\nВЕРДИКТЫ:")
    for title, result in verdicts:
        print(f"\n{title}:\n  {result.verdict_ru}")
        lines += [f"**{title}**", "", result.verdict_ru, ""]

    out = Path("docs/research_2026-07.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nПолный отчёт: {out}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
