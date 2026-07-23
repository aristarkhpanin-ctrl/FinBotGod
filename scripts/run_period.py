"""Что было бы, если начать инвестировать с заданной даты.

Запуск (по умолчанию — старт инвестирования в мае 2022):
    python scripts/run_period.py
    python scripts/run_period.py --test-start 2023-01-01

Дисциплина честности та же, что в walk-forward:
* параметры каждой гипотезы подбираются ТОЛЬКО на обучающем периоде,
  который заканчивается ДО старта инвестирования;
* к периоду инвестирования выбранная конфигурация применяется без
  единого изменения;
* для контроля множественного тестирования все конфигурации тоже
  прогоняются на периоде инвестирования — в отчёте видны лучшая
  и медианная, а не только выбранная;
* каждый прогон пишется в журнал гипотез.

ВАЖНО: выбор точки старта задним числом — сам по себе подгонка.
Май 2022 — это дно после обвала; отчёт прямо предупреждает об этом.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexClient
from trading.data.validation import build_trading_calendar, validate_candles
from trading.engine.backtest import BacktestEngine
from trading.formatting import fmt_rub
from trading.logging_setup import setup_logging
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import expand_grid
from trading.settings import load_settings, load_universe
from trading.strategy.examples.monthly_ranked import (
    AbsoluteMomentum,
    CrossSectionalMomentum,
    MeanReversion,
)

HYPOTHESES = [
    ("Кросс-секционный моментум", CrossSectionalMomentum,
     {"lookback": [60, 120, 250], "top_n": [2, 3, 4]}),
    ("Моментум с фильтром падения", AbsoluteMomentum,
     {"lookback": [120, 250], "top_n": [2, 4]}),
    ("Возврат к среднему", MeanReversion,
     {"lookback": [5, 10, 21], "top_n": [2, 3, 4]}),
]


def slice_frames(frames: dict, start: str, end: str) -> dict:
    out = {}
    for secid, df in frames.items():
        d = df[(df["date"] >= start) & (df["date"] <= end)]
        if not d.empty:
            out[secid] = d
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", default="2018-05-01")
    parser.add_argument("--train-end", default="2022-04-30")
    parser.add_argument("--test-start", default="2022-05-01")
    parser.add_argument("--test-end", default="2026-07-22")
    args = parser.parse_args()

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
    imoex = market.index_candles(settings.data.history_start, "2026-07-22")
    imoex_test = imoex[
        (imoex["date"] >= args.test_start) & (imoex["date"] <= args.test_end)
    ].set_index("date")["close"]

    # Стратегии видят историю с самого начала данных (это прошлое — его
    # знал бы любой инвестор), но СДЕЛКИ начинаются только с trade_from.
    train = slice_frames(clean, settings.data.history_start, args.train_end)
    test = slice_frames(clean, settings.data.history_start, args.test_end)
    period_tag = f"{args.test_start}…{args.test_end}"

    lines = [
        f"# Если бы мы начали инвестировать {args.test_start}",
        "",
        f"Капитал на старте: {fmt_rub(settings.capital.start_amount, 0)} ₽. "
        f"Период инвестирования: {period_tag}.",
        f"Параметры подобраны только на обучении "
        f"{args.train_start}…{args.train_end} — будущее не подсматривалось.",
        "",
        "**ПРЕДУПРЕЖДЕНИЕ О ТОЧКЕ СТАРТА.** Дата начала выбрана задним "
        "числом. Май 2022 — дно после обвала: любой, кто «начинает» с этой "
        "точки в ретроспективе, получает подарок, которого не имел бы в "
        "реальном времени — в мае 2022 никто не знал, что дно уже позади.",
        "",
    ]
    print("\n".join(lines))

    def run(candles, strategy, tag, trade_from):
        return BacktestEngine(
            candles=candles, lot_sizes=lot_sizes, settings=settings,
            strategy=strategy, ledger=ledger,
            data_hash=f"{data_hash}|{tag}", source="llm",
            imoex_close=imoex_test if tag.startswith("тест") else None,
            trade_from=trade_from,
        ).run()

    for title, cls, grid in HYPOTHESES:
        combos = expand_grid(grid)
        best_params, best_metric = None, None
        for params in combos:
            r = run(train, cls(**params),
                    f"обучение {args.train_start}…{args.train_end}",
                    trade_from=args.train_start)
            metric = r.metrics.get("annual_return")
            if metric is not None and (best_metric is None or metric > best_metric):
                best_params, best_metric = params, metric

        oos_annuals = {}
        chosen = None
        for params in combos:
            r = run(test, cls(**params), f"тест {period_tag}",
                    trade_from=args.test_start)
            oos_annuals[str(params)] = r.metrics["annual_return"]
            if params == best_params:
                chosen = r

        end_equity = chosen.metrics["end_equity"]
        profit = end_equity - settings.capital.start_amount
        values = sorted(oos_annuals.values())
        block = [
            f"## {title}",
            "",
            f"Выбрано на обучении: `{best_params}` "
            f"(на обучении дало {best_metric:.1%} годовых).",
            "",
            f"**Итог: {fmt_rub(end_equity, 0)} ₽ — "
            f"{'прибыль' if profit >= 0 else 'убыток'} {fmt_rub(abs(profit), 0)} ₽ "
            f"({chosen.metrics['total_return']:+.1%} за период, "
            f"{chosen.metrics['annual_return']:+.1%} годовых)**",
            "",
            f"- Макс. просадка: {chosen.metrics['max_drawdown']:.1%}",
            f"- Сделок: {chosen.metrics['n_trades']} | "
            f"Издержки: {fmt_rub(chosen.metrics['total_costs'])} ₽",
            f"- НДФЛ (по годам): {fmt_rub(sum(chosen.ndfl_by_year.values()), 0)} ₽",
            f"- Предохранители: {len(chosen.guard_events)} срабатываний",
            f"- Все {len(combos)} конфигураций на этом периоде: лучшая "
            f"{max(values):+.1%}, медианная {statistics.median(values):+.1%}, "
            f"худшая {min(values):+.1%} годовых",
            "",
        ]
        lines += block
        print("\n".join(block))

    # --- Бенчмарки --------------------------------------------------------
    days = (imoex_test.index[-1] - imoex_test.index[0]).days
    years = days / 365.25
    imoex_factor = float(imoex_test.iloc[-1]) / float(imoex_test.iloc[0])
    mm_factor = (1 + settings.benchmark.risk_free_rate) ** years
    start_cap = settings.capital.start_amount
    bench = [
        "## Бенчмарки на том же периоде",
        "",
        f"- Купил и держи индекс IMOEX: {fmt_rub(start_cap * imoex_factor, 0)} ₽ "
        f"({imoex_factor - 1:+.1%} за период, "
        f"{imoex_factor ** (1 / years) - 1:+.1%} годовых; без дивидендов)",
        f"- Фонд денежного рынка под {settings.benchmark.risk_free_rate:.2%}: "
        f"{fmt_rub(start_cap * mm_factor, 0)} ₽ ({mm_factor - 1:+.1%} за период)",
        "",
        "Ставка фонда ДР взята постоянной из конфига; фактическая ключевая "
        "ставка в 2022–2026 менялась примерно от 7,5% до 21%, поэтому реальный "
        "результат фонда ДР был бы того же порядка, но не точно таким.",
        "",
        "Дивиденды не учтены ни в стратегиях, ни в индексе — обе цифры занижены.",
    ]
    lines += bench
    print("\n".join(bench))

    out = Path(f"docs/research_from_{args.test_start[:7]}.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nПолный отчёт: {out}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
