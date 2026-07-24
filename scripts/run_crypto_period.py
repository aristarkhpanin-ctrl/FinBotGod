"""Крипта: обучение с 2022, инвестирование с 2025 (фиксированные периоды).

Запуск:
    python scripts/run_crypto_period.py
    python scripts/run_crypto_period.py --train-start 2022-01-01 --test-start 2025-01-01

Честная схема (как run_period.py для Мосбиржи):
* параметры каждой гипотезы подбираются ТОЛЬКО на обучающем окне
  (2022-01-01 … конец 2024) и применяются к периоду инвестирования
  (2025-01-01 … 2026-07-22) без единого изменения;
* стратегия видит историю с 2022 года для разогрева, но сделки и метрики —
  строго с даты старта инвестирования (trade_from);
* все конфигурации прогоняются и на периоде инвестирования — для медианы
  (защита от множественного тестирования);
* бенчмарки: купил-и-держи BTC и фонд денежного рынка под 4,5%.

ПРЕДУПРЕЖДЕНИЕ: выбор дат задним числом — сам по себе подгонка; период
инвестирования короткий (~1,5 года); выживаемость выборки завышает результат.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.crypto_client import BitfinexClient, CryptoData, crypto_lot_size
from trading.data.validation import build_trading_calendar, validate_candles
from trading.engine.backtest import BacktestEngine
from trading.formatting import fmt_rub
from trading.logging_setup import setup_logging
from trading.reporting.metrics import annualize
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import expand_grid
from trading.settings import DEFAULT_SETTINGS_PATH, load_settings, load_universe
from trading.strategy.examples.monthly_ranked import (
    CrossSectionalMomentum,
    MeanReversion,
)

HYPOTHESES = [
    ("Моментум", CrossSectionalMomentum, {"lookback": [30, 60, 120], "top_n": [3, 4, 6]}),
    ("Возврат к среднему", MeanReversion, {"lookback": [5, 10, 21], "top_n": [3, 4, 6]}),
]


def slice_frames(frames, start, end):
    out = {}
    for secid, df in frames.items():
        d = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(d) > 130:
            out[secid] = d
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-07-22")
    args = parser.parse_args()

    setup_logging()
    cfg = DEFAULT_SETTINGS_PATH.parent
    settings = load_settings(cfg / "settings_crypto.yaml")
    tickers = load_universe(cfg / "crypto_universe.yaml", uppercase=False)
    cache = ParquetCache(settings.data.cache_dir)
    market = CryptoData(BitfinexClient(), cache)

    frames = {t: market.daily_candles(t) for t in tickers}
    calendar = build_trading_calendar(frames)
    clean, excluded = {}, []
    for t, df in frames.items():
        rep = validate_candles(df, t, jump_threshold=settings.data.price_jump_threshold,
                               trading_calendar=calendar)
        if rep.suspicious:
            excluded.append(t)
        elif not df.empty:
            clean[t] = df

    lot_sizes = {t: crypto_lot_size(float(df["close"].median())) for t, df in clean.items()}
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")
    btc = clean.get("tBTCUSD")

    # Данные видны только с 2022 (обучение начинается с этого года).
    visible = slice_frames(clean, args.train_start, args.test_end)
    train_span = slice_frames(clean, args.train_start, args.train_end)
    n_coins = len(visible)

    lines = [
        f"# Крипта: обучение с {args.train_start[:4]}, инвестирование с {args.test_start[:4]}",
        "",
        f"Капитал: $250. Обучение {args.train_start}…{args.train_end}, "
        f"инвестирование {args.test_start}…{args.test_end}. Монет: {n_coins} "
        f"(исключены по скачку >80%: {', '.join(excluded) or 'нет'}).",
        "",
        "Параметры подобраны только на обучении. Стратегия видит историю "
        "с 2022 для разогрева; сделки и метрики — с даты старта инвестирования.",
        "",
        "**ПРЕДУПРЕЖДЕНИЕ.** Период инвестирования короткий (~1,5 года) — "
        "статистически это одна точка. Выживаемость выборки (только живые "
        "на 2026 монеты) завышает результат. Выбор дат задним числом — "
        "форма подгонки.",
        "",
    ]
    print("\n".join(lines))

    def run(candles, strategy, tag, trade_from):
        s = settings.model_copy(deep=True)
        return BacktestEngine(
            candles=candles, lot_sizes=lot_sizes, settings=s, strategy=strategy,
            ledger=ledger, data_hash=f"{data_hash}|{tag}", source="llm",
            trade_from=trade_from,
        ).run()

    for title, cls, grid in HYPOTHESES:
        combos = expand_grid(grid)
        # 1. Выбор лучшей конфигурации на обучении.
        best_params, best_metric = None, None
        for params in combos:
            r = run(train_span, cls(**params), f"обуч|{title}", args.train_start)
            m = r.metrics.get("annual_return")
            if m is not None and (best_metric is None or m > best_metric):
                best_params, best_metric = params, m

        # 2. Все конфигурации на периоде инвестирования — для медианы.
        test_annuals, chosen = {}, None
        for params in combos:
            r = run(visible, cls(**params), f"инвест|{title}", args.test_start)
            test_annuals[str(params)] = r.metrics["annual_return"]
            if params == best_params:
                chosen = r

        end_eq = chosen.metrics["end_equity"]
        profit = end_eq - settings.capital.start_amount
        vals = sorted(test_annuals.values())
        block = [
            f"## {title}",
            "",
            f"Выбрано на обучении: `{best_params}` (на обучении {best_metric:+.1%} годовых).",
            "",
            f"**Итог: ${end_eq:.0f} — "
            f"{'прибыль' if profit >= 0 else 'убыток'} ${abs(profit):.0f} "
            f"({chosen.metrics['total_return']:+.1%} за период, "
            f"{chosen.metrics['annual_return']:+.1%} годовых)**",
            "",
            f"- Макс. просадка: {chosen.metrics['max_drawdown']:.1%} | "
            f"Шарп: {chosen.metrics['sharpe']:.2f} | сделок: {chosen.metrics['n_trades']}",
            f"- Предохранители: {len(chosen.guard_events)} | "
            f"остановлен: {'да' if chosen.halted_reason else 'нет'}",
            f"- Все {len(combos)} конфигураций на периоде инвестирования: "
            f"лучшая {max(vals):+.1%}, медианная {statistics.median(vals):+.1%}, "
            f"худшая {min(vals):+.1%} годовых",
            "",
        ]
        lines += block
        print("\n".join(block))

    # Бенчмарки на периоде инвестирования.
    bench = ["## Бенчмарки на периоде инвестирования", ""]
    if btc is not None:
        seg = btc[(btc["date"] >= args.test_start) & (btc["date"] <= args.test_end)]
        if len(seg) >= 2:
            days = (seg["date"].iloc[-1] - seg["date"].iloc[0]).days
            factor = float(seg["close"].iloc[-1]) / float(seg["close"].iloc[0])
            bench.append(f"- Купил и держи BTC: ${250*factor:.0f} "
                         f"({factor-1:+.1%} за период, {annualize(factor, days):+.1%} годовых)")
    yrs = (pd.Timestamp(args.test_end) - pd.Timestamp(args.test_start)).days / 365.25
    mm = (1 + settings.benchmark.risk_free_rate) ** yrs
    bench.append(f"- Фонд денежного рынка под {settings.benchmark.risk_free_rate:.1%}: "
                 f"${250*mm:.0f} ({mm-1:+.1%} за период)")
    lines += bench + [""]
    print("\n".join(bench))

    out = Path(f"docs/research_crypto_from_{args.test_start[:7]}.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nПолный отчёт: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
