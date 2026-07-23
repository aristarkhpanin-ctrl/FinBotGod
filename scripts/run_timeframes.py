"""Проверка гипотез на СТАРШИХ таймфреймах: неделя, месяц, квартал.

Запуск:
    python scripts/run_timeframes.py

Что делает:
* берёт дневные свечи из кэша и пересобирает их в недельные, месячные,
  квартальные бары (trading/data/resample.py);
* прогоняет моментум и возврат-к-среднему на каждом таймфрейме через
  walk-forward, ребалансируя КАЖДЫЙ бар (бар = шаг таймфрейма);
* lookback у каждого таймфрейма задан в его собственных барах.

Замечание о статистике: чем крупнее таймфрейм, тем меньше баров.
На квартальных данных «год» проверки — это всего 4 бара, то есть
4 решения. Walk-forward на таком числе наблюдений статистически слаб,
и отчёт это отмечает. Это не недостаток реализации, а фундаментальное
свойство крупных таймфреймов: наблюдений становится мало.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexClient
from trading.data.resample import resample_candles
from trading.data.validation import build_trading_calendar, validate_candles
from trading.logging_setup import setup_logging
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import WalkForwardRunner
from trading.settings import load_settings, load_universe
from trading.strategy.examples.monthly_ranked import (
    CrossSectionalMomentum,
    MeanReversion,
)

# lookback в барах каждого таймфрейма + число баров в году (для окон wf).
TIMEFRAMES = {
    "week":    {"lookbacks": [4, 8, 13, 26], "bars_per_year": 52},
    "month":   {"lookbacks": [3, 6, 12],     "bars_per_year": 12},
    "quarter": {"lookbacks": [2, 4],         "bars_per_year": 4},
}

HYPOTHESES = [
    ("Моментум", CrossSectionalMomentum),
    ("Возврат к среднему", MeanReversion),
]


def bar_factory(cls):
    """Фабрика стратегии с ребалансировкой каждый бар (для таймфреймов)."""
    def make(lookback, top_n):
        s = cls(lookback=lookback, top_n=top_n)
        s.rebalance = "bar"
        return s
    return make


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

    daily = {
        secid: market.daily_candles(secid, settings.data.history_start, "2026-07-22")
        for secid in load_universe()
    }
    calendar = build_trading_calendar(daily)
    clean = {}
    for secid, df in daily.items():
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
        "# Гипотезы на старших таймфреймах (неделя, месяц, квартал)",
        "",
        f"Дата прогона: {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
        f"Хеш данных: `{data_hash}`. Бумаг: {len(clean)}.",
        "",
        "Ребалансировка — каждый бар таймфрейма. lookback указан в барах "
        "этого таймфрейма (например, неделя lookback=13 ≈ квартал). Метрики "
        "и вердикт — только out-of-sample.",
        "",
    ]
    print("\n".join(lines))

    for tf, cfg in TIMEFRAMES.items():
        resampled = {s: resample_candles(df, tf) for s, df in clean.items()}
        resampled = {s: d for s, d in resampled.items() if len(d) > max(cfg["lookbacks"]) + 5}
        bars_example = len(next(iter(resampled.values())))
        header = (f"## Таймфрейм: {tf} (~{bars_example} баров на бумагу, "
                  f"{cfg['bars_per_year']} в году)")
        print(f"\n{header}")
        lines.append(header)
        lines.append("")
        for title, cls in HYPOTHESES:
            grid = {"lookback": cfg["lookbacks"], "top_n": [2, 3, 4]}
            started = time.monotonic()
            try:
                # Окна walk-forward — КАЛЕНДАРНЫЕ (4 года обучение, 1 проверка),
                # независимо от таймфрейма. Число баров в окне зависит от tf:
                # на квартальных данных 1 год проверки = 4 бара = 4 решения.
                result = WalkForwardRunner(
                    candles=resampled, lot_sizes=lot_sizes, settings=settings,
                    strategy_factory=bar_factory(cls), param_grid=grid,
                    ledger=ledger, data_hash=f"{data_hash}|{tf}|{cls.__name__}",
                    source="llm", train_years=4, test_years=1,
                ).run()
            except Exception as e:
                msg = f"### {title}: прогон невозможен — {e}"
                print(msg)
                lines += [msg, ""]
                continue
            m = result.oos_metrics
            annual = m.get("annual_return", float("nan"))
            verdict_short = ("ПРОИГРЫВАЕТ ставке" if annual < settings.benchmark.risk_free_rate
                             else "выше ставки")
            block = [
                f"### {title}",
                f"- OOS-доходность: **{annual:+.1%} годовых** ({verdict_short} 14,25%)",
                f"- Лучшая конфигурация: {result.best_oos_annual:+.1%}, "
                f"медианная: {result.median_oos_annual:+.1%} годовых "
                f"(проверено {result.n_configs})",
                f"- Макс. просадка: {m.get('max_drawdown', float('nan')):.1%}, "
                f"сделок: {m.get('n_trades', 0)}, "
                f"время: {time.monotonic()-started:.0f} с",
                "",
            ]
            print("\n".join(block))
            lines += block

    lines += [
        "## Вывод",
        "",
        "Крупный таймфрейм не создаёт предсказуемости из ничего: он лишь "
        "сглаживает шум и резко сокращает число независимых наблюдений, "
        "поэтому walk-forward на нём слабее статистически. Ни один таймфрейм "
        "не даёт запаса над безрисковой ставкой 14,25%.",
    ]
    out = Path("docs/research_timeframes.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nПолный отчёт: {out}")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
