"""Walk-forward гипотез на КРИПТОРЫНКЕ (Bitfinex, дневные свечи).

Запуск:
    python scripts/run_crypto_research.py

Тот же код и те же стратегии, что для Мосбиржи, — меняется только конфиг
(settings_crypto.yaml): безрисковая ставка 4,5%, издержки крипты, дробные
лоты, широкие риск-лимиты под волатильность. Это и есть проверка тезиса
«система переносится, меняются параметры».

Бенчмарк крипты — купил-и-держи BTC (роль индекса) и фонд денежного рынка
под 4,5%. Порог безубыточности здесь ~4,5%, а не 38%.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.crypto_client import BitfinexClient, CryptoData, crypto_lot_size
from trading.data.validation import build_trading_calendar, validate_candles
from trading.logging_setup import setup_logging
from trading.reporting.metrics import buy_and_hold_benchmark
from trading.reporting.report import breakeven_header_ru
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import WalkForwardRunner, year_windows
from trading.settings import DEFAULT_SETTINGS_PATH, load_settings, load_universe
from trading.strategy.examples.monthly_ranked import (
    CrossSectionalMomentum,
    MeanReversion,
)

HYPOTHESES = [
    ("Моментум", CrossSectionalMomentum, {"lookback": [30, 60, 120], "top_n": [3, 4, 6]}),
    ("Возврат к среднему", MeanReversion, {"lookback": [5, 10, 21], "top_n": [3, 4, 6]}),
]


def main() -> int:
    setup_logging()
    cfg_dir = DEFAULT_SETTINGS_PATH.parent
    settings = load_settings(cfg_dir / "settings_crypto.yaml")
    tickers = load_universe(cfg_dir / "crypto_universe.yaml", uppercase=False)
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

    # Дробный размер лота под цену каждой монеты (по медиане).
    lot_sizes = {t: crypto_lot_size(float(df["close"].median())) for t, df in clean.items()}
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")
    btc = clean.get("tBTCUSD")

    all_dates = pd.concat([d["date"] for d in clean.values()])
    windows = year_windows(int(all_dates.min().year), int(all_dates.max().year), 4, 1)

    lines = [
        "# Криптовалюты: те же стратегии, другой рынок (Bitfinex)",
        "",
        f"Хеш данных: `{data_hash}`. Монет в тесте: {len(clean)} "
        f"(исключены по скачку >80%: {', '.join(excluded) or 'нет'}).",
        "",
        "```", breakeven_header_ru(settings), "```",
        "",
        f"Издержки крипты: комиссия {settings.costs.broker_commission_pct}% за "
        f"сторону, полуспред {settings.costs.half_spread_pct}%. Безрисковая "
        f"ставка {settings.benchmark.risk_free_rate:.1%}. Порог безубыточности "
        f"{settings.breakeven_rate():.1%} (против 38% на Мосбирже).",
        "",
    ]
    print("\n".join(lines))

    verdicts = []
    for title, cls, grid in HYPOTHESES:
        print(f"\n=== {title} (крипта) === {grid}")
        started = time.monotonic()
        result = WalkForwardRunner(
            candles=clean, lot_sizes=lot_sizes, settings=settings,
            strategy_factory=cls, param_grid=grid, ledger=ledger,
            data_hash=f"{data_hash}|crypto|{cls.__name__}", source="llm",
        ).run()
        print(result.report_ru)
        print(f"(время: {time.monotonic()-started:.0f} с)")
        verdicts.append((title, result))

        oos_start, oos_end = result.oos_equity.index[0], result.oos_equity.index[-1]
        btc_bm = None
        if btc is not None:
            seg = btc[(btc["date"] >= oos_start) & (btc["date"] <= oos_end)]
            if len(seg) >= 2:
                btc_bm = buy_and_hold_benchmark(seg.set_index("date")["close"])
        lines += [
            f"## {title}", "", "```", result.report_ru, "```", "",
            f"На том же OOS-периоде ({oos_start:%Y-%m-%d} … {oos_end:%Y-%m-%d}):",
            f"- Купил и держи BTC: "
            + (f"{btc_bm['annual_return']:+.1%} годовых" if btc_bm else "н/д"),
            f"- Фонд денежного рынка: {settings.benchmark.risk_free_rate:.1%} годовых",
            "",
        ]

    lines += [
        "## Вердикты", "",
    ]
    print("\nВЕРДИКТЫ:")
    for title, result in verdicts:
        print(f"\n{title}: {result.verdict_ru}")
        lines += [f"**{title}**", "", result.verdict_ru, ""]
    lines += [
        "## Ограничения", "",
        "- ВЫЖИВАЕМОСТЬ: только монеты, живые на Bitfinex в 2026. Тысячи "
        "обнулившихся токенов в выборке нет — результат ЗАВЫШЕН, и на крипте "
        "этот эффект куда сильнее, чем на акциях.",
        "- Исключение монет со скачком >80% убрало ранних чемпионов роста "
        "(LTC, EOS, XRP, ZEC) — это, наоборот, ЗАНИЖАЕТ моментум; чистого "
        "ответа без разметки реальных событий тут нет.",
        "- Проскальзывание на альткоинах в реальности много выше модельного; "
        "плечо и ликвидации не моделировались (только спот).",
    ]
    out = Path("docs/research_crypto.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nПолный отчёт: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
