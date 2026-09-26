"""Честный пересчёт трёх гипотез Мосбиржи: было / стало.

Запуск:
    python scripts/run_honest_recompute.py [--jobs 4]

Одна и та же walk-forward проверка в четырёх вариантах, каждый следующий
исправляет ещё одну ошибку прежних исследований:
A  — как было: 34 сегодняшние бумаги, свободные деньги под 0%, без
     дивидендов. Должен воспроизвести прежние +3,0 / +5,6 / +3,5% — проверка,
     что пересчёт ничего не сломал;
B  — + свободные деньги в фонде денежного рынка по исторической ставке ЦБ,
     + дивиденды после НДФЛ; сравнение с фактическим денежным рынком и MCFTR;
C1 — + исторический состав IMOEX: на каждую дату стратегия видит только
     бумаги, входившие тогда в индекс (включая ушедших с биржи); бумаги со
     скачками цены >35% исключены, как принято раньше;
C2 — то же, но все бумаги: скачки не исключаются (исключение обвалов само
     по себе завышает результат — та же ошибка выживаемости).
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.dividends import DividendsData, check_dividend_yields, ex_dividend_dates
from trading.data.moex_client import MarketData, MoexClient
from trading.data.rates import RatesData, money_market_benchmark_series, money_market_rate
from trading.data.universe import index_schedule
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

DATE_FROM, TILL = "2015-01-01", "2026-07-22"
RATES_FROM, RATES_TILL = "2013-09-13", "2026-09-26"
LEDGER = "journal/hypotheses.sqlite"
REPORT = Path("docs/research_honest_recompute.md")

HYPOTHESES = [
    ("Моментум", CrossSectionalMomentum, {"lookback": [60, 120, 250], "top_n": [2, 3, 4]}),
    ("Моментум с фильтром", AbsoluteMomentum, {"lookback": [120, 250], "top_n": [2, 4]}),
    ("Возврат к среднему", MeanReversion, {"lookback": [5, 10, 21], "top_n": [2, 3, 4]}),
]
PREVIOUS = {"Моментум": 0.030, "Моментум с фильтром": 0.056, "Возврат к среднему": 0.035}

VARIANTS = {
    "A": "как было: 34 сегодняшние бумаги, кэш под 0%, без дивидендов",
    "B": "+ кэш под историческую ставку ЦБ, + дивиденды после НДФЛ",
    "C1": "+ исторический состав IMOEX, бумаги со скачками исключены",
    "C2": "+ исторический состав IMOEX, все бумаги",
}

# Заполняется в main() до запуска пула: процессы-работники получают данные
# через fork, без повторной загрузки и сериализации.
DATA: dict = {}


def prepare(settings) -> dict:
    cache = ParquetCache(settings.data.cache_dir)
    client = MoexClient(requests_per_second=settings.data.requests_per_second,
                        timeout_seconds=settings.data.timeout_seconds,
                        retries=settings.data.retries)
    market = MarketData(client, cache)

    current = load_universe()
    schedule = index_schedule(market, DATE_FROM, TILL)
    members = sorted(set().union(*schedule.values))
    tickers = sorted(set(current) | set(members))
    frames = {t: market.daily_candles(t, DATE_FROM, TILL) for t in tickers}
    frames = {t: df for t, df in frames.items() if df is not None and not df.empty}
    calendar = build_trading_calendar(frames)
    cal = pd.DatetimeIndex(sorted(calendar))

    flagged = {}
    for t, df in frames.items():
        rep = validate_candles(df, t, jump_threshold=settings.data.price_jump_threshold,
                               trading_calendar=calendar,
                               confirmed_events=settings.data.confirmed_events)
        if rep.suspicious:
            flagged[t] = rep.price_jumps

    lots = market.lot_sizes()
    no_lot = sorted(t for t in frames if t not in lots)
    lot_sizes = {**{t: 1 for t in no_lot}, **lots}

    dd = DividendsData(cache)
    dividends, rejected, n_payments = {}, {}, 0
    for t, df in frames.items():
        ex = ex_dividend_dates(dd.history(t), cal)
        lo, hi = pd.Timestamp(df["date"].min()), pd.Timestamp(df["date"].max())
        ex = ex[(ex["ex_date"] > lo) & (ex["ex_date"] <= hi)]
        ex, bad = check_dividend_yields(ex, df)
        if bad:
            rejected[t] = bad
        if not ex.empty:
            dividends[t] = ex
            n_payments += len(ex)

    key_rate = RatesData(cache).key_rate(RATES_FROM, RATES_TILL)
    cash_rate = money_market_rate(key_rate, settings.benchmark.mm_spread_pp, end=TILL)

    indices = {}
    for name, board in (("IMOEX", "SNDX"), ("MCFTR", "SNDX"), ("MCFTRR", "RTSI")):
        try:
            df = market.index_candles(DATE_FROM, TILL, name, board)
        except Exception:
            df = None
        if df is not None and not df.empty:
            indices[name] = df.assign(date=pd.to_datetime(df["date"])).set_index("date")["close"]
    data_hash = cache.data_version_hash()
    client.close()

    last_day = max(calendar)
    delisted = sorted(t for t in members if t in frames and
                      pd.Timestamp(frames[t]["date"].max()) < last_day - pd.Timedelta(days=30))
    a_universe = [t for t in current if t in frames and t not in flagged]
    c_all = [t for t in members if t in frames]
    extras = {"cash_rate": cash_rate, "dividends": dividends}
    return {
        "variants": {
            "A": ({t: frames[t] for t in a_universe}, {}),
            "B": ({t: frames[t] for t in a_universe}, dict(extras)),
            "C1": ({t: frames[t] for t in c_all if t not in flagged},
                   {**extras, "universe_schedule": schedule}),
            "C2": ({t: frames[t] for t in c_all}, {**extras, "universe_schedule": schedule}),
        },
        "lot_sizes": lot_sizes, "schedule": schedule, "members": members,
        "current": current, "flagged": flagged, "no_lot": no_lot, "delisted": delisted,
        "dividends": dividends, "rejected": rejected, "n_payments": n_payments,
        "cash_rate": cash_rate, "indices": indices, "data_hash": data_hash,
        "a_universe": a_universe,
    }


def run_job(job: tuple[str, int]) -> dict:
    variant, h = job
    title, cls, grid = HYPOTHESES[h]
    candles, engine_kwargs = DATA["variants"][variant]
    settings = load_settings()
    started = time.monotonic()
    result = WalkForwardRunner(
        candles=candles, lot_sizes=DATA["lot_sizes"], settings=settings,
        strategy_factory=cls, param_grid=grid,
        ledger=HypothesisLedger(LEDGER),
        data_hash=f"{DATA['data_hash']}|{cls.__name__}|пересчёт-{variant}",
        source="llm", engine_kwargs=engine_kwargs,
    ).run()
    elapsed = time.monotonic() - started
    print(f"  готово: {variant} / {title} — OOS {result.oos_metrics['annual_return']:+.1%} "
          f"({elapsed/60:.0f} мин)", flush=True)
    return {"variant": variant, "title": title, "metrics": result.oos_metrics,
            "mm": result.mm_oos_annual, "report": result.report_ru,
            "verdict": result.verdict_ru, "equity": result.oos_equity,
            "median": result.median_oos_annual, "best": result.best_oos_annual,
            "n_configs": result.n_configs, "elapsed": elapsed}


def pct(x: float) -> str:
    return f"{x:+.1%}".replace(".", ",")


def rub(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def build_report(settings, results: list[dict]) -> str:
    d = DATA
    eq = results[0]["equity"]
    oos_start, oos_end = eq.index[0], eq.index[-1]
    mm = money_market_benchmark_series(1.0, oos_start, oos_end, d["cash_rate"])
    idx_bm = {}
    for name, s in d["indices"].items():
        seg = s[(s.index >= oos_start) & (s.index <= oos_end)]
        idx_bm[name] = buy_and_hold_benchmark(seg)["annual_return"]
    infra = settings.benchmark.infra_cost_rub_year / settings.capital.start_amount
    threshold = mm["annual_return"] + infra
    by_key = {(r["variant"], r["title"]): r for r in results}

    lines = [
        "# Честный пересчёт: было / стало",
        "",
        f"Дата прогона: {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
        f"Хеш версии данных: `{d['data_hash']}`.",
        "",
        "Те же три гипотезы, та же walk-forward схема (обучение 4 г. → проверка 1 г.), "
        "тот же капитал 20 000 ₽ и издержки. Меняется только честность симуляции.",
        "",
        "| Вариант | Что исправлено |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in VARIANTS.items()],
        "",
        f"## Бенчмарки на том же OOS-периоде ({oos_start:%Y-%m-%d} … {oos_end:%Y-%m-%d})",
        "",
        f"- Фонд денежного рынка по фактической ставке ЦБ − "
        f"{settings.benchmark.mm_spread_pp:g} п.п.: **{pct(mm['annual_return'])} годовых** "
        f"(средняя ставка {mm['average_rate']:.2%}).",
    ]
    names = {"IMOEX": "IMOEX (только цены, без дивидендов)",
             "MCFTR": "MCFTR (с дивидендами, без налога)",
             "MCFTRR": "MCFTRR (с дивидендами после налога) — честный «купил и держи индекс»"}
    for name, v in idx_bm.items():
        lines.append(f"- {names[name]}: {pct(v)} годовых (без комиссий фонда).")
    lines += [
        f"- Порог ТЗ (денежный рынок + сервер "
        f"{rub(settings.benchmark.infra_cost_rub_year)} ₽/год на капитал "
        f"{rub(settings.capital.start_amount)} ₽): {pct(threshold)} годовых.",
        "",
        "## Итог: было / стало (OOS, годовых)",
        "",
        "| Гипотеза | Раньше | A | B | C1 | C2 | Денежный рынок |",
        "|---|---|---|---|---|---|---|",
    ]
    for title, _, _ in HYPOTHESES:
        cells = [pct(by_key[(v, title)]["metrics"]["annual_return"])
                 if (v, title) in by_key else "—" for v in VARIANTS]
        lines.append(f"| {title} | {pct(PREVIOUS[title])} | " + " | ".join(cells)
                     + f" | {pct(mm['annual_return'])} |")
    lines += [
        "",
        "**Почему A не равен «Раньше».** Прежние цифры получены до исправления "
        "9e453d4: потолок одной заявки был абсолютным (6 000 ₽), и когда капитал "
        "в проверочном окне вырастал выше ~24 000 ₽, обычная заявка его превышала — "
        "срабатывал предохранитель «ошибка в единицах» и система ложно "
        "останавливалась до конца окна, случайно пропуская плохие периоды. "
        "Исправление сделали для крипты, но отчёты Мосбиржи не пересчитали. "
        "Проверено: код до 9e453d4 на тех же данных даёт ровно прежние +5,6%, "
        "код после — −0,1%. Прежние цифры были завышены артефактом.",
        "",
        "## Подробно по вариантам",
        "",
        "Walk-forward каждый год выбирает конфигурацию по обучению — это одна "
        "цепочка выборов, и ей может повезти. Медиана по всем конфигурациям "
        "показывает типичный результат идеи; доверять надо ей (так требует "
        "поправка на множественное тестирование).",
        "",
        "| Вариант | Гипотеза | OOS годовых | Медиана конфигураций | Итог, ₽ | Макс. просадка | "
        "Шарп* | Дивиденды нетто, ₽ | Доход на кэш, ₽ | Против денежного рынка |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in VARIANTS:
        for title, _, _ in HYPOTHESES:
            r = by_key.get((v, title))
            if r is None:
                continue
            m = r["metrics"]
            gap = m["annual_return"] - mm["annual_return"]
            lines.append(
                f"| {v} | {title} | {pct(m['annual_return'])} | {pct(r['median'])} | "
                f"{rub(m['end_equity'])} | "
                f"{pct(m['max_drawdown'])} | {m['sharpe']:.2f} | "
                f"{rub(m.get('dividends_net', 0))} | {rub(m.get('cash_interest', 0))} | "
                f"{f'{gap * 100:+.1f}'.replace('.', ',')} п.п. |"
            )
    lines += [
        "",
        "\\* Шарп в варианте A — к постоянной ставке "
        f"{settings.benchmark.risk_free_rate:.2%} (как раньше), в B/C — к исторической "
        "ставке ЦБ.",
        "",
        "## Данные исторического универсума",
        "",
        f"- Участников IMOEX за {DATE_FROM[:4]}–{TILL[:4]}: {len(d['members'])} "
        f"(раньше исследования шли по {len(d['current'])} сегодняшним бумагам).",
        f"- Ушли с торгов до конца периода ({len(d['delisted'])}): "
        f"{', '.join(d['delisted'])}. Позиция в такой бумаге после последних торгов "
        "замораживается по последней цене — для бумаг, выкупленных или "
        "конвертированных, это близко к правде; для остальных — оптимистично.",
        f"- Без сегодняшнего размера лота, взят лот 1 ({len(d['no_lot'])}): "
        f"{', '.join(d['no_lot'])}. Лот 1 — мягче реальности (дробнее сайзинг).",
        f"- Дивиденды: {len(d['dividends'])} бумаг, {d['n_payments']} выплат с "
        "экс-датой в период торгов бумагой. Бумаги без найденных выплат считаются "
        "бездивидендными (занижение).",
        "- Отброшены как неправдоподобные (сайт дал дивиденд без поправки на сплит, "
        "ISS — цену с поправкой):",
    ]
    if d["rejected"]:
        for t, bad in sorted(d["rejected"].items()):
            lines.append(f"  - {t}: {'; '.join(bad)}")
    else:
        lines.append("  - нет")
    lines += [
        f"- Скачки цены >{settings.data.price_jump_threshold:.0%} "
        f"(кроме подтверждённого 24.02.2022), исключены в C1, оставлены в C2:",
    ]
    for t, jumps in sorted(d["flagged"].items()):
        lines.append(f"  - {t}: " + ", ".join(f"{dt} {c:+.0%}" for dt, c in jumps))
    lines += ["", "## Полные отчёты walk-forward", ""]
    for v in VARIANTS:
        for title, _, _ in HYPOTHESES:
            r = by_key.get((v, title))
            if r is None:
                continue
            lines += [f"### {v} — {title}", "", "```", r["report"], "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    setup_logging()
    settings = load_settings()

    print("Подготовка данных…", flush=True)
    DATA.update(prepare(settings))
    for v, (candles, kw) in DATA["variants"].items():
        print(f"  {v}: {len(candles)} бумаг, доп.: {sorted(kw) or 'нет'}", flush=True)
    print(breakeven_header_ru(settings), flush=True)

    variants = [v for v in args.variants.split(",") if v in VARIANTS]
    # Самые долгие (много бумаг) — первыми, чтобы ядра не простаивали в конце.
    jobs = [(v, h) for v in reversed(variants) for h in range(len(HYPOTHESES))]
    print(f"Прогонов walk-forward: {len(jobs)}, параллельно: {args.jobs}", flush=True)
    with Pool(args.jobs) as pool:
        results = pool.map(run_job, jobs, chunksize=1)

    REPORT.write_text(build_report(settings, results), encoding="utf-8")
    print(f"\nОтчёт: {REPORT}")
    print("\n".join(REPORT.read_text(encoding="utf-8").split("## Подробно")[0].splitlines()[-12:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
