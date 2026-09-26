"""Архитектура v2: распределение между фондами по прогнозу риска.

Запуск:
    python scripts/download_v2_data.py      # один раз: данные в кэш
    python scripts/run_v2_allocator.py

Что проверяется:
1. Предсказуема ли просадка индекса акций на месяц вперёд (AUC по годам
   2008–2026, только out-of-sample, ежегодное переобучение с очисткой).
   Сравнение с «только волатильностью» — если ML не лучше одного признака,
   ML не нужен (лестница сложности ТЗ, раздел 20).
2. Даёт ли распределение «акции, пока риск обычный; иначе деньги/ОФЗ»
   больше фонда денежного рынка — и с какой просадкой.

Решение пересматривается раз в неделю (пятница), сделка — на следующий
торговый день. Такую систему можно вести вручную: сервер не нужен.

Сетка ML-правила (3 параметра — лимит ТЗ) и основная конфигурация
зафиксированы ДО прогона (см. PRE_REGISTERED): иначе выбор лучшей из
12 по результату был бы подгонкой.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from trading.allocation.assets import ASSETS, BONDS, CASH, EQUITY, fund_returns
from trading.allocation.risk_model import (
    FEATURE_NAMES_RU,
    auc,
    drawdown_labels,
    risk_features,
    walk_forward_risk,
)
from trading.allocation.simulator import (
    easing_regime,
    ml_risk_targets,
    rest_asset,
    simulate,
    static_targets,
    trend_targets,
    vol_targets,
)
from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexClient
from trading.data.rates import RatesData, money_market_rate
from trading.logging_setup import setup_logging
from trading.ml.validation import cscv_pbo
from trading.reporting.metrics import (
    compute_metrics,
    deflated_sharpe_ratio,
    per_period_sharpe,
)
from trading.research.ledger import HypothesisLedger
from trading.settings import load_settings

DATE_FROM, DATE_TILL = "2003-01-01", "2026-09-25"
RATES_FROM, RATES_TILL = "2013-09-13", "2026-09-26"
FIRST_MODEL_YEAR = 2008          # первый проверочный год модели (обучение 2004–2007)
ALLOC_START = "2014-01-01"       # с этой даты есть ключевая ставка → денежный рынок
COMPARE_START = "2019-01-03"     # период прежних исследований — для сравнения
COMPARE_END = "2026-07-22"
HORIZON, DEPTH = 21, 0.05        # «просадка ≥5% в ближайший месяц»
EQUITY_TER, BOND_TER = 0.008, 0.006
REPORT = Path("docs/research_v2_allocator.md")

GRID = {"k": [1.0, 1.5, 2.0], "w_max": [0.5, 1.0], "rest": [CASH, "офз_при_снижении"]}
PRE_REGISTERED = {"k": 1.0, "w_max": 1.0, "rest": CASH}


def pct(x: float, sign: bool = True) -> str:
    return (f"{x:+.1%}" if sign else f"{x:.1%}").replace(".", ",")


def rub(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def main() -> int:
    setup_logging()
    settings = load_settings()
    cache = ParquetCache(settings.data.cache_dir)
    market = MarketData(MoexClient(), cache)
    idx = {n: market.index_history(n, DATE_FROM, DATE_TILL).set_index("date")["close"]
           for n in ("MCFTRR", "RGBITR", "MCFTR", "IMOEX")}
    key_rate = RatesData(cache).key_rate(RATES_FROM, RATES_TILL)
    cash_rate = money_market_rate(key_rate, settings.benchmark.mm_spread_pp, end=DATE_TILL)
    c = settings.costs
    cost_rate = (c.broker_commission_pct + c.exchange_fee_pct + c.half_spread_pct
                 + c.base_slippage_pct) / 100.0 * c.stress_multiplier
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")

    eq, bd = idx["MCFTRR"], idx["RGBITR"]
    # --- 1. Модель риска ------------------------------------------------
    feats = risk_features(eq, bd)
    labels = drawdown_labels(eq, HORIZON, DEPTH)
    fc = walk_forward_risk(feats, labels, FIRST_MODEL_YEAR, HORIZON)
    # Ступень ниже по лестнице сложности: та же модель на одном признаке.
    fc1 = walk_forward_risk(feats[["vol20"]], labels, FIRST_MODEL_YEAR, HORIZON)
    y_all = labels.reindex(fc.proba.index)
    known = y_all.notna()
    auc_model = auc(y_all[known].to_numpy(), fc.proba[known].to_numpy())
    auc_one = auc(y_all[known].to_numpy(), fc1.proba.reindex(fc.proba.index)[known].to_numpy())
    auc_vol = auc(y_all[known].to_numpy(), feats["vol20"].reindex(fc.proba.index)[known].to_numpy())
    print(f"Модель риска OOS {FIRST_MODEL_YEAR}–2026: AUC {auc_model:.3f} "
          f"(1 признак: {auc_one:.3f}, сырая волатильность: {auc_vol:.3f})", flush=True)

    # --- 2. Распределение -----------------------------------------------
    rets = fund_returns(eq, bd, cash_rate, EQUITY_TER, BOND_TER, start=ALLOC_START)
    days = rets.index
    easing = easing_regime(key_rate, days)
    proba = fc.proba.reindex(days)
    base = fc.base_rate.reindex(days)

    def run(targets: pd.DataFrame) -> dict:
        sim = simulate(rets, targets, cost_rate, settings.capital.start_amount)
        m = compute_metrics(sim.equity, settings.benchmark.risk_free_rate, rf_series=cash_rate)
        cmp = sim.equity[(sim.equity.index >= COMPARE_START) & (sim.equity.index <= COMPARE_END)]
        m_cmp = compute_metrics(cmp, settings.benchmark.risk_free_rate, rf_series=cash_rate)
        yearly = sim.equity.resample("YE").last()
        yearly = pd.concat([pd.Series([sim.equity.iloc[0]], index=[days[0]]), yearly])
        year_ret = yearly.pct_change().dropna()
        return {"sim": sim, "m": m, "m_cmp": m_cmp, "year_ret": year_ret,
                "monthly": sim.equity.resample("ME").last().pct_change().dropna()}

    rules = {
        "100% денежный рынок": static_targets(days),
        "100% фонд акций (MCFTRR − 0,8%)": static_targets(days, equity=1.0),
        "100% фонд ОФЗ (RGBITR − 0,6%)": static_targets(days, bonds=1.0),
        "50% акций / 50% денег": static_targets(days, equity=0.5),
        "60% акций / 40% ОФЗ": static_targets(days, equity=0.6, bonds=0.4),
        "Правило: акции выше 200-дневной средней": trend_targets(eq, days, CASH),
        "Правило: волатильность ниже годовой медианы": vol_targets(feats["vol20"], days, CASH),
    }
    results = {name: run(t) for name, t in rules.items()}

    ml_names, one_names = [], []
    for k, w_max, rest in product(GRID["k"], GRID["w_max"], GRID["rest"]):
        name = f"ML: k={k:g}, акций до {w_max:.0%}, остаток → {rest}"
        t = ml_risk_targets(proba, base, k, w_max, rest_asset(rest, easing, days))
        results[name] = run(t)
        results[name]["params"] = {"k": k, "w_max": w_max, "rest": rest}
        ml_names.append(name)
    proba1, base1 = fc1.proba.reindex(days), fc1.base_rate.reindex(days)
    for k, w_max, rest in product(GRID["k"], GRID["w_max"], GRID["rest"]):
        name = f"1 признак: k={k:g}, акций до {w_max:.0%}, остаток → {rest}"
        t = ml_risk_targets(proba1, base1, k, w_max, rest_asset(rest, easing, days))
        results[name] = run(t)
        results[name]["params"] = {"k": k, "w_max": w_max, "rest": rest, "модель": "vol20"}
        one_names.append(name)
    main_name = next(n for n in ml_names if results[n]["params"] == PRE_REGISTERED)

    for name, r in results.items():
        ledger.record(
            hypothesis=f"v2 распределение между фондами: {name}", source="llm",
            params=r.get("params", {"правило": name}),
            data_hash=f"{data_hash}|v2|{ALLOC_START}",
            metrics_oos={k: r["m"].get(k) for k in ("annual_return", "max_drawdown", "sharpe")},
        )

    # --- 3. Проверка на переподгонку -------------------------------------
    monthly = pd.DataFrame({n: results[n]["monthly"] for n in ml_names + one_names}).dropna()
    pbo = cscv_pbo(monthly, n_partitions=16)
    tested = list(results)                      # все испытанные варианты, включая правила
    trial_sharpes = [per_period_sharpe(results[n]["monthly"]) for n in tested
                     if not n.startswith(("100%", "50%", "60%"))]
    mm_monthly = results["100% денежный рынок"]["monthly"]
    excess_main = (results[main_name]["monthly"] - mm_monthly).dropna()
    excess_trials = [per_period_sharpe((results[n]["monthly"] - mm_monthly).dropna())
                     for n in tested if not n.startswith(("100%", "50%", "60%"))]
    dsr = deflated_sharpe_ratio(excess_main, excess_trials)

    # --- 4. Отчёт --------------------------------------------------------
    mm = results["100% денежный рынок"]
    lines = [
        "# Архитектура v2: распределение между фондами по прогнозу риска",
        "",
        f"Дата прогона: {pd.Timestamp.now():%Y-%m-%d %H:%M}. Хеш данных: `{data_hash}`. "
        f"Данные по {pd.Timestamp(rets.index[-1]).date()}.",
        "",
        "## Идея одной строкой",
        "",
        "Держать фонд на индекс акций, пока модель оценивает риск месячной просадки "
        "≥5% как обычный; когда риск выше обычного — переходить в фонд денежного "
        "рынка (или ОФЗ, если ЦБ снижает ставку). Решение — раз в неделю.",
        "",
        "## 1. Можно ли предсказать просадку?",
        "",
        f"Вопрос модели: упадёт ли индекс MCFTRR на {DEPTH:.0%} ниже сегодняшнего уровня "
        f"хотя бы раз за {HORIZON} торговый день. Логистическая регрессия на "
        f"{len(FEATURE_NAMES_RU)} нормализованных признаках, переобучение раз в год "
        f"только на прошлом, с очисткой меток на границе.",
        "",
        f"**AUC вне обучения {FIRST_MODEL_YEAR}–2026: {auc_model:.3f}** (0,5 — монетка; "
        f"в среднем по годам {fc.yearly['AUC модели'].mean():.3f}). "
        f"Та же модель на одном признаке «волатильность за месяц»: {auc_one:.3f} "
        f"(по годам {fc1.yearly['AUC модели'].mean():.3f}); "
        f"сама волатильность без модели: {auc_vol:.3f}.",
        "",
        "Лестница сложности (ТЗ, раздел 20): модель на 9 признаках обязана быть лучше "
        "простейшей ступени. Если нет — используется простейшая.",
        "",
        f"Калибровка вероятностей отключалась в {int(fc.yearly['калибровка отключена'].sum())} "
        f"годах из {len(fc.yearly)}: на отложенном хвосте обучения связь развернулась, и "
        "калибратор перевернул бы порядок оценок (найденная и исправленная ошибка — "
        "в первом прогоне v2 это роняло AUC до 0,59). В такие годы используется только "
        "сдвиг уровня вероятности, порядок оценок сохраняется.",
        "",
        "| Год | AUC 9 признаков | AUC 1 признака | AUC волатильности | Доля опасных дней | В обучении |",
        "|---|---|---|---|---|---|",
    ]
    for year, row in fc.yearly.iterrows():
        one = fc1.yearly["AUC модели"].get(year, float("nan"))
        lines.append(f"| {year} | {row['AUC модели']:.2f} | {one:.2f} | "
                     f"{row['AUC только волатильности']:.2f} | "
                     f"{row['частота просадок в году']:.0%} | {row['частота в обучении']:.0%} |")
    last_year = max(fc.explanations)
    lines += [
        "",
        f"Последнее решение модели ({pd.Timestamp(fc.proba.index[-1]).date()}): "
        f"{fc.explanations[last_year]}",
        "",
        "## 2. Результат распределения",
        "",
        f"Период: {pd.Timestamp(days[0]).date()} … {pd.Timestamp(days[-1]).date()} "
        f"(с 2014 года — есть ключевая ставка). Капитал {rub(settings.capital.start_amount)} ₽. "
        f"Издержки {cost_rate:.2%} от оборота на каждую сторону; комиссии фондов вычтены.",
        "",
        "| Вариант | Годовых | Итог, ₽ | Макс. просадка | Худший год | Убыточных лет | "
        "Шарп* | Сделок | Доля акций | Годовых 2019–07.2026 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(name: str, r: dict, bold: bool = False) -> str:
        m, sim, yr = r["m"], r["sim"], r["year_ret"]
        label = f"**{name}**" if bold else name
        return (f"| {label} | {pct(m['annual_return'])} | {rub(m['end_equity'])} | "
                f"{pct(m['max_drawdown'])} | {pct(yr.min())} | {(yr < 0).sum()} из {len(yr)} | "
                f"{m['sharpe']:.2f} | {sim.n_rebalances} | {sim.equity_share_avg:.0%} | "
                f"{pct(r['m_cmp']['annual_return'])} |")

    for name, r in results.items():
        if name not in ml_names and name not in one_names:
            lines.append(row(name, r))
    lines.append(row(main_name + " — основная (заявлена заранее)", results[main_name], bold=True))
    for name in ml_names:
        if name != main_name:
            lines.append(row(name, results[name]))
    lines.append("| *Добавлено после просмотра результатов (ступень ниже):* | | | | | | | | | |")
    for name in one_names:
        lines.append(row(name, results[name]))
    ml_annual = [results[n]["m"]["annual_return"] for n in ml_names]
    one_annual = [results[n]["m"]["annual_return"] for n in one_names]
    beat = sum(a > mm["m"]["annual_return"] for a in ml_annual)
    beat1 = sum(a > mm["m"]["annual_return"] for a in one_annual)
    lines += [
        "",
        "\\* Шарп — к исторической ставке денежного рынка (0 = как денежный рынок).",
        "",
        f"ML на 9 признаках, {len(ml_names)} вариантов: медиана "
        f"{pct(float(np.median(ml_annual)))}, от {pct(min(ml_annual))} до "
        f"{pct(max(ml_annual))} годовых; денежный рынок обогнали {beat} из {len(ml_names)}.",
        f"Модель на 1 признаке, {len(one_names)} вариантов: медиана "
        f"{pct(float(np.median(one_annual)))}, от {pct(min(one_annual))} до "
        f"{pct(max(one_annual))}; денежный рынок обогнали {beat1} из {len(one_names)}.",
        "",
        "## 3. Проверка на случайность",
        "",
        f"- {pbo['verdict_ru']} (по месячной доходности {len(ml_names) + len(one_names)} "
        f"вариантов модели, {pbo['n_combinations']} разбиений).",
        f"- Основная конфигурация против денежного рынка: {dsr['verdict_ru']}",
        "",
        "## 4. Годы основной конфигурации и бенчмарков",
        "",
        "| Год | Основная | Денежный рынок | Фонд акций | Фонд ОФЗ | Доля акций в среднем |",
        "|---|---|---|---|---|---|",
    ]
    main = results[main_name]
    eq_share = main["sim"].weights[EQUITY].resample("YE").mean()
    for d, v in main["year_ret"].items():
        def yr(n):
            s = results[n]["year_ret"]
            return pct(s.get(d, float("nan")))
        lines.append(f"| {d.year} | {pct(v)} | {yr('100% денежный рынок')} | "
                     f"{yr('100% фонд акций (MCFTRR − 0,8%)')} | {yr('100% фонд ОФЗ (RGBITR − 0,6%)')} | "
                     f"{eq_share.get(d, float('nan')):.0%} |")
    lines += [
        "",
        "## Ограничения",
        "",
        "- Фонды смоделированы индексами минус комиссия фонда: реальный фонд отстаёт "
        "от индекса ещё на ошибку слежения (обычно 0,1–0,5% в год).",
        "- Налог: при каждом переключении с прибылью платится НДФЛ 13% — в расчёте "
        "не вычитается ни у стратегии, ни у бенчмарков (на ИИС-3 это верно; на обычном "
        "счёте стратегия с переключениями теряет отсрочку налога, которая есть у "
        "«купил и держи»).",
        "- 2022: биржа не работала с 25.02 по 24.03 — выйти из акций было невозможно; "
        "в расчёте, как и в жизни, сделка исполняется при открытии торгов.",
        "- Доходность денежного рынка — ключевая ставка минус "
        f"{settings.benchmark.mm_spread_pp:g} п.п. (как у фондов LQDT/SBMM).",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[lines.index("## 2. Результат распределения"):]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
