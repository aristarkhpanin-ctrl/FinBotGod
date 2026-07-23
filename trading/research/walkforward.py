"""Walk-forward валидация — ФАЗА 5 (ТЗ, раздел 8).

Бэктест на всей истории с параметрами, подобранными на этой же истории,
ничего не доказывает. Схема:

    Окно 1: обучение 2015–2018 → проверка 2019
    Окно 2: обучение 2016–2019 → проверка 2020
    ...

Правила, зашитые в код:
* параметры подбираются ТОЛЬКО на обучающем окне и применяются
  к проверочному без единого изменения;
* итоговые метрики считаются исключительно по склейке проверочных окон
  (in-sample результаты в отчёт не попадают вовсе);
* не более 3 оптимизируемых параметров — больше означает подгонку,
  и это ошибка, а не предупреждение;
* каждая проверенная конфигурация автоматически попадает в журнал
  гипотез (это делает движок бэктеста, обойти нельзя);
* отчёт всегда показывает медиану по всем конфигурациям, а не только
  лучшую, с поправкой на множественное тестирование;
* вердикт по критерию остановки проекта (ТЗ, раздел 15) печатается
  в конце каждого отчёта.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field

import pandas as pd

from trading.engine.backtest import BacktestEngine
from trading.formatting import fmt_rub
from trading.logging_setup import get_logger
from trading.reporting.metrics import annualize, compute_metrics
from trading.research.ledger import HypothesisLedger
from trading.settings import Settings

log = get_logger("walkforward")


class ParamLimitError(Exception):
    """Больше 3 оптимизируемых параметров — это подгонка, а не стратегия."""


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def label(self) -> str:
        return (
            f"обучение {self.train_start.year}–{self.train_end.year} "
            f"→ проверка {self.test_start.year}"
            + (f"–{self.test_end.year}" if self.test_end.year != self.test_start.year else "")
        )


def year_windows(
    first_year: int, last_year: int, train_years: int = 4, test_years: int = 1
) -> list[WalkForwardWindow]:
    """Скользящие годовые окна: обучение train_years лет, проверка test_years."""
    windows = []
    y = first_year
    while y + train_years + test_years - 1 <= last_year:
        windows.append(
            WalkForwardWindow(
                train_start=pd.Timestamp(y, 1, 1),
                train_end=pd.Timestamp(y + train_years - 1, 12, 31),
                test_start=pd.Timestamp(y + train_years, 1, 1),
                test_end=pd.Timestamp(y + train_years + test_years - 1, 12, 31),
            )
        )
        y += test_years
    return windows


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """Раскрывает сетку параметров в список конфигураций. Максимум 3 параметра."""
    if len(param_grid) > 3:
        raise ParamLimitError(
            f"Оптимизируемых параметров: {len(param_grid)}, разрешено не более 3 "
            f"(ТЗ, раздел 8). Каждый лишний параметр экспоненциально увеличивает "
            f"риск подгонки. Если стратегии нужно столько параметров, чтобы "
            f"работать, — она не работает."
        )
    if not param_grid:
        return [{}]
    keys = sorted(param_grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(param_grid[k] for k in keys))
    ]


def stop_criterion_verdict_ru(
    oos_annual: float, settings: Settings, safety_margin: float = 0.10
) -> str:
    """Критерий остановки проекта (ТЗ, раздел 15) — прямым текстом.

    ``safety_margin`` — требуемый запас над порогом безубыточности,
    в процентных пунктах годовых (0.10 = 10 пп).
    """
    rf = settings.benchmark.risk_free_rate
    breakeven = settings.breakeven_rate()
    required = breakeven + safety_margin
    if oos_annual is None or (isinstance(oos_annual, float) and math.isnan(oos_annual)):
        return "ВЕРДИКТ: OOS-данных недостаточно — выводы делать не по чему."
    shown = f"{oos_annual:.1%}"
    if oos_annual < rf:
        return (
            f"ВЕРДИКТ: OOS-доходность {shown} годовых ПРОИГРЫВАЕТ безрисковой "
            f"ставке {rf:.2%}. ПРОЕКТ ОСТАНАВЛИВАЕТСЯ (ТЗ, раздел 15): фонд "
            f"денежного рынка даёт больше без единой строчки кода. Это результат "
            f"исследования, полученный бесплатно, а не поражение."
        )
    if oos_annual < breakeven:
        return (
            f"ВЕРДИКТ: OOS-доходность {shown} годовых выше ставки, но НЕ окупает "
            f"инфраструктуру (порог безубыточности {breakeven:.0%}). "
            f"ПРОЕКТ ОСТАНАВЛИВАЕТСЯ (ТЗ, раздел 15)."
        )
    if oos_annual < required:
        return (
            f"ВЕРДИКТ: OOS-доходность {shown} годовых выше порога "
            f"{breakeven:.0%}, но БЕЗ ЗАПАСА ПРОЧНОСТИ (требуется "
            f"{required:.0%}+). ПРОЕКТ ОСТАНАВЛИВАЕТСЯ (ТЗ, раздел 15): "
            f"стратегия без запаса умрёт при первом же росте издержек."
        )
    return (
        f"ВЕРДИКТ: OOS-доходность {shown} годовых превышает порог "
        f"{breakeven:.0%} с запасом (требовалось {required:.0%}+). "
        f"Критерий остановки НЕ сработал — можно переходить к следующей фазе. "
        f"Перед решением прогнать стресс-режим с удвоенными издержками."
    )


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    chosen_params_by_window: list[dict]
    oos_equity: pd.Series          # склейка проверочных окон, капитал переносится
    oos_metrics: dict              # метрики ТОЛЬКО по out-of-sample
    n_configs: int
    configs_oos_annual: dict[str, float]   # конфигурация → OOS-доходность годовых
    best_oos_annual: float
    median_oos_annual: float
    verdict_ru: str
    report_ru: str = ""
    limitations: list[str] = field(default_factory=list)


class WalkForwardRunner:
    def __init__(
        self,
        candles: dict[str, pd.DataFrame],
        lot_sizes: dict[str, int],
        settings: Settings,
        strategy_factory,           # callable(**params) -> Strategy
        param_grid: dict[str, list],
        ledger: HypothesisLedger,
        data_hash: str,
        train_years: int = 4,
        test_years: int = 1,
        select_metric: str = "annual_return",
        safety_margin: float = 0.10,
        source: str = "перебор",     # источник гипотезы: человек | llm | перебор
    ):
        self.combos = expand_grid(param_grid)   # проверка лимита — сразу
        self.candles = {}
        for secid, df in candles.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
            self.candles[secid] = d.sort_values("date").reset_index(drop=True)
        if not self.candles:
            raise ValueError("Walk-forward без данных невозможен: нет свечей.")
        self.lot_sizes = lot_sizes
        self.settings = settings
        self.strategy_factory = strategy_factory
        self.ledger = ledger
        self.data_hash = data_hash
        self.train_years = train_years
        self.test_years = test_years
        self.select_metric = select_metric
        self.safety_margin = safety_margin
        self.source = source

        all_dates = pd.concat([d["date"] for d in self.candles.values()])
        self.first_year = int(all_dates.min().year)
        self.last_year = int(all_dates.max().year)

    def _slice(self, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
        out = {}
        for secid, df in self.candles.items():
            d = df[(df["date"] >= start) & (df["date"] <= end)]
            if not d.empty:
                out[secid] = d
        return out

    def _run_engine(self, candles, params, capital, tag) -> "BacktestResult":
        settings = self.settings.model_copy(deep=True)
        settings.capital.start_amount = capital
        engine = BacktestEngine(
            candles=candles,
            lot_sizes=self.lot_sizes,
            settings=settings,
            strategy=self.strategy_factory(**params),
            ledger=self.ledger,
            data_hash=f"{self.data_hash}|{tag}",
            source=self.source,
        )
        return engine.run()

    def run(self) -> WalkForwardResult:
        s = self.settings
        windows = year_windows(
            self.first_year, self.last_year, self.train_years, self.test_years
        )
        if not windows:
            raise ValueError(
                f"Данных {self.first_year}–{self.last_year} не хватает даже на "
                f"одно окно (обучение {self.train_years} лет + проверка "
                f"{self.test_years}). Walk-forward невозможен."
            )

        chosen_by_window: list[dict] = []
        window_lines: list[str] = []
        limitations: list[str] = []
        oos_segments: list[pd.Series] = []
        oos_fills: list = []
        carried_capital = s.capital.start_amount
        # OOS-результаты каждой конфигурации (фиксированный капитал) —
        # для медианы и лучшей задним числом.
        combo_factors: dict[str, list[float]] = {self._key(c): [] for c in self.combos}
        combo_days: dict[str, float] = {self._key(c): 0.0 for c in self.combos}

        for w in windows:
            train = self._slice(w.train_start, w.train_end)
            test = self._slice(w.test_start, w.test_end)
            if not train or not test:
                limitations.append(f"Окно «{w.label}» пропущено: нет данных.")
                continue

            # 1. Подбор ТОЛЬКО на обучении.
            best_params, best_metric = None, None
            for params in self.combos:
                r = self._run_engine(
                    train, params, s.capital.start_amount,
                    f"train|{w.label}",
                )
                metric = r.metrics.get(self.select_metric)
                if metric is None or (isinstance(metric, float) and math.isnan(metric)):
                    metric = float("-inf")
                if best_metric is None or metric > best_metric:
                    best_params, best_metric = params, metric

            # 2. Все конфигурации на проверке — для распределения OOS.
            for params in self.combos:
                r = self._run_engine(
                    test, params, s.capital.start_amount,
                    f"test|{w.label}",
                )
                key = self._key(params)
                combo_factors[key].append(
                    r.metrics["end_equity"] / r.metrics["start_equity"]
                )
                combo_days[key] += (w.test_end - w.test_start).days

            # 3. Выбранная конфигурация на проверке с переносом капитала —
            #    непрерывная OOS-кривая.
            chosen = self._run_engine(
                test, best_params, carried_capital, f"oos|{w.label}"
            )
            carried_capital = float(chosen.equity.iloc[-1])
            oos_segments.append(chosen.equity)
            oos_fills.extend(chosen.fills)
            chosen_by_window.append(best_params)
            window_lines.append(
                f"  {w.label} | выбрано: {best_params or 'без параметров'} | "
                f"доходность окна: {chosen.metrics['total_return']:+.1%}"
            )

        if not oos_segments:
            raise ValueError("Ни одно окно не дало OOS-результата — данных нет.")

        oos_equity = pd.concat(oos_segments)
        oos_metrics = compute_metrics(
            oos_equity, s.benchmark.risk_free_rate, fills=oos_fills
        )

        configs_annual = {
            key: annualize(math.prod(factors), combo_days[key])
            for key, factors in combo_factors.items()
            if factors
        }
        # OOS-итог каждой конфигурации — отдельной записью в журнал гипотез:
        # сводка журнала обязана видеть out-of-sample, а не только in-sample.
        for params in self.combos:
            key = self._key(params)
            if key in configs_annual and not math.isnan(configs_annual[key]):
                self.ledger.record(
                    hypothesis=self.strategy_factory(**params).explain_ru(),
                    source=self.source,
                    params=params,
                    data_hash=f"{self.data_hash}|oos-склейка",
                    metrics_oos={"annual_return": configs_annual[key]},
                )
        annual_values = [v for v in configs_annual.values() if not math.isnan(v)]
        best_oos = max(annual_values) if annual_values else float("nan")
        median_oos = statistics.median(annual_values) if annual_values else float("nan")

        verdict = stop_criterion_verdict_ru(
            oos_metrics.get("annual_return"), s, self.safety_margin
        )
        result = WalkForwardResult(
            windows=windows,
            chosen_params_by_window=chosen_by_window,
            oos_equity=oos_equity,
            oos_metrics=oos_metrics,
            n_configs=len(self.combos),
            configs_oos_annual=configs_annual,
            best_oos_annual=best_oos,
            median_oos_annual=median_oos,
            verdict_ru=verdict,
            limitations=limitations,
        )
        result.report_ru = self._build_report(result, window_lines)
        return result

    @staticmethod
    def _key(params: dict) -> str:
        return str(dict(sorted(params.items()))) if params else "без параметров"

    def _build_report(self, r: WalkForwardResult, window_lines: list[str]) -> str:
        s = self.settings
        m = r.oos_metrics
        best_key = (
            max(r.configs_oos_annual, key=lambda k: r.configs_oos_annual[k])
            if r.configs_oos_annual else "н/д"
        )
        n_trials = r.n_configs * len(window_lines)
        lines = [
            "WALK-FORWARD ВАЛИДАЦИЯ",
            f"Схема: обучение {self.train_years} г. → проверка {self.test_years} г., "
            f"окон: {len(window_lines)}",
            *window_lines,
            "",
            "МЕТРИКИ ПО СКЛЕЙКЕ ПРОВЕРОЧНЫХ ОКОН "
            "(только out-of-sample, in-sample не участвует):",
            f"  OOS-доходность: {m.get('annual_return', float('nan')):.1%} годовых "
            f"(итог {fmt_rub(m.get('end_equity', 0), 0)} ₽ "
            f"из {fmt_rub(s.capital.start_amount, 0)} ₽)",
            f"  Макс. просадка: {m.get('max_drawdown', float('nan')):.1%} | "
            f"Шарп (безрисковая {s.benchmark.risk_free_rate:.2%}): "
            f"{m.get('sharpe', float('nan')):.2f}",
            f"  Сделок: {m.get('n_trades', 0)} | "
            f"Издержки: {fmt_rub(m.get('total_costs', 0))} ₽",
            "",
            "МНОЖЕСТВЕННОЕ ТЕСТИРОВАНИЕ:",
            f"  Проверено конфигураций: {r.n_configs} "
            f"(испытаний с учётом окон: {n_trials})",
            f"  Лучшая по OOS-доходности: {r.best_oos_annual:.1%} годовых ({best_key})",
            f"  Медианная по OOS: {r.median_oos_annual:.1%} годовых",
            "  Поправка на множественное тестирование: результат лучшей "
            "конфигурации сам по себе не является статистически значимым — "
            "доверять можно медиане, а не максимуму. Эту защиту нельзя отключить.",
            "",
            "КРИТЕРИЙ ОСТАНОВКИ ПРОЕКТА:",
            f"  Порог безубыточности {s.breakeven_rate():.0%} + запас "
            f"{self.safety_margin:.0%} = требуется {s.breakeven_rate() + self.safety_margin:.0%}.",
            f"  {r.verdict_ru}",
        ]
        if r.limitations:
            lines += [""] + [f"ОГРАНИЧЕНИЕ: {t}" for t in r.limitations]
        return "\n".join(lines)
