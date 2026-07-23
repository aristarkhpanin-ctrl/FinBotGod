"""Тесты walk-forward валидации — критерии приёмки фазы 5.

- Метрики считаются только по out-of-sample.
- Журнал всех проверенных конфигураций сохраняется.
- В отчёте есть медиана по всем конфигурациям, а не только лучшая.
- Не более 3 оптимизируемых параметров.
- Критерий остановки проекта печатается прямым текстом.
"""

import pandas as pd
import pytest

from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import (
    ParamLimitError,
    WalkForwardRunner,
    expand_grid,
    stop_criterion_verdict_ru,
    year_windows,
)
from trading.settings import load_settings
from trading.strategy.examples.buy_and_hold import BuyAndHold


def yearly_candles(year_returns: dict[int, float], start_price: float = 100.0):
    """Синтетика: каждый год цена равномерно растёт/падает на заданный %."""
    rows = []
    price = start_price
    for year in sorted(year_returns):
        dates = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
        daily = (1 + year_returns[year]) ** (1 / len(dates))
        for d in dates:
            o = price
            price = price * daily
            rows.append((d, o, max(o, price), min(o, price), price, 1e8, 1000))
    return pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "value", "volume"]
    )


def make_runner(candles, grid, tmp_path, settings=None, **kwargs):
    return WalkForwardRunner(
        candles={"TEST": candles},
        lot_sizes={"TEST": 10},
        settings=settings or load_settings(),
        strategy_factory=lambda weight: BuyAndHold("TEST", weight),
        param_grid=grid,
        ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
        data_hash="синтетика",
        **kwargs,
    )


class TestWindows:
    def test_windows_match_tz_example(self):
        windows = year_windows(2015, 2021, train_years=4, test_years=1)
        assert len(windows) == 3
        w1 = windows[0]
        assert w1.train_start.year == 2015 and w1.train_end.year == 2018
        assert w1.test_start.year == 2019 and w1.test_end.year == 2019
        assert windows[2].test_start.year == 2021
        assert "обучение 2015–2018 → проверка 2019" == w1.label

    def test_not_enough_years(self):
        assert year_windows(2020, 2022, train_years=4, test_years=1) == []


class TestParamLimit:
    def test_four_params_rejected(self):
        with pytest.raises(ParamLimitError, match="не более 3"):
            expand_grid({"a": [1], "b": [1], "c": [1], "d": [1]})

    def test_three_params_ok(self):
        combos = expand_grid({"a": [1, 2], "b": [3], "c": [4, 5]})
        assert len(combos) == 4

    def test_empty_grid_is_one_config(self):
        assert expand_grid({}) == [{}]


class TestVerdict:
    """Четыре ветки критерия остановки (порог 38,25%, запас 10 пп)."""

    def test_loses_to_risk_free(self):
        v = stop_criterion_verdict_ru(0.10, load_settings())
        assert "ПРОИГРЫВАЕТ" in v and "ОСТАНАВЛИВАЕТСЯ" in v

    def test_below_breakeven(self):
        v = stop_criterion_verdict_ru(0.20, load_settings())
        assert "НЕ окупает" in v and "ОСТАНАВЛИВАЕТСЯ" in v

    def test_no_safety_margin(self):
        v = stop_criterion_verdict_ru(0.40, load_settings())
        assert "БЕЗ ЗАПАСА" in v and "ОСТАНАВЛИВАЕТСЯ" in v

    def test_passes_with_margin(self):
        v = stop_criterion_verdict_ru(0.60, load_settings())
        assert "НЕ сработал" in v and "ОСТАНАВЛИВАЕТСЯ" not in v


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """Рост 2015–2018, падение 2019–2021: на обучении выбирается
    «инвестировать», проверка честно показывает убыток."""
    tmp = tmp_path_factory.mktemp("wf")
    candles = yearly_candles(
        {2015: 0.30, 2016: 0.30, 2017: 0.30, 2018: 0.30,
         2019: -0.20, 2020: -0.20, 2021: -0.20}
    )
    runner = make_runner(candles, {"weight": [0.0, 0.25]}, tmp)
    r = runner.run()
    r._ledger = runner.ledger  # для теста журнала
    return r


class TestOverfitIsCaught:

    def test_train_winner_chosen_despite_bad_oos(self, result):
        # На каждом обучающем окне выигрывает weight=0.25 (рынок рос).
        assert result.chosen_params_by_window == [{"weight": 0.25}] * 3

    def test_oos_metrics_only_from_test_windows(self, result):
        assert result.oos_equity.index.min() >= pd.Timestamp(2019, 1, 1)
        assert result.oos_equity.index.max() <= pd.Timestamp(2021, 12, 31)
        # Просевший рынок → OOS-убыток, хотя in-sample был блестящим.
        assert result.oos_metrics["annual_return"] < 0

    def test_median_and_best_in_report(self, result):
        # Лучшая задним числом — «не инвестировать» (0%), медиана ниже.
        assert result.best_oos_annual == pytest.approx(0.0, abs=1e-9)
        assert result.median_oos_annual < result.best_oos_annual
        assert "Проверено конфигураций: 2" in result.report_ru
        assert "Медианная по OOS" in result.report_ru
        assert "только out-of-sample" in result.report_ru

    def test_verdict_stops_project(self, result):
        assert "ОСТАНАВЛИВАЕТСЯ" in result.verdict_ru
        assert result.verdict_ru in result.report_ru

    def test_every_config_recorded_in_ledger(self, result):
        # 3 окна × 2 конфигурации × (обучение + проверка) + 3 прогона склейки.
        assert result._ledger.count() == 3 * 2 * 2 + 3


class TestPassingStrategy:
    def test_strong_synthetic_passes_criterion(self, tmp_path):
        candles = yearly_candles({y: 0.80 for y in range(2015, 2022)})
        settings = load_settings().model_copy(deep=True)
        settings.risk.max_position_pct = 1.0
        result = make_runner(
            candles, {"weight": [1.0]}, tmp_path, settings=settings
        ).run()
        assert result.oos_metrics["annual_return"] > 0.48
        assert "НЕ сработал" in result.verdict_ru
