"""Тесты движка бэктеста — критерии приёмки фазы 4.

1. Тест известного ответа: «купить и держать» на одной бумаге даёт
   результат, совпадающий с ручным расчётом до копейки.
2. Тест на заглядывание в будущее: сдвиг сигнала на день вперёд
   заметно ухудшает доходность.
3. Отчёт содержит все три бенчмарка.
4. Прогон с удвоенными издержками — одним флагом.
5. Каждый прогон автоматически пишется в журнал гипотез.
"""

import pandas as pd
import pytest

from trading.engine.backtest import BacktestEngine
from trading.research.ledger import HypothesisLedger
from trading.settings import load_settings
from trading.strategy.examples.buy_and_hold import BuyAndHold


def flat_candles(n: int = 10, price: float = 100.0, value: float = 1e7) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates, "open": price, "high": price, "low": price,
            "close": price, "value": value, "volume": 1000,
        }
    )


def momentum_candles(cycles: int = 20, run: int = 6) -> pd.DataFrame:
    """Синтетика с идеальным дневным моментумом: run дней +2%, run дней −2%."""
    closes, c = [], 100.0
    for _ in range(cycles):
        for _ in range(run):
            c *= 1.02
            closes.append(c)
        for _ in range(run):
            c *= 0.98
            closes.append(c)
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.Series(closes)
    open_ = close.shift(1).fillna(100.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": pd.concat([open_, close], axis=1).max(axis=1),
            "low": pd.concat([open_, close], axis=1).min(axis=1),
            "close": close,
            "value": 1e8,
            "volume": 1000,
        }
    )


class Momentum1D:
    """Тестовая стратегия: вчера рост → купить на всё, иначе в кэш."""

    name = "momentum_1d_test"

    def params(self) -> dict:
        return {"lookback": 1}

    def target_weights(self, data_until_t):
        df = data_until_t.get("TEST")
        if df is None or len(df) < 2:
            return {}
        grew = df["close"].iloc[-1] > df["close"].iloc[-2]
        return {"TEST": 1.0} if grew else {}

    def explain_ru(self) -> str:
        return "Если вчера цена выросла — купить на всё, иначе выйти в кэш."


def make_engine(candles, strategy, tmp_path, settings=None, **kwargs):
    return BacktestEngine(
        candles={"TEST": candles},
        lot_sizes={"TEST": 10},
        settings=settings or load_settings(),
        strategy=strategy,
        ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
        data_hash="тест",
        **kwargs,
    )


class TestKnownAnswer:
    """Ручной расчёт (цена всегда 100 ₽, лот 10 шт, капитал 20 000 ₽, доля 25%):

    Покупка на открытии дня 2: 5 лотов = 50 шт = 5 000 ₽.
    Издержки: брокер 2,50 + биржа 0,50 + полспреда 2,50
      + проскальзывание 5 000 * (0,05% + 5,0 * 5000/10 000 000) = 2,625
      = 8,125 ₽.
    Кэш: 20 000 − 5 000 − 8,125 = 14 991,875 ₽.
    Стоимость портфеля со дня 2 и до конца: 14 991,875 + 5 000 = 19 991,875 ₽.
    """

    def test_final_equity_to_the_kopeck(self, tmp_path):
        result = make_engine(flat_candles(), BuyAndHold("TEST", 0.25), tmp_path).run()
        assert result.equity.iloc[0] == pytest.approx(20_000.0)
        assert result.equity.iloc[-1] == pytest.approx(19_991.875, abs=0.005)

    def test_exactly_one_buy_no_oscillation(self, tmp_path):
        """Округление лотов не должно рождать бесконечные продажи/покупки."""
        result = make_engine(flat_candles(), BuyAndHold("TEST", 0.25), tmp_path).run()
        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.side == "buy" and fill.shares == 50
        assert fill.costs.total == pytest.approx(8.125)
        assert any("ПОКУПКА" in d and "5 лот" in d for d in result.decisions)

    def test_costs_metrics(self, tmp_path):
        result = make_engine(flat_candles(), BuyAndHold("TEST", 0.25), tmp_path).run()
        assert result.metrics["total_costs"] == pytest.approx(8.125)
        assert result.metrics["turnover"] == pytest.approx(5_000.0)


class TestLookAheadGuard:
    """Обязательный тест ТЗ: сдвиг сигнала должен заметно ухудшить результат."""

    def _run(self, tmp_path, shift: int) -> float:
        settings = load_settings().model_copy(deep=True)
        settings.risk.max_position_pct = 1.0  # моментум на всё — виднее эффект
        settings.risk.max_order_value = 10_000_000  # иначе риск-слой остановит
        result = make_engine(
            momentum_candles(), Momentum1D(), tmp_path,
            settings=settings, signal_shift_days=shift,
        ).run()
        assert result.metrics["n_trades"] > 10  # стратегия реально торговала
        return result.metrics["total_return"]

    def test_shifted_signal_is_notably_worse(self, tmp_path):
        normal = self._run(tmp_path / "normal", 0)
        shifted = self._run(tmp_path / "shifted", 1)
        assert normal > 0  # идеальный моментум на синтетике прибылен
        # Сдвиг на день съедает больше трети преимущества — утечки будущего нет.
        assert (1 + shifted) < (1 + normal) * 0.7


class TestBenchmarksInReport:
    def test_three_benchmarks_always_present(self, tmp_path):
        report = make_engine(flat_candles(), BuyAndHold("TEST", 0.25), tmp_path).run().report_ru
        assert "Стратегия:" in report
        assert "Купил и держи IMOEX" in report
        assert "Фонд денежного рынка" in report

    def test_losing_to_money_market_shouted_first(self, tmp_path):
        """Плоская бумага проигрывает 14,25% — вердикт до всех цифр."""
        report = make_engine(flat_candles(), BuyAndHold("TEST", 0.25), tmp_path).run().report_ru
        assert "ПРОИГРЫВАЕТ ФОНДУ ДЕНЕЖНОГО РЫНКА" in report
        assert report.index("ПРОИГРЫВАЕТ") < report.index("МЕТРИКИ")

    def test_imoex_absence_is_explicit(self, tmp_path):
        result = make_engine(flat_candles(), BuyAndHold("TEST", 0.25), tmp_path).run()
        assert result.benchmarks["IMOEX"] is None
        assert any("IMOEX" in t for t in result.limitations)

    def test_resolution_report_before_metrics(self, tmp_path):
        report = make_engine(flat_candles(), BuyAndHold("TEST", 0.25), tmp_path).run().report_ru
        assert "РАЗРЕШАЮЩАЯ СПОСОБНОСТЬ КАПИТАЛА" in report
        assert report.index("РАЗРЕШАЮЩАЯ") < report.index("МЕТРИКИ")


class TestStressMode:
    def test_double_costs_with_single_flag(self, tmp_path):
        settings = load_settings().model_copy(deep=True)
        settings.costs.stress_multiplier = 2.0
        result = make_engine(
            flat_candles(), BuyAndHold("TEST", 0.25), tmp_path, settings=settings
        ).run()
        # Издержки ровно вдвое: 16,25 ₽ вместо 8,125 ₽.
        assert result.metrics["total_costs"] == pytest.approx(16.25)
        assert result.equity.iloc[-1] == pytest.approx(19_983.75, abs=0.005)
        assert "СТРЕСС-ПРОГОН" in result.report_ru


class TestLedgerIntegration:
    def test_run_recorded_automatically(self, tmp_path):
        ledger = HypothesisLedger(tmp_path / "ledger.sqlite")
        engine = BacktestEngine(
            candles={"TEST": flat_candles()},
            lot_sizes={"TEST": 10},
            settings=load_settings(),
            strategy=BuyAndHold("TEST", 0.25),
            ledger=ledger,
            data_hash="тест",
        )
        result = engine.run()
        assert ledger.count() == 1
        record = ledger.get(result.run_hash)
        assert record is not None
        assert record.params["signal_shift_days"] == 0
        assert record.metrics_is["n_trades"] == 1

    def test_engine_without_ledger_refused(self):
        with pytest.raises(ValueError, match="журнала гипотез"):
            BacktestEngine(
                candles={"TEST": flat_candles()},
                lot_sizes={"TEST": 10},
                settings=load_settings(),
                strategy=BuyAndHold("TEST", 0.25),
                ledger=None,
                data_hash="тест",
            )


class TestTradeFrom:
    def test_no_trades_before_trade_from(self, tmp_path):
        """Разогрев: сигналы по всей истории, сделки — только с даты старта."""
        candles = flat_candles(10)
        start = candles["date"].iloc[6]
        result = BacktestEngine(
            candles={"TEST": candles},
            lot_sizes={"TEST": 10},
            settings=load_settings(),
            strategy=BuyAndHold("TEST", 0.25),
            ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
            data_hash="тест",
            trade_from=start.isoformat(),
        ).run()
        assert all(pd.Timestamp(f.day) >= start for f in result.fills)
        assert len(result.fills) == 1
        # Метрики считаются с даты старта, разогрев не разбавляет доходность.
        assert result.equity.index[0] == start
        assert len(result.equity) == 4


class TestTimeDiscipline:
    def test_strategy_never_sees_future(self, tmp_path):
        """Архитектурная проверка: срез данных стратегии кончается днём T."""
        seen_max_dates = []

        class Spy(BuyAndHold):
            def target_weights(self, data_until_t):
                if "TEST" in data_until_t:
                    seen_max_dates.append(data_until_t["TEST"]["date"].max())
                return super().target_weights(data_until_t)

        candles = flat_candles(6)
        make_engine(candles, Spy("TEST", 0.25), tmp_path).run()
        dates = list(candles["date"])
        # Решение, исполненное в день i, видело данные максимум до дня i-1.
        assert seen_max_dates == dates[:-1]
