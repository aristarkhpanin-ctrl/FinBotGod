"""Тесты коротких позиций: портфель, плата за заём, движок, риск-слой.

Все числа посчитаны вручную. Параметры издержек — из settings.yaml.
"""

from datetime import date

import pandas as pd
import pytest

from trading.core.costs import CostModel
from trading.core.portfolio import Portfolio, PortfolioError
from trading.engine.backtest import BacktestEngine
from trading.research.ledger import HypothesisLedger
from trading.risk.guards import RiskGuards
from trading.settings import load_settings
from trading.strategy.examples.monthly_ranked import ShortMomentum, ShortOverbought
from tests.test_backtest import flat_candles

DAY = date(2026, 7, 23)


def costs_for(order_value: float, adv: float = 1e8):
    return CostModel(load_settings().costs).trade_costs(order_value, adv)


class TestPortfolioShort:
    def test_short_forbidden_by_default(self):
        p = Portfolio(20_000)
        with pytest.raises(PortfolioError, match="Короткие продажи запрещены"):
            p.sell("SBER", 10, 100.0, costs_for(1000), DAY)

    def test_open_short_credits_cash(self):
        p = Portfolio(20_000, allow_short=True)
        c = costs_for(5_000)
        p.sell("SBER", 50, 100.0, c, DAY)
        assert p.shares_of("SBER") == -50
        assert p.cash == pytest.approx(20_000 + 5_000 - c.total)
        # Стоимость портфеля не изменилась (кэш вырос, позиция отрицательная).
        assert p.equity({"SBER": 100.0}) == pytest.approx(25_000 - c.total - 5_000)

    def test_short_profits_when_price_falls(self):
        p = Portfolio(20_000, allow_short=True)
        p.sell("SBER", 50, 100.0, costs_for(5_000), DAY)
        # Цена упала до 80: позиция −50 * 80 = −4 000, выигрыш 1 000 ₽.
        assert p.equity({"SBER": 80.0}) == pytest.approx(
            20_000 + 5_000 - costs_for(5_000).total - 4_000
        )

    def test_cover_fixes_profit(self):
        p = Portfolio(20_000, allow_short=True)
        p.sell("SBER", 50, 100.0, costs_for(5_000), DAY)
        realized = p.buy("SBER", 50, 80.0, costs_for(4_000), DAY)
        # Продали по 100, откупили по 80: (100 − 80) * 50 = 1 000 ₽.
        assert realized == pytest.approx(1_000.0)
        assert p.shares_of("SBER") == 0
        assert p.realized_pnl_by_year[DAY.year] == pytest.approx(1_000.0)

    def test_cover_fixes_loss_when_price_rises(self):
        p = Portfolio(20_000, allow_short=True)
        p.sell("SBER", 50, 100.0, costs_for(5_000), DAY)
        realized = p.buy("SBER", 50, 120.0, costs_for(6_000), DAY)
        assert realized == pytest.approx(-1_000.0)

    def test_extend_short_averages_entry(self):
        p = Portfolio(20_000, allow_short=True)
        p.sell("SBER", 10, 100.0, costs_for(1_000), DAY)
        p.sell("SBER", 10, 80.0, costs_for(800), DAY)
        pos = p.positions["SBER"]
        assert pos.shares == -20
        assert pos.avg_cost == pytest.approx(90.0)

    def test_sell_through_zero_splits_long_and_short(self):
        p = Portfolio(20_000, allow_short=True)
        p.buy("SBER", 20, 100.0, costs_for(2_000), DAY)
        realized = p.sell("SBER", 50, 110.0, costs_for(5_500), DAY)
        # Лонг 20 шт закрыт с прибылью (110−100)*20 = 200 ₽, шорт −30 шт.
        assert realized == pytest.approx(200.0)
        assert p.shares_of("SBER") == -30
        assert p.positions["SBER"].avg_cost == pytest.approx(110.0)

    def test_borrow_fee_charged(self):
        p = Portfolio(20_000, allow_short=True)
        p.charge_borrow_fee(3.57)
        assert p.cash == pytest.approx(20_000 - 3.57)
        assert p.total_borrow_fees == pytest.approx(3.57)


class TestGuardsShort:
    def test_short_open_outside_whitelist_rejected(self):
        guards = RiskGuards(load_settings().risk, whitelist={"SBER"})
        check = guards.check_order(
            DAY, "ХЛАМ", "sell", 3000, 20000,
            position_value=0, n_positions=0, is_new_position=True,
            increases_risk=True,   # открытие шорта увеличивает риск
        )
        assert not check.allowed
        assert "ИНЦИДЕНТ" in check.reason_ru

    def test_short_open_respects_position_limit(self):
        guards = RiskGuards(load_settings().risk, whitelist={"SBER"})
        check = guards.check_order(
            DAY, "SBER", "sell", 6000, 20000,
            position_value=0, n_positions=0, is_new_position=True,
            increases_risk=True,
        )
        assert check.allowed and check.max_value == pytest.approx(5_000)

    def test_cover_always_allowed(self):
        guards = RiskGuards(load_settings().risk, whitelist=set())
        check = guards.check_order(
            DAY, "ХЛАМ", "buy", 3000, 20000,
            position_value=3000, n_positions=1, is_new_position=False,
            increases_risk=False,   # закрытие шорта снижает риск
        )
        assert check.allowed


class ShortAndHold:
    """Тестовая стратегия: постоянный шорт одной бумаги."""

    name = "short_and_hold"

    def __init__(self, secid: str, weight: float = 0.25):
        self.secid, self.weight = secid, weight

    def params(self):
        return {"secid": self.secid, "weight": self.weight}

    def target_weights(self, data_until_t):
        return {self.secid: -self.weight} if self.secid in data_until_t else {}

    def explain_ru(self):
        return f"Шортить {self.secid} на {self.weight:.0%} портфеля постоянно."


def make_short_engine(candles, tmp_path, borrow_rate=None, **kwargs):
    settings = load_settings().model_copy(deep=True)
    if borrow_rate is not None:
        settings.costs.short_borrow_rate = borrow_rate
    return BacktestEngine(
        candles={"TEST": candles},
        lot_sizes={"TEST": 10},
        settings=settings,
        strategy=ShortAndHold("TEST", 0.25),
        ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
        data_hash="тест",
        allow_short=True,
        **kwargs,
    )


class TestEngineShort:
    def test_known_answer_flat_price(self, tmp_path):
        """Цена стоит на месте: шорт теряет ровно издержки входа
        (при нулевой ставке займа) — зеркало лонгового теста: 19 991,875 ₽."""
        result = make_short_engine(flat_candles(10), tmp_path, borrow_rate=0.0).run()
        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.side == "sell" and fill.shares == 50
        assert result.equity.iloc[-1] == pytest.approx(19_991.875, abs=0.005)

    def test_borrow_fee_hand_computed(self, tmp_path):
        """Ставка займа 18%: шорт 5 000 ₽ стоит 5000·0.18/252 = 3,571 ₽
        за каждый день удержания (здесь 9 дней после входа)."""
        no_fee = make_short_engine(flat_candles(10), tmp_path / "a", borrow_rate=0.0).run()
        with_fee = make_short_engine(flat_candles(10), tmp_path / "b", borrow_rate=0.18).run()
        expected_fees = 5_000 * 0.18 / 252 * 9
        assert with_fee.metrics["borrow_fees"] == pytest.approx(expected_fees, rel=1e-9)
        assert no_fee.equity.iloc[-1] - with_fee.equity.iloc[-1] == pytest.approx(
            expected_fees, rel=1e-9
        )

    def test_short_earns_on_falling_price(self, tmp_path):
        """Цена падает — шорт зарабатывает (за вычетом издержек и займа)."""
        candles = flat_candles(20)
        prices = [100.0] * 3 + [70.0] * 17   # обвал после входа
        candles["open"] = candles["high"] = candles["low"] = candles["close"] = prices
        result = make_short_engine(candles, tmp_path).run()
        assert result.equity.iloc[-1] > 20_000
        assert result.metrics["total_return"] > 0

    def test_short_forbidden_without_flag(self, tmp_path):
        """Без allow_short стратегия с шорт-весами не открывает позиций."""
        result = BacktestEngine(
            candles={"TEST": flat_candles(10)},
            lot_sizes={"TEST": 10},
            settings=load_settings(),
            strategy=ShortAndHold("TEST", 0.25),
            ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
            data_hash="тест",
            allow_short=False,
        ).run()
        assert result.fills == []
        assert any("короткие позиции выключены" in d for d in result.decisions)


class TestShortStrategies:
    def test_short_momentum_shorts_fallers_only(self):
        from tests.test_strategies import make_slice

        s = ShortMomentum(lookback=20, top_n=2)
        weights = s.target_weights(
            make_slice({"AAAA": 0.30, "BBBB": -0.10, "CCCC": -0.25})
        )
        assert set(weights) == {"BBBB", "CCCC"}
        assert all(w == pytest.approx(-0.5) for w in weights.values())

    def test_short_momentum_cash_when_nothing_falls(self):
        from tests.test_strategies import make_slice

        s = ShortMomentum(lookback=20, top_n=2)
        assert s.target_weights(make_slice({"AAAA": 0.30, "BBBB": 0.10})) == {}

    def test_short_overbought_shorts_leaders(self):
        from tests.test_strategies import make_slice

        s = ShortOverbought(lookback=20, top_n=1)
        weights = s.target_weights(
            make_slice({"AAAA": 0.30, "BBBB": -0.10, "CCCC": 0.05})
        )
        assert set(weights) == {"AAAA"}
        assert weights["AAAA"] == pytest.approx(-1.0)
