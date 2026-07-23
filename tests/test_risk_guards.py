"""Тесты риск-слоя — фаза 6. Каждый предохранитель из таблицы ТЗ (раздел 9).

Риск-слой не зависит от логики стратегии: юнит-тесты дёргают его напрямую,
интеграционные — через движок со стратегиями-вредителями.
"""

from datetime import date

import pandas as pd
import pytest

from trading.risk.guards import RiskGuards, SystemHalted
from trading.settings import load_settings

DAY = date(2026, 7, 23)


@pytest.fixture
def guards() -> RiskGuards:
    # Лимиты из settings.yaml: позиция 25%, позиций 4, дневной убыток 3%,
    # просадка 15%, заявка 6 000 ₽, заявок в день 8.
    return RiskGuards(load_settings().risk, whitelist={"SBER", "GAZP", "LKOH"})


def buy(guards, secid="SBER", value=3000, pv=20000, pos_value=0.0,
        n_pos=0, new=True, day=DAY):
    return guards.check_order(
        day, secid, "buy", value, pv,
        position_value=pos_value, n_positions=n_pos, is_new_position=new,
    )


class TestWhitelist:
    def test_buy_outside_whitelist_is_incident(self, guards):
        check = buy(guards, secid="ХЛАМ")
        assert not check.allowed
        assert "ИНЦИДЕНТ" in check.reason_ru

    def test_sell_outside_whitelist_allowed(self, guards):
        """Выйти из позиции вне белого списка можно — застревать нельзя."""
        check = guards.check_order(
            DAY, "ХЛАМ", "sell", 3000, 20000,
            position_value=3000, n_positions=1, is_new_position=False,
        )
        assert check.allowed


class TestMaxOrderValue:
    def test_oversized_order_halts_system(self, guards):
        with pytest.raises(SystemHalted, match="расчёте единиц"):
            buy(guards, value=6001)

    def test_sell_also_checked(self, guards):
        """Баг единиц бывает и в продаже."""
        with pytest.raises(SystemHalted):
            guards.check_order(
                DAY, "SBER", "sell", 600_100, 20000,
                position_value=600_100, n_positions=1, is_new_position=False,
            )


class TestMaxOrdersPerDay:
    def test_ninth_order_halts(self, guards):
        for _ in range(8):
            assert buy(guards, value=1000).allowed
        with pytest.raises(SystemHalted, match="зацикливани"):
            buy(guards, value=1000)

    def test_counter_resets_next_day(self, guards):
        for _ in range(8):
            buy(guards, value=1000)
        next_day = date(2026, 7, 24)
        assert buy(guards, value=1000, day=next_day).allowed


class TestMaxPositions:
    def test_fifth_position_rejected(self, guards):
        check = buy(guards, n_pos=4, new=True)
        assert not check.allowed
        assert "лимит числа позиций" in check.reason_ru

    def test_adding_to_existing_position_allowed(self, guards):
        check = buy(guards, value=1000, pos_value=3000, n_pos=4, new=False)
        assert check.allowed


class TestPositionLimitTrim:
    def test_order_trimmed_to_limit(self, guards):
        # Лимит 25% от 20 000 = 5 000 ₽. Позиция уже 4 000 ₽,
        # заявка 3 000 ₽ → обрезать до 1 000 ₽.
        check = buy(guards, value=3000, pos_value=4000, n_pos=1, new=False)
        assert check.allowed
        assert check.max_value == pytest.approx(1000.0)
        assert "обрезана риск-слоем" in check.reason_ru

    def test_full_position_rejected(self, guards):
        check = buy(guards, value=1000, pos_value=5000, n_pos=1, new=False)
        assert not check.allowed


class TestDailyLossLimit:
    def test_daily_loss_triggers_liquidation_for_a_day(self, guards):
        assert guards.check_equity(DAY, 20_000) is None
        verdict = guards.check_equity(date(2026, 7, 24), 19_000)  # −5% > 3%
        assert verdict is not None and "daily_loss_limit" in verdict
        assert guards.liquidation_pending == "день"
        # После ликвидации в тот день покупки запрещены, продажи — нет.
        liq_day = date(2026, 7, 25)
        guards.liquidation_done(liq_day)
        assert not buy(guards, value=1000, day=liq_day).allowed
        assert buy(guards, value=1000, day=date(2026, 7, 26)).allowed

    def test_small_loss_does_not_trigger(self, guards):
        guards.check_equity(DAY, 20_000)
        assert guards.check_equity(date(2026, 7, 24), 19_500) is None  # −2,5%


class TestMaxDrawdown:
    def test_drawdown_halts_forever(self, guards):
        guards.check_equity(DAY, 20_000)
        verdict = guards.check_equity(date(2026, 7, 24), 16_900)  # −15,5%
        assert verdict is not None and "max_drawdown_from_peak" in verdict
        assert guards.halted_forever is not None
        assert guards.liquidation_pending == "навсегда"
        # Финальная ликвидация разрешена…
        sell = guards.check_order(
            date(2026, 7, 25), "SBER", "sell", 3000, 16_900,
            position_value=3000, n_positions=1, is_new_position=False,
        )
        assert sell.allowed
        guards.liquidation_done(date(2026, 7, 25))
        # …а после неё не проходит ничего, и покупки, и продажи.
        assert not buy(guards, value=1000, day=date(2026, 7, 28)).allowed
        assert not guards.check_order(
            date(2026, 7, 28), "SBER", "sell", 1000, 16_900,
            position_value=1000, n_positions=1, is_new_position=False,
        ).allowed

    def test_drawdown_measured_from_peak_not_start(self, guards):
        """Портфель сползает мелкими шагами (< 3% в день, дневной лимит
        молчит), но накопленная просадка от пика 30 000 ₽ добивает до 15%."""
        guards.check_equity(DAY, 20_000)
        guards.check_equity(date(2026, 7, 24), 30_000)      # новый пик
        slide = [29_500, 28_900, 28_300, 27_700, 27_100, 26_500, 25_900]
        for i, equity in enumerate(slide):
            verdict = guards.check_equity(date(2026, 8, 1 + i), equity)
            assert verdict is None, f"рано сработал на {equity}: {verdict}"
        # 25 400 — это −15,3% от пика 30 000, хотя лишь −1,9% за день.
        verdict = guards.check_equity(date(2026, 8, 20), 25_400)
        assert verdict is not None and "max_drawdown_from_peak" in verdict


# ---------- Интеграция с движком ----------

from trading.engine.backtest import BacktestEngine
from trading.research.ledger import HypothesisLedger
from trading.settings import load_settings as _ls
from trading.strategy.examples.buy_and_hold import BuyAndHold
from tests.test_backtest import flat_candles


def run_engine(candles, strategy, tmp_path, settings=None, **kwargs):
    return BacktestEngine(
        candles=candles,
        lot_sizes={s: 10 for s in candles},
        settings=settings or _ls(),
        strategy=strategy,
        ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
        data_hash="тест",
        **kwargs,
    ).run()


def crash_candles(n_flat: int = 5, n_crash: int = 3, drop: float = 0.30):
    """Плоская цена, затем обвал по −30% в день."""
    prices, p = [], 100.0
    for _ in range(n_flat):
        prices.append(p)
    for _ in range(n_crash):
        p *= 1 - drop
        prices.append(p)
    dates = pd.bdate_range("2024-01-01", periods=len(prices))
    close = pd.Series(prices)
    open_ = close.shift(1).fillna(100.0)
    return pd.DataFrame(
        {
            "date": dates, "open": open_,
            "high": pd.concat([open_, close], axis=1).max(axis=1),
            "low": pd.concat([open_, close], axis=1).min(axis=1),
            "close": close, "value": 1e8, "volume": 1000,
        }
    )


class TestEngineIntegration:
    def test_non_whitelisted_buy_rejected_as_incident(self, tmp_path):
        result = run_engine(
            {"TEST": flat_candles()}, BuyAndHold("TEST", 0.25), tmp_path,
            whitelist=["SBER"],   # TEST вне белого списка
        )
        assert result.fills == []
        assert any("ИНЦИДЕНТ" in d for d in result.decisions)

    def test_drawdown_liquidates_and_halts(self, tmp_path):
        settings = _ls().model_copy(deep=True)
        settings.risk.max_position_pct = 1.0
        settings.risk.max_order_value = 50_000
        result = run_engine(
            {"TEST": crash_candles()}, BuyAndHold("TEST", 1.0), tmp_path,
            settings=settings,
        )
        # Предохранитель просадки сработал и записан.
        assert any("max_drawdown_from_peak" in e for e in result.guard_events)
        assert result.halted_reason is not None
        assert any("ЛИКВИДАЦИЯ (навсегда)" in d for d in result.decisions)
        # После ликвидации позиций нет: последняя сделка — продажа.
        assert result.fills[-1].side == "sell"
        assert "СИСТЕМА ОСТАНОВЛЕНА ПРЕДОХРАНИТЕЛЕМ" in result.report_ru
        assert "ПРЕДОХРАНИТЕЛИ РИСК-СЛОЯ:" in result.report_ru

    def test_unit_bug_halts_system(self, tmp_path):
        """Потолок суммы заявки ловит баг в расчёте единиц."""
        settings = _ls().model_copy(deep=True)
        settings.risk.max_order_value = 100   # заведомо меньше любой покупки
        result = run_engine(
            {"TEST": flat_candles()}, BuyAndHold("TEST", 0.25), tmp_path,
            settings=settings,
        )
        assert result.halted_reason is not None
        assert "расчёте единиц" in result.halted_reason
        assert result.fills == []   # ни одна сделка не прошла

    def test_no_guards_fired_on_calm_run(self, tmp_path):
        result = run_engine(
            {"TEST": flat_candles()}, BuyAndHold("TEST", 0.25), tmp_path
        )
        assert result.guard_events == []
        assert result.halted_reason is None
        assert "не срабатывали" in result.report_ru
