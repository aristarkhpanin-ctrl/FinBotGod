"""Тесты модели издержек — критерии приёмки фазы 2.

Каждый расчёт проверен вручную, калькулятором, и записан в комментарии.
Параметры по умолчанию из settings.yaml:
  комиссия брокера 0,05% (мин. 1 ₽), биржа 0,01%, полспреда 0,05%,
  базовое проскальзывание 0,05%, коэффициент влияния 5,0,
  предупреждение при 1% ADV, отказ при 5% ADV.
"""

import pytest

from trading.core.costs import CostModel, UnfillableOrderError
from trading.settings import load_settings


@pytest.fixture
def model() -> CostModel:
    return CostModel(load_settings().costs)


def stressed_model(multiplier: float) -> CostModel:
    cfg = load_settings().costs.model_copy(update={"stress_multiplier": multiplier})
    return CostModel(cfg)


class TestKnownAnswer:
    """Сделка с известными параметрами даёт вручную посчитанный результат."""

    def test_hand_computed_trade(self, model):
        # Сделка 10 000 ₽ при дневном обороте 10 000 000 ₽ (0,1% ADV):
        #   брокер:        10 000 * 0,05% = 5,00 ₽ (больше минимума 1 ₽)
        #   биржа:         10 000 * 0,01% = 1,00 ₽
        #   полспреда:     10 000 * 0,05% = 5,00 ₽
        #   проскальзывание: 0,05% + 5,0 * (10 000/10 000 000) = 0,055%
        #                  10 000 * 0,055% = 5,50 ₽
        #   ИТОГО: 16,50 ₽
        costs = model.trade_costs(order_value=10_000, adv_20=10_000_000)
        assert costs.broker_commission == pytest.approx(5.00)
        assert costs.exchange_fee == pytest.approx(1.00)
        assert costs.half_spread == pytest.approx(5.00)
        assert costs.slippage_pct == pytest.approx(0.055)
        assert costs.slippage == pytest.approx(5.50)
        assert costs.total == pytest.approx(16.50)

    def test_explain_ru_is_human_checkable(self, model):
        costs = model.trade_costs(order_value=10_000, adv_20=10_000_000)
        text = costs.explain_ru()
        assert "16.50" in text
        assert "комиссия брокера" in text
        assert "проскальзывание" in text


class TestMinimalCommission:
    """Минимальная комиссия применяется на маленьких сделках."""

    def test_min_commission_kicks_in(self, model):
        # Сделка 500 ₽: 0,05% = 0,25 ₽ < минимума 1 ₽ → берётся 1 ₽.
        costs = model.trade_costs(order_value=500, adv_20=10_000_000)
        assert costs.broker_commission == pytest.approx(1.00)

    def test_min_commission_not_applied_to_large(self, model):
        # Сделка 100 000 ₽: 0,05% = 50 ₽ — минимум не при чём.
        costs = model.trade_costs(order_value=100_000, adv_20=100_000_000)
        assert costs.broker_commission == pytest.approx(50.00)


class TestSlippageGrowsWithSize:
    """Проскальзывание растёт при росте размера заявки."""

    def test_monotonic_slippage(self, model):
        adv = 10_000_000
        small = model.trade_costs(order_value=10_000, adv_20=adv)
        big = model.trade_costs(order_value=100_000, adv_20=adv)
        assert big.slippage_pct > small.slippage_pct
        assert big.slippage > small.slippage

    def test_slippage_formula(self, model):
        # 100 000 ₽ при ADV 10 000 000 ₽ (1% ADV):
        # 0,05% + 5,0 * 0,01 = 0,10%; 100 000 * 0,10% = 100 ₽.
        assert model.slippage_pct(100_000, 10_000_000) == pytest.approx(0.10)


class TestLiquidityLimits:
    """Заявка объёмом 10% от ADV отклоняется (критерий приёмки)."""

    def test_10_percent_adv_rejected(self, model):
        with pytest.raises(UnfillableOrderError, match="отклонена"):
            model.trade_costs(order_value=1_000_000, adv_20=10_000_000)

    def test_just_below_5_percent_accepted(self, model):
        costs = model.trade_costs(order_value=490_000, adv_20=10_000_000)
        assert costs.total > 0

    def test_unknown_adv_rejected(self, model):
        with pytest.raises(UnfillableOrderError):
            model.trade_costs(order_value=10_000, adv_20=0)

    def test_nonpositive_order_rejected(self, model):
        with pytest.raises(ValueError):
            model.trade_costs(order_value=0, adv_20=10_000_000)


class TestStressMode:
    """Стресс-режим: издержки, увеличенные в 2 раза, одним параметром."""

    def test_stress_doubles_all_costs(self):
        normal = stressed_model(1.0).trade_costs(10_000, 10_000_000)
        stressed = stressed_model(2.0).trade_costs(10_000, 10_000_000)
        assert stressed.broker_commission == pytest.approx(2 * normal.broker_commission)
        assert stressed.exchange_fee == pytest.approx(2 * normal.exchange_fee)
        assert stressed.half_spread == pytest.approx(2 * normal.half_spread)
        assert stressed.slippage == pytest.approx(2 * normal.slippage)
        assert stressed.total == pytest.approx(2 * normal.total)  # 33,00 ₽


class TestNdfl:
    """НДФЛ 13% с зафиксированной прибыли, в годовом разрезе."""

    def test_ndfl_by_year(self, model):
        taxes = model.ndfl({2024: 10_000.0, 2025: -5_000.0, 2026: 0.0})
        assert taxes[2024] == pytest.approx(1_300.0)
        assert taxes[2025] == 0.0  # убыточный год налога не создаёт
        assert taxes[2026] == 0.0
