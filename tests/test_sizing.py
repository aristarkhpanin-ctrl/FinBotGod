"""Тесты сайзинга — обязательные проверки фазы 3 (ТЗ, раздел 6).

Все числа посчитаны вручную и записаны в комментариях.
Параметры издержек — из settings.yaml (комиссия 0,05% мин. 1 ₽,
биржа 0,01%, полспреда 0,05%, проскальзывание 0,05% + 5,0 * доля ADV).
"""

import pytest

from trading.core.costs import CostModel
from trading.core.sizing import resolution_report_ru, size_position
from trading.settings import load_settings

ADV = 100_000_000  # ликвидная бумага: заявки в тестах < 1% ADV


@pytest.fixture
def cost_model() -> CostModel:
    return CostModel(load_settings().costs)


class TestBasicFormula:
    def test_lots_are_floored(self, cost_model):
        # Портфель 20 000 ₽, доля 25% → цель 5 000 ₽.
        # Лот SBER: 312,40 * 10 = 3 124 ₽. floor(5000/3124) = 1 лот = 10 шт.
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=0.25,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert not d.skipped
        assert d.lots == 1
        assert d.shares == 10
        assert d.order_value == pytest.approx(3_124.0)
        assert "1 лот" in d.reason_ru

    def test_multiple_lots(self, cost_model):
        # Цель 10 000 ₽, лот 3 124 ₽ → floor = 3 лота = 30 шт = 9 372 ₽.
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=0.50,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.50, cost_model=cost_model, adv_20=ADV,
        )
        assert d.lots == 3
        assert d.order_value == pytest.approx(9_372.0)


class TestCheckOne_LotTooExpensive:
    """Проверка 1: lots == 0 → пропуск с причиной из ТЗ."""

    def test_skip_with_russian_reason(self, cost_model):
        # Пример из ТЗ: целевая сумма 4 200 ₽, лот 4 850 ₽ → пропуск.
        d = size_position(
            "LKOH", price=4_850.0, lot_size=1, target_weight=0.21,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert d.skipped
        assert d.lots == 0 and d.shares == 0 and d.order_value == 0
        assert "Пропуск LKOH" in d.reason_ru
        assert "4 200 ₽" in d.reason_ru
        assert "4 850 ₽" in d.reason_ru


class TestCheckTwo_PositionLimit:
    """Проверка 2: превышение max_position_pct → обрезание, пересчёт вниз."""

    def test_target_trimmed_to_limit(self, cost_model):
        # Доля 60% от 20 000 ₽ = 12 000 ₽, но лимит 25% = 5 000 ₽.
        # Лот 3 124 ₽ → floor(5000/3124) = 1 лот, а не 3.
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=0.60,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert d.lots == 1
        assert d.order_value == pytest.approx(3_124.0)
        assert "обрезана лимитом 25%" in d.reason_ru

    def test_position_never_exceeds_limit(self, cost_model):
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=1.0,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert d.order_value <= 20_000 * 0.25


class TestCheckThree_CashIncludingCosts:
    """Проверка 3: достаточность денег с учётом комиссии, не только цены."""

    def test_lots_reduced_when_costs_dont_fit(self, cost_model):
        # Цель 2 лота = 6 248 ₽, издержки ≈ 10,32 ₽. Свободно ровно 6 250 ₽:
        # 6 248 + 10,32 = 6 258,32 > 6 250 → влезает только 1 лот.
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=0.50,
            portfolio_value=12_500, cash_available=6_250,
            max_position_pct=0.50, cost_model=cost_model, adv_20=ADV,
        )
        assert d.lots == 1

    def test_cash_only_for_price_without_costs_is_not_enough(self, cost_model):
        # Свободно ровно 3 124 ₽ — цена лота без издержек. Пропуск.
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=0.25,
            portfolio_value=12_496, cash_available=3_124,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert d.skipped
        assert "с издержками" in d.reason_ru

    def test_order_plus_costs_fit_in_cash(self, cost_model):
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=0.25,
            portfolio_value=20_000, cash_available=3_200,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert not d.skipped
        assert d.order_value + d.costs.total <= 3_200


class TestLiquidity:
    def test_illiquid_order_becomes_skip_not_crash(self, cost_model):
        # Заявка 9 372 ₽ при ADV 100 000 ₽ — 9,4% > порога 5% → пропуск.
        d = size_position(
            "НЕЛИКВИД", price=312.40, lot_size=10, target_weight=0.50,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.50, cost_model=cost_model, adv_20=100_000,
        )
        assert d.skipped
        assert "отклонена" in d.reason_ru


class TestBadInputs:
    def test_zero_price_skipped(self, cost_model):
        d = size_position(
            "SBER", price=0.0, lot_size=10, target_weight=0.25,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert d.skipped
        assert "битые данные" in d.reason_ru

    def test_zero_weight_skipped(self, cost_model):
        d = size_position(
            "SBER", price=312.40, lot_size=10, target_weight=0.0,
            portfolio_value=20_000, cash_available=20_000,
            max_position_pct=0.25, cost_model=cost_model, adv_20=ADV,
        )
        assert d.skipped


class TestResolutionReport:
    """Отчёт о разрешающей способности — заказчик видит его до кривой доходности."""

    def test_report_matches_tz_example(self):
        # Средняя стоимость лота 4 600 ₽ при капитале 20 000 ₽:
        # максимум позиций по деньгам floor(20000/4600) = 4,
        # минимальный шаг 4600/20000 = 23%.
        report = resolution_report_ru(
            capital=20_000,
            lot_costs={"AAAA": 4_000, "BBBB": 4_600, "CCCC": 5_200},
            max_positions=4,
            max_position_pct=0.25,
        )
        assert "Капитал: 20 000 ₽" in report
        assert "Средняя стоимость лота в универсуме: 4 600 ₽" in report
        assert "Максимум одновременных позиций: 4" in report
        assert "Минимальный шаг позиции: 23% портфеля" in report
        assert "ВНИМАНИЕ" in report
        assert "2% на сделку нереализуем" in report

    def test_expensive_lots_listed(self):
        # Лот дороже лимита 25% * 20 000 = 5 000 ₽ — бумага недоступна.
        report = resolution_report_ru(
            capital=20_000,
            lot_costs={"SBER": 3_124, "LKOH": 7_100},
            max_positions=4,
            max_position_pct=0.25,
        )
        assert "LKOH" in report
        assert "Недоступны при лимите 25%" in report

    def test_no_warning_for_large_capital(self):
        # Капитал 10 млн ₽: шаг позиции 0,05% — правило 2% реализуемо.
        report = resolution_report_ru(
            capital=10_000_000,
            lot_costs={"SBER": 3_124, "LKOH": 7_100},
            max_positions=10,
            max_position_pct=0.25,
        )
        assert "ВНИМАНИЕ" not in report

    def test_empty_universe(self):
        assert "пуст" in resolution_report_ru(20_000, {}, 4, 0.25)
