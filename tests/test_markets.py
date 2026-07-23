"""Тесты рыночных профилей: MOEX совпадает с боевым конфигом, пороги
других рынков ниже."""

import pytest

from trading.markets import CRYPTO, MOEX, US_STOCKS
from trading.settings import load_settings


def test_moex_profile_matches_live_config():
    """Профиль Мосбиржи даёт тот же порог ~38%, что и settings.yaml."""
    assert MOEX.breakeven_rate() == pytest.approx(load_settings().breakeven_rate())
    assert MOEX.breakeven_rate() == pytest.approx(0.3825, abs=1e-4)


def test_moex_round_trip_matches_known_value():
    # 2*(0.05+0.01+0.05+0.05)% = 0.32% — как в разборе внутридневной торговли.
    assert MOEX.round_trip_cost_pct() == pytest.approx(0.32)


def test_other_markets_have_much_lower_breakeven():
    """Ключевой вывод: порог на США/крипте в разы ниже, чем на Мосбирже."""
    assert US_STOCKS.breakeven_rate() < 0.10
    assert CRYPTO.breakeven_rate() < 0.10
    assert MOEX.breakeven_rate() > 3 * US_STOCKS.breakeven_rate()


def test_crypto_costs_higher_than_us():
    """Крипта дешевле по инфраструктуре, но дороже за сделку."""
    assert CRYPTO.round_trip_cost_pct() > US_STOCKS.round_trip_cost_pct()


def test_only_moex_has_lots():
    assert MOEX.has_lots
    assert not US_STOCKS.has_lots      # штучные/дробные акции
    assert not CRYPTO.has_lots         # дробные объёмы
