"""Тесты метрик: числа с известным ответом, Шарп с безрисковой 14,25%."""

import pandas as pd
import pytest

from trading.reporting.metrics import (
    annualize,
    buy_and_hold_benchmark,
    compute_metrics,
    max_drawdown,
    money_market_benchmark,
)


def test_annualize_one_year_exact():
    # Рост в 1,1425 раза ровно за год = 14,25% годовых.
    assert annualize(1.1425, 365.25) == pytest.approx(0.1425)


def test_annualize_two_years():
    # Рост в 1,21 раза за два года = 10% годовых.
    assert annualize(1.21, 730.5) == pytest.approx(0.10)


def test_money_market_benchmark():
    bm = money_market_benchmark(20_000, 365.25, 0.1425)
    assert bm["annual_return"] == 0.1425
    assert bm["end_equity"] == pytest.approx(22_850.0)


def test_buy_and_hold_benchmark():
    prices = pd.Series(
        [100.0, 114.25],
        index=pd.to_datetime(["2025-01-01", "2026-01-01"]),
    )
    bm = buy_and_hold_benchmark(prices)
    assert bm["annual_return"] == pytest.approx(0.1425, rel=0.01)


def test_max_drawdown_known_case():
    # Пик 110 в день 2, дно 99: просадка 99/110 − 1 = −10%.
    # Восстановление на пик в день 5 → длительность день 2 → день 5 = 3 дня.
    equity = pd.Series(
        [100, 110, 99, 105, 110, 120],
        index=pd.date_range("2024-01-01", periods=6),
        dtype=float,
    )
    dd, days = max_drawdown(equity)
    assert dd == pytest.approx(99 / 110 - 1)
    assert days == 3


def test_max_drawdown_never_recovered():
    equity = pd.Series(
        [100, 120, 90, 91],
        index=pd.date_range("2024-01-01", periods=4),
        dtype=float,
    )
    dd, days = max_drawdown(equity)
    assert dd == pytest.approx(90 / 120 - 1)
    assert days == 2  # со дня падения до конца данных


def test_sharpe_uses_nonzero_risk_free_rate():
    """Портфель, растущий медленнее безрисковой ставки, имеет Шарп < 0.

    С нулевой ставкой Шарп был бы положительным — вот почему ноль
    на российском рынке 2026 года — бессмысленное число.
    """
    dates = pd.date_range("2024-01-01", periods=253)
    slow_growth = pd.Series(
        [100 * (1.0002 + (0.0001 if i % 2 else -0.0001)) ** i for i in range(253)],
        index=dates,
    )
    with_rf = compute_metrics(slow_growth, risk_free_rate=0.1425)
    with_zero = compute_metrics(slow_growth, risk_free_rate=0.0)
    assert with_rf["sharpe"] < 0 < with_zero["sharpe"]


def test_metrics_structure():
    equity = pd.Series(
        [100.0, 101.0, 102.5, 101.5, 103.0],
        index=pd.date_range("2024-01-01", periods=5),
    )
    m = compute_metrics(equity, risk_free_rate=0.1425)
    for key in ("annual_return", "max_drawdown", "sharpe", "sortino", "calmar"):
        assert key in m
    assert m["total_return"] == pytest.approx(0.03)
