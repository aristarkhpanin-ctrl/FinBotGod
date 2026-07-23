"""Метрики доходности — ФАЗА 4.

Ключевое требование ТЗ (раздел 7): коэффициент Шарпа считается
с безрисковой ставкой 14,25%, а не с нулём. Шарп с нулевой безрисковой
ставкой на российском рынке 2026 года — бессмысленное число.
"""

from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualize(total_return_factor: float, days: float) -> float:
    """Годовая доходность из фактора роста и календарной длительности."""
    if days <= 0 or total_return_factor <= 0:
        return float("nan")
    years = days / 365.25
    return total_return_factor ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Максимальная просадка (отрицательное число) и её длительность в днях.

    Длительность — самый долгий календарный период от пика до возврата
    на пик (или до конца данных, если пик так и не восстановлен).
    """
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = float(dd.min()) if len(dd) else 0.0

    longest = 0
    peak_date = None      # дата пика, с которого началась текущая просадка
    prev_day = None
    for day, below in (equity < peak).items():
        if below and peak_date is None:
            peak_date = prev_day if prev_day is not None else day
        elif not below and peak_date is not None:
            longest = max(longest, (day - peak_date).days)
            peak_date = None
        prev_day = day
    if peak_date is not None and len(equity):
        longest = max(longest, (equity.index[-1] - peak_date).days)
    return max_dd, longest


def compute_metrics(
    equity: pd.Series,
    risk_free_rate: float,
    fills: list | None = None,
) -> dict:
    """Метрики по кривой стоимости портфеля (индекс — даты, значения — ₽)."""
    if len(equity) < 2:
        return {"error": "слишком мало данных для метрик"}

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    days = (equity.index[-1] - equity.index[0]).days
    cagr = annualize(end / start, days)

    returns = equity.pct_change().dropna()
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - rf_daily

    std = float(returns.std(ddof=1))
    sharpe = (
        float(excess.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if std > 0 else float("nan")
    )
    downside = excess[excess < 0]
    downside_std = (
        math.sqrt(float((downside ** 2).sum()) / len(excess)) if len(excess) else 0.0
    )
    sortino = (
        float(excess.mean()) / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_std > 0 else float("nan")
    )

    dd, dd_days = max_drawdown(equity)
    calmar = cagr / abs(dd) if dd < 0 else float("nan")

    metrics = {
        "start_equity": start,
        "end_equity": end,
        "total_return": end / start - 1,
        "annual_return": cagr,
        "max_drawdown": dd,
        "max_drawdown_days": dd_days,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "risk_free_rate": risk_free_rate,
    }

    if fills is not None:
        total_costs = sum(f.costs.total for f in fills)
        turnover = sum(f.order_value for f in fills)
        gross_profit = (end - start) + total_costs  # прибыль ДО издержек
        closers = [f for f in fills if f.realized_pnl is not None]
        wins = [f for f in closers if f.realized_pnl > 0]
        gains = sum(f.realized_pnl for f in closers if f.realized_pnl > 0)
        losses = -sum(f.realized_pnl for f in closers if f.realized_pnl < 0)
        metrics.update(
            {
                "n_trades": len(fills),
                "turnover": turnover,
                "total_costs": total_costs,
                "costs_pct_of_gross": (
                    total_costs / gross_profit if gross_profit > 0 else float("nan")
                ),
                "win_rate": len(wins) / len(closers) if closers else float("nan"),
                "profit_factor": (
                    gains / losses if losses > 0
                    else (float("inf") if gains > 0 else float("nan"))
                ),
            }
        )
    return metrics


def money_market_benchmark(start_capital: float, days: float, rate: float) -> dict:
    """Фонд денежного рынка: капитал растёт под безрисковую ставку."""
    end = start_capital * (1 + rate) ** (days / 365.25)
    return {"annual_return": rate, "end_equity": end}


def buy_and_hold_benchmark(prices: pd.Series) -> dict:
    """Купил и держи (например, индекс IMOEX) — без издержек, идеализированно."""
    if len(prices) < 2:
        return {"annual_return": float("nan"), "error": "нет данных"}
    days = (prices.index[-1] - prices.index[0]).days
    factor = float(prices.iloc[-1]) / float(prices.iloc[0])
    return {"annual_return": annualize(factor, days), "total_return": factor - 1}
