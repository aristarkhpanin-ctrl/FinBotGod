"""Симулятор распределения капитала между фондами и правила распределения.

Время исполнения (консервативно): решение принимается по закрытию дня t
(по данным до t включительно), заявка исполняется по закрытию дня t+1.
Новая доля начинает зарабатывать с доходности дня t+2. Никакого
«решил и тут же купил по той же цене».

Издержки — доля от оборота на каждой стороне (продажа одного фонда и
покупка другого платятся обе). Фонды, в отличие от отдельных акций,
покупаются практически любой суммой (цена пая — единицы рублей), поэтому
лотность здесь не ограничивает.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.allocation.assets import ASSETS, BONDS, CASH, EQUITY


@dataclass
class SimResult:
    equity: pd.Series            # стоимость портфеля на закрытие каждого дня
    weights: pd.DataFrame        # фактические доли (после исполнения) на закрытие
    turnover_rub: float          # суммарный оборот, ₽
    costs_rub: float             # суммарные издержки, ₽
    n_rebalances: int            # сколько раз меняли распределение
    equity_share_avg: float      # средняя доля акций


def simulate(returns: pd.DataFrame, targets: pd.DataFrame, cost_rate: float,
             start_capital: float = 20_000.0, drift_band: float = 0.05) -> SimResult:
    """``targets`` — целевые доли, РЕШЁННЫЕ по закрытию дня (индекс = день
    решения). Исполнение — на закрытии следующего дня. Ребалансировка — когда
    цель изменилась или фактическая доля ушла от цели дальше ``drift_band``.
    До первого исполнения капитал лежит в фонде денежного рынка."""
    cols = list(ASSETS)
    r = returns[cols].to_numpy()
    tg = targets.reindex(returns.index)[cols].to_numpy()
    n = len(returns)
    values = np.zeros(3)
    values[cols.index(CASH)] = start_capital
    eq_out, w_out = np.empty(n), np.empty((n, 3))
    turnover = costs = 0.0
    rebalances = 0
    last_target = None
    for i in range(n):
        values = values * (1.0 + r[i])                  # доходность дня i
        if i >= 1 and not np.isnan(tg[i - 1]).any():     # решение дня i−1 → сделка на закрытии i
            target = tg[i - 1]
            total = values.sum()
            current = values / total if total > 0 else values
            changed = last_target is None or not np.allclose(target, last_target)
            drifted = np.abs(current - target).max() > drift_band
            if changed or drifted:
                trade = np.abs(target * total - values).sum()
                cost = trade * cost_rate
                values = target * (total - cost)
                turnover += trade
                costs += cost
                rebalances += 1
                last_target = target
        eq_out[i] = values.sum()
        w_out[i] = values / eq_out[i] if eq_out[i] > 0 else values
    weights = pd.DataFrame(w_out, index=returns.index, columns=cols)
    return SimResult(
        equity=pd.Series(eq_out, index=returns.index, name="equity"),
        weights=weights, turnover_rub=turnover, costs_rub=costs,
        n_rebalances=rebalances, equity_share_avg=float(weights[EQUITY].mean()),
    )


# ---------- Правила распределения (цели по дню решения) ----------

def _weights(index, eq: pd.Series, rest: pd.Series | str) -> pd.DataFrame:
    """Доля акций ``eq``, остаток — в ``rest`` (CASH/BONDS или ряд с именами)."""
    out = pd.DataFrame(0.0, index=index, columns=list(ASSETS))
    out[EQUITY] = eq.reindex(index).to_numpy()
    rest_names = (pd.Series(rest, index=index) if isinstance(rest, str)
                  else rest.reindex(index))
    for name in (BONDS, CASH):
        out[name] = np.where(rest_names.to_numpy() == name, 1.0 - out[EQUITY], 0.0)
    return out


def static_targets(index: pd.DatetimeIndex, equity: float = 0.0, bonds: float = 0.0
                   ) -> pd.DataFrame:
    """Неизменное распределение (бенчмарки: 100% денег, 100% акций, 60/40 …)."""
    out = pd.DataFrame(0.0, index=index, columns=list(ASSETS))
    out[EQUITY], out[BONDS] = equity, bonds
    out[CASH] = 1.0 - equity - bonds
    return out


def weekly(signal: pd.Series) -> pd.Series:
    """Решение пересматривается раз в неделю — в последний торговый день
    недели; между пересмотрами держится прошлое решение. Так систему можно
    вести вручную раз в неделю — без сервера."""
    s = signal.copy()
    week = s.index.to_period("W")
    last_day_of_week = pd.Series(s.index, index=s.index).groupby(week).transform("max")
    keep = s.index == last_day_of_week.to_numpy()
    return s.where(keep).ffill()


def easing_regime(key_rate: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """True, если последнее известное на дату изменение ключевой ставки —
    снижение (цикл смягчения: облигации дорожают). Только прошлые решения ЦБ."""
    s = key_rate.set_index("date")["rate_pct"].sort_index()
    change = s.diff()
    direction = change[change != 0].dropna().apply(np.sign)
    pos = direction.index.searchsorted(index, side="right") - 1
    vals = np.where(pos >= 0, direction.to_numpy()[np.clip(pos, 0, None)], 0.0)
    return pd.Series(vals < 0, index=index)


def rest_asset(mode: str, easing: pd.Series | None, index) -> pd.Series | str:
    """Куда идёт не-акционная часть: 'деньги' — всегда фонд денежного рынка;
    'офз_при_снижении' — в ОФЗ, пока ЦБ снижает ставку, иначе в деньги."""
    if mode == CASH:
        return CASH
    if mode == "офз_при_снижении":
        return pd.Series(np.where(easing.reindex(index).fillna(False), BONDS, CASH), index=index)
    raise ValueError(f"неизвестный режим остатка: {mode}")


def ml_risk_targets(proba: pd.Series, base_rate: pd.Series, k: float, w_max: float,
                    rest: pd.Series | str) -> pd.DataFrame:
    """ML-правило: акции (доля ``w_max``), пока модель оценивает риск просадки
    не выше обычного больше чем в ``k`` раз; иначе — всё в остаток."""
    risky = proba > k * base_rate.reindex(proba.index)
    eq = weekly(pd.Series(np.where(risky, 0.0, w_max), index=proba.index))
    return _weights(proba.index, eq, rest)


def trend_targets(equity_index: pd.Series, index, rest) -> pd.DataFrame:
    """Правило без ML: акции, пока индекс выше 200-дневной средней."""
    ma = equity_index.rolling(200).mean()
    eq = weekly((equity_index > ma).astype(float).where(ma.notna()))
    return _weights(index, eq, rest)


def vol_targets(vol20: pd.Series, index, rest) -> pd.DataFrame:
    """Правило без ML: акции, пока волатильность месяца ниже своей медианы за год."""
    med = vol20.rolling(250).median()
    eq = weekly((vol20 < med).astype(float).where(med.notna()))
    return _weights(index, eq, rest)
