"""Дневная доходность трёх «фондов» из индексов полной доходности.

* Акции   — MCFTRR (дивиденды реинвестируются после налога) минус комиссия
            фонда на индекс (TER);
* ОФЗ     — RGBITR (купоны реинвестируются) минус комиссия фонда облигаций;
* Деньги  — фонд денежного рынка: ключевая ставка ЦБ минус спред.

Доходность дня s — от закрытия s−1 до закрытия s. Комиссия фонда и доход
денежного рынка начисляются по календарным дням между закрытиями — так же,
как в движке бэктеста (выходные тоже приносят проценты).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EQUITY, BONDS, CASH = "акции", "офз", "деньги"
ASSETS = (EQUITY, BONDS, CASH)


def fund_returns(
    equity_index: pd.Series,
    bond_index: pd.Series,
    cash_rate: pd.Series,
    equity_ter: float = 0.008,
    bond_ter: float = 0.006,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Таблица дневных доходностей (индекс — торговые дни акций).

    ``cash_rate`` — годовая ставка фонда денежного рынка по календарным дням
    (доля, as-of). До начала ряда ставки доходность денег не определена —
    такие дни отбрасываются (честнее, чем выдумывать ставку).
    """
    eq = equity_index.sort_index()
    calendar = eq.index
    if start is not None:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    if end is not None:
        calendar = calendar[calendar <= pd.Timestamp(end)]
    bonds = bond_index.sort_index().reindex(eq.index.union(bond_index.index)).ffill()
    days = pd.Series(calendar, index=calendar).diff().dt.days
    out = pd.DataFrame(index=calendar)
    out[EQUITY] = eq.reindex(calendar).pct_change() - equity_ter * days / 365.0
    out[BONDS] = bonds.reindex(calendar).pct_change() - bond_ter * days / 365.0
    if cash_rate is not None and len(cash_rate):
        # Ставка, известная на закрытие предыдущего дня (as-of), — как в движке.
        r = cash_rate.sort_index()
        prev = calendar[:-1]
        pos = r.index.searchsorted(prev, side="right") - 1
        rate_prev = np.where(pos >= 0, r.to_numpy()[np.clip(pos, 0, None)], np.nan)
        cash = np.full(len(calendar), np.nan)
        cash[1:] = rate_prev * days.to_numpy()[1:] / 365.0
        out[CASH] = cash
    else:
        out[CASH] = float("nan")
    out["дней"] = days
    return out.iloc[1:].dropna()
