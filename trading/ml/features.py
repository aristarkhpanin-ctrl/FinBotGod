"""Признаки для мета-модели — ЧАСТЬ II.

Один и тот же расчёт используется и при генерации обучающей выборки, и в
живой стратегии-фильтре — так гарантируется, что модель видит в бою ровно
те признаки, на которых обучалась.

Все признаки считаются ТОЛЬКО по данным до момента входа t включительно:
цена, курс USD/RUB и индекс IMOEX берутся срезами по t. Утечки будущего
нет по построению.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mom_20", "mom_60", "mom_120",   # моментум бумаги на разных окнах
    "vol_20",                        # волатильность бумаги
    "dist_ma50",                     # отклонение от средней за 50 дней
    "usd_mom_60",                    # межрыночный сигнал: моментум курса
    "imoex_mom_20",                  # режим рынка: моментум индекса
    "imoex_vol_20",                  # волатильность рынка
]


def _mom(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    past = close.iloc[-1 - window]
    return close.iloc[-1] / past - 1 if past > 0 else np.nan


def _vol(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    return float(close.pct_change().iloc[-window:].std())


def bet_features(
    stock_close: pd.Series,
    usd_close: pd.Series,
    imoex_close: pd.Series,
) -> dict[str, float]:
    """Признаки одной ставки. Все ряды уже обрезаны по дате входа t."""
    ma50 = stock_close.iloc[-50:].mean() if len(stock_close) >= 50 else np.nan
    return {
        "mom_20": _mom(stock_close, 20),
        "mom_60": _mom(stock_close, 60),
        "mom_120": _mom(stock_close, 120),
        "vol_20": _vol(stock_close, 20),
        "dist_ma50": (stock_close.iloc[-1] / ma50 - 1) if ma50 and ma50 > 0 else np.nan,
        "usd_mom_60": _mom(usd_close, 60) if usd_close is not None else np.nan,
        "imoex_mom_20": _mom(imoex_close, 20) if imoex_close is not None else np.nan,
        "imoex_vol_20": _vol(imoex_close, 20) if imoex_close is not None else np.nan,
    }


def features_frame(rows: list[dict]) -> pd.DataFrame:
    """Список словарей признаков → DataFrame в каноническом порядке колонок."""
    df = pd.DataFrame(rows)
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURE_NAMES]
