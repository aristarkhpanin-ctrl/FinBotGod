"""Пересборка дневных свечей в старшие таймфреймы: неделя, месяц, квартал.

Используются уже скачанные дневные данные — сеть не нужна. Свечи всех
бумаг ресемплятся ОДНИМ правилом, поэтому даты баров у них совпадают
и движок корректно их выравнивает.

Дисциплина времени сохраняется: движок принимает решение по закрытию
бара T и исполняет по открытию бара T+1 (например, решение по закрытию
недели — сделка по открытию следующей недели).
"""

from __future__ import annotations

import pandas as pd

# Правила pandas resample. 'W-FRI' — недели, заканчивающиеся пятницей
# (совпадают с торговой неделей Мосбиржи). 'ME'/'QE' — конец месяца/квартала.
RULES = {"week": "W-FRI", "month": "ME", "quarter": "QE"}

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "value": "sum",
    "volume": "sum",
}


def resample_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Дневные свечи → бары заданного таймфрейма (week | month | quarter).

    Дата бара — последний торговый день внутри периода (реальная дата,
    в которую доступно закрытие бара). Открытие бара — открытие первого
    дня периода, максимум/минимум — по всему периоду, объём — сумма.
    """
    if timeframe not in RULES:
        raise ValueError(f"Неизвестный таймфрейм {timeframe!r}, доступны: "
                         f"{', '.join(RULES)}")
    if df.empty:
        return df.copy()
    d = df.sort_values("date").set_index(pd.to_datetime(df.sort_values("date")["date"]))
    grouped = d.resample(RULES[timeframe])
    bars = grouped.agg(_AGG).dropna(subset=["close"])
    # Реальная дата закрытия бара — последний торговый день периода,
    # а не календарный конец (пятница/последнее число могли быть выходными).
    last_day = grouped["date"].last().dropna()
    bars["date"] = pd.to_datetime(last_day.values)
    bars = bars[bars["volume"].notna()]
    return bars.reset_index(drop=True)[
        ["date", "open", "high", "low", "close", "value", "volume"]
    ]
