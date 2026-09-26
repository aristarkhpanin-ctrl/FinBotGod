"""Ключевая ставка ЦБ и доходность денежного рынка.

Зачем: раньше свободные деньги в симуляции лежали под 0%, а бенчмарк
«фонд денежного рынка» считался по постоянной ставке 14,25% на всю историю.
Обе вещи искажали результат:
* в реальности свободные деньги держат в фонде денежного рынка (например,
  LQDT) и получают почти ключевую ставку — стратегия с долей кэша не должна
  за это наказываться;
* ставка за 2015–2026 менялась от 4,25% до 21%; постоянные 14,25% завышали
  планку для 2016–2021 годов и занижали для 2024–2025.

Доходность денежного рынка = ключевая ставка − спред (по умолчанию 0,5 п.п.:
комиссия фонда и отставание ставок МБК от ключевой). Спред задаётся в
settings.yaml. Одна и та же ставка используется и для дохода на кэш, и для
бенчмарка — чтобы сравнение было честным.

Источник: cbr.ru, таблица ключевой ставки (публичная, без ключа).
"""

from __future__ import annotations

import re

import httpx
import numpy as np
import pandas as pd

from trading.data.cache import ParquetCache

CBR_KEY_RATE_URL = "https://www.cbr.ru/hd_base/KeyRate/"


class RatesError(Exception):
    """Не удалось получить или разобрать историю ставки."""


def parse_cbr_key_rate_html(text: str) -> pd.DataFrame:
    """Разбор HTML-таблицы cbr.ru: строки «ДД.ММ.ГГГГ | ставка»."""
    pairs = re.findall(r"<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td>([\d,]+)</td>", text)
    if not pairs:
        raise RatesError("В ответе cbr.ru нет таблицы ключевой ставки.")
    df = pd.DataFrame(
        {"date": [pd.to_datetime(d, dayfirst=True) for d, _ in pairs],
         "rate_pct": [float(v.replace(",", ".")) for _, v in pairs]}
    )
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


class RatesData:
    """История ключевой ставки с кэшем: сеть — только при промахе кэша."""

    def __init__(self, cache: ParquetCache, timeout_seconds: float = 40.0,
                 transport: httpx.BaseTransport | None = None):
        self._cache = cache
        self._http = httpx.Client(timeout=timeout_seconds, transport=transport,
                                  headers={"User-Agent": "finbotgod-research/0.1"})

    def key_rate(self, date_from: str, date_till: str) -> pd.DataFrame:
        key = f"rates/cbr_keyrate_{date_from}_{date_till}"
        if self._cache.has(key):
            return self._cache.load(key)
        params = {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": pd.Timestamp(date_from).strftime("%d.%m.%Y"),
            "UniDbQuery.To": pd.Timestamp(date_till).strftime("%d.%m.%Y"),
        }
        resp = self._http.get(CBR_KEY_RATE_URL, params=params)
        if resp.status_code != 200:
            raise RatesError(f"cbr.ru ответил {resp.status_code}")
        df = parse_cbr_key_rate_html(resp.text)
        self._cache.save(key, df)
        return df


def money_market_rate(key_rate: pd.DataFrame, spread_pp: float = 0.5,
                      start: str | None = None, end: str | None = None) -> pd.Series:
    """Годовая доходность денежного рынка по календарным дням (доля, не %).

    Ставка на дату — последняя объявленная ЦБ (as-of назад): решение ЦБ
    известно в момент вступления в силу, заглядывания в будущее нет.
    """
    s = key_rate.set_index("date")["rate_pct"].sort_index()
    lo = pd.Timestamp(start) if start else s.index.min()
    hi = pd.Timestamp(end) if end else s.index.max()
    days = pd.date_range(min(lo, s.index.min()), hi, freq="D")
    daily = s.reindex(days).ffill().bfill()
    daily = daily[daily.index >= lo]
    return ((daily - spread_pp).clip(lower=0.0) / 100.0).rename("mm_rate")


def daily_rates(rate: pd.Series, start, end) -> pd.Series:
    """Ставка на каждый календарный день [start, end): последняя известная
    на этот день (as-of назад). Устойчиво к пропускам и к тому, что ряд
    ставки заканчивается раньше периода."""
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", inclusive="left")
    if len(days) == 0 or rate.empty:
        return pd.Series(dtype=float)
    r = rate.sort_index()
    pos = np.clip(r.index.searchsorted(days, side="right") - 1, 0, len(r) - 1)
    return pd.Series(r.to_numpy()[pos], index=days)


def money_market_growth(rate: pd.Series, start, end) -> float:
    """Во сколько раз вырастут деньги в фонде денежного рынка за [start, end)
    при ежедневной капитализации по календарным дням."""
    seg = daily_rates(rate, start, end)
    if seg.empty:
        return 1.0
    return float(np.prod(1.0 + seg.to_numpy() / 365.0))


def money_market_benchmark_series(start_capital: float, start, end,
                                  rate: pd.Series) -> dict:
    """Бенчмарк «фонд денежного рынка» по исторической ставке."""
    growth = money_market_growth(rate, start, end)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    annual = growth ** (365.25 / days) - 1 if days > 0 else float("nan")
    seg = daily_rates(rate, start, end)
    return {
        "annual_return": annual,
        "end_equity": start_capital * growth,
        "average_rate": float(seg.mean()) if not seg.empty else float("nan"),
        "source": "историческая ключевая ставка ЦБ минус спред",
    }
