"""История дивидендов по акциям Мосбиржи и экс-дивидендные даты.

Зачем: ISS отдаёт цены без дивидендов. Раньше стратегии, державшие акции,
теряли дивидендный доход (на уровне индекса это ~7-8% годовых: IMOEX
+0,6% против MCFTR +8,3% за 2016-2026). Теперь держатель бумаги на закрытии
дня перед экс-датой получает дивиденд за вычетом НДФЛ.

Источники (публичные HTML-таблицы, ISS историю дивидендов больше не отдаёт):
* dohod.ru — полная история выплат по текущим бумагам (дата закрытия
  реестра + сумма на акцию);
* smart-lab.ru — запасной источник, в том числе по части ушедших с биржи
  бумаг; содержит и «последний день с правом» (дата T-1).
Строки с пометкой «прогноз» отбрасываются — это не факт, а ожидание.

Экс-дата (первый день без права на дивиденд, в этот день цена «гэпает»):
* до 31.07.2023 расчёты шли в режиме Т+2: последний день с правом —
  за 2 торговых дня до закрытия реестра, экс-дата — за 1 день;
* с 31.07.2023 — режим Т+1: последний день с правом — за 1 торговый день,
  экс-дата — день закрытия реестра.
Если источник указал последний день с правом явно — берётся он.
"""

from __future__ import annotations

import html as html_lib
import re
import time

import httpx
import numpy as np
import pandas as pd

from trading.data.cache import ParquetCache
from trading.logging_setup import get_logger

log = get_logger("dividends")

T_PLUS_1_FROM = pd.Timestamp("2023-07-31")   # переход Мосбиржи на режим Т+1

# Переименованные бумаги: выплаты старого тикера лежат на странице нового.
ALIASES = {"TCSG": "T", "YNDX": "YDEX", "FIVE": "X5", "MAIL": "VKCO", "LNTA": "LENT"}

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124 Safari/537.36"}
_COLUMNS = ["record_date", "amount", "last_day_with_rights", "source"]


def _cells(table_html: str) -> list[list[str]]:
    rows = re.findall(r"<tr.*?</tr>", table_html, re.S)
    return [
        [html_lib.unescape(re.sub(r"<.*?>", "", c)).strip()
         for c in re.findall(r"<t[dh].*?</t[dh]>", r, re.S)]
        for r in rows
    ]


def _amount(text: str) -> float | None:
    cleaned = re.sub(r"[^\d,.\-]", "", text.replace("\xa0", "")).replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def _date(text: str) -> pd.Timestamp | None:
    m = re.search(r"\d{2}\.\d{2}\.\d{4}", text)
    return pd.to_datetime(m.group(0), dayfirst=True) if m else None


def parse_dohod(page: str) -> pd.DataFrame:
    """Таблица dohod.ru: «Дата объявления | Дата закрытия реестра | Год | Дивиденд»."""
    out = []
    for table in re.findall(r"<table.*?</table>", page, re.S):
        cells = _cells(table)
        if not cells or "Дата закрытия реестра" not in " ".join(cells[0]):
            continue
        header = cells[0]
        i_rec = next(i for i, h in enumerate(header) if "закрытия реестра" in h)
        i_amt = next(i for i, h in enumerate(header) if h.startswith("Дивиденд"))
        for row in cells[1:]:
            if len(row) <= max(i_rec, i_amt) or "прогноз" in " ".join(row).lower():
                continue
            rec, amt = _date(row[i_rec]), _amount(row[i_amt])
            if rec is not None and amt is not None:
                out.append((rec, amt, pd.NaT, "dohod.ru"))
    return pd.DataFrame(out, columns=_COLUMNS)


def parse_smartlab(page: str) -> pd.DataFrame:
    """Таблица smart-lab: «Тикер | дата T-1 | дата отсечки | Период | дивиденд | …»."""
    out = []
    for table in re.findall(r"<table.*?</table>", page, re.S):
        cells = _cells(table)
        header_idx = next((k for k, r in enumerate(cells) if "дата отсечки" in " ".join(r)), None)
        if header_idx is None:
            continue
        header = cells[header_idx]
        i_t1 = next((i for i, h in enumerate(header) if "T-1" in h), None)
        i_rec = next(i for i, h in enumerate(header) if "отсечки" in h)
        i_amt = next(i for i, h in enumerate(header) if h.lower().startswith("дивиденд"))
        for row in cells[header_idx + 1:]:
            if len(row) <= max(i_rec, i_amt) or "прогноз" in " ".join(row).lower():
                continue
            rec, amt = _date(row[i_rec]), _amount(row[i_amt])
            t1 = _date(row[i_t1]) if i_t1 is not None and i_t1 < len(row) else None
            if rec is not None and amt is not None:
                out.append((rec, amt, t1 if t1 is not None else pd.NaT, "smart-lab.ru"))
    return pd.DataFrame(out, columns=_COLUMNS)


class DividendsData:
    """Дивиденды с кэшем. Пустой результат тоже кэшируется — сеть не
    дёргается повторно. Между запросами пауза (вежливость к сайтам)."""

    def __init__(self, cache: ParquetCache, pause_seconds: float = 1.0,
                 transport: httpx.BaseTransport | None = None):
        self._cache = cache
        self._pause = pause_seconds
        self._http = httpx.Client(timeout=30, headers=_UA, follow_redirects=True,
                                  transport=transport)

    def _get(self, url: str) -> str | None:
        time.sleep(self._pause)
        try:
            resp = self._http.get(url)
        except httpx.HTTPError as e:
            log.warning("дивиденды: сеть недоступна", адрес=url, ошибка=str(e))
            return None
        return resp.text if resp.status_code == 200 else None

    def history(self, ticker: str) -> pd.DataFrame:
        key = f"dividends/{ticker}"
        if self._cache.has(key):
            return self._cache.load(key)
        frames = []
        for page_ticker in [ticker] + ([ALIASES[ticker]] if ticker in ALIASES else []):
            page = self._get(f"https://www.dohod.ru/ik/analytics/dividend/{page_ticker.lower()}")
            if page:
                frames.append(parse_dohod(page))
            if not frames or frames[-1].empty:
                page = self._get(f"https://smart-lab.ru/q/{page_ticker}/dividend/")
                if page:
                    frames.append(parse_smartlab(page))
            if frames and not frames[-1].empty:
                break
        df = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=_COLUMNS))
        df = (df.drop_duplicates(["record_date", "amount"])
                .sort_values("record_date").reset_index(drop=True))
        self._cache.save(key, df)
        return df


def ex_dividend_dates(divs: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Экс-дивидендные даты по торговому календарю (см. правила в шапке модуля)."""
    if divs is None or divs.empty:
        return pd.DataFrame(columns=["ex_date", "amount", "record_date"])
    cal = pd.DatetimeIndex(sorted(calendar))
    rows = []
    for rec, amt, last_right in zip(divs["record_date"], divs["amount"],
                                    divs["last_day_with_rights"]):
        rec = pd.Timestamp(rec)
        if pd.notna(last_right):
            pos = cal.searchsorted(pd.Timestamp(last_right), side="right")
        else:
            lag = 1 if rec >= T_PLUS_1_FROM else 2
            last_trading = cal.searchsorted(rec, side="right") - 1
            pos = last_trading - lag + 1
        if 0 < pos < len(cal):
            rows.append((cal[pos], float(amt), rec))
    out = pd.DataFrame(rows, columns=["ex_date", "amount", "record_date"])
    return out.drop_duplicates(["ex_date", "amount"]).reset_index(drop=True)


def check_dividend_yields(ex: pd.DataFrame, candles: pd.DataFrame,
                          low: float = 0.0005, high: float = 0.35
                          ) -> tuple[pd.DataFrame, list[str]]:
    """Отсев неправдоподобных выплат по доходности к цене накануне экс-даты.

    ISS отдаёт цены С поправкой на сплиты (GMKN 1:100, VTBR 5000:1), а сайты
    с дивидендами — часто БЕЗ неё. Тогда «дивиденд» выходит в сотни процентов
    от цены (или в тысячные доли процента). Такие выплаты отбрасываются —
    это занижает доход, но не выдумывает его. Список отброшенного — в отчёт.
    """
    if ex is None or ex.empty or candles is None or candles.empty:
        return ex, []
    c = candles.sort_values("date")
    dates = pd.to_datetime(c["date"]).values
    closes = c["close"].values
    keep, rejected = [], []
    for row in ex.itertuples():
        pos = int(np.searchsorted(dates, pd.Timestamp(row.ex_date).to_datetime64())) - 1
        if pos < 0:
            keep.append(False)
            continue
        ratio = float(row.amount) / float(closes[pos])
        ok = low <= ratio <= high
        keep.append(ok)
        if not ok:
            rejected.append(f"{pd.Timestamp(row.ex_date).date()} {row.amount:g} ₽ "
                            f"при цене {closes[pos]:g} ₽ ({ratio:.2%})")
    return ex[np.array(keep, dtype=bool)].reset_index(drop=True), rejected
