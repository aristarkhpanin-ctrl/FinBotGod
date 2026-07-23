"""Клиент MOEX ISS API: загрузка с пагинацией, ретраями и кэшированием.

Правила из ТЗ (раздел 4):
* пагинация через блок ``cursor`` и параметр ``start`` — не полагаться
  на то, что первая страница содержит всё;
* не более 3 запросов в секунду, таймаут 10 секунд, экспоненциальные ретраи;
* скачанное однажды складывается в Parquet и больше не запрашивается:
  повторный прогон не делает ни одного сетевого запроса;
* обрыв сети посреди загрузки не портит кэш (атомарная запись в cache.py:
  данные пишутся в кэш только после полного скачивания диапазона).
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

from trading.data.cache import ParquetCache
from trading.logging_setup import get_logger

log = get_logger("moex")

BASE_URL = "https://iss.moex.com/iss"
BOARD_PATH = "/engines/stock/markets/shares/boards/TQBR"


class MoexApiError(Exception):
    """Сетевая ошибка или некорректный ответ ISS после всех ретраев."""


class MoexClient:
    """Тонкий HTTP-клиент к ISS. Не знает про кэш — только сеть."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        requests_per_second: float = 3.0,
        timeout_seconds: float = 10.0,
        retries: int = 5,
        retry_delay_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._min_interval = 1.0 / requests_per_second
        self._retries = retries
        self._retry_delay = retry_delay_seconds
        self._last_request_at = 0.0
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": "finbotgod-research/0.1"},
        )

    def close(self) -> None:
        self._http.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _get_json(self, path: str, params: dict) -> dict:
        """GET с троттлингом и экспоненциальными ретраями."""
        params = {"iss.meta": "off", **params}
        delay = self._retry_delay
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            if attempt > 0:
                log.warning(
                    "повтор запроса к ISS",
                    попытка=attempt, задержка_сек=delay, путь=path,
                    ошибка=str(last_error),
                )
                time.sleep(delay)
                delay *= 2
            self._throttle()
            try:
                self._last_request_at = time.monotonic()
                resp = self._http.get(path, params=params)
                if resp.status_code >= 500:
                    last_error = MoexApiError(f"ISS ответил {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_error = e
        raise MoexApiError(
            f"Запрос {path} не удался после {self._retries + 1} попыток: {last_error}"
        )

    @staticmethod
    def _table(payload: dict, name: str) -> pd.DataFrame:
        block = payload.get(name)
        if not block or "columns" not in block:
            raise MoexApiError(f"В ответе ISS нет таблицы {name!r}")
        return pd.DataFrame(block["data"], columns=block["columns"])

    def _paginated(self, path: str, params: dict, table: str) -> pd.DataFrame:
        """Собирает все страницы таблицы.

        Если ISS отдаёт блок ``{table}.cursor`` (INDEX, TOTAL, PAGESIZE) —
        листаем по нему. Иначе (например, свечи) — увеличиваем ``start``,
        пока страницы не закончатся.
        """
        pages: list[pd.DataFrame] = []
        start = 0
        while True:
            payload = self._get_json(path, {**params, "start": start})
            page = self._table(payload, table)
            if page.empty:
                break
            pages.append(page)
            cursor_block = payload.get(f"{table}.cursor")
            if cursor_block and cursor_block.get("data"):
                cursor = dict(zip(cursor_block["columns"], cursor_block["data"][0]))
                start = int(cursor["INDEX"]) + int(cursor["PAGESIZE"])
                if start >= int(cursor["TOTAL"]):
                    break
            else:
                start += len(page)
        if not pages:
            return pd.DataFrame()
        return pd.concat(pages, ignore_index=True)

    # ---------- Публичные методы ----------

    def securities(self) -> pd.DataFrame:
        """Список бумаг основного режима TQBR. Содержит LOTSIZE."""
        payload = self._get_json(
            f"{BOARD_PATH}/securities.json",
            {"securities.columns": "SECID,SHORTNAME,LOTSIZE,ISIN,PREVPRICE"},
        )
        df = self._table(payload, "securities")
        if "LOTSIZE" not in df.columns:
            raise MoexApiError("В ответе ISS нет поля LOTSIZE — проверить эндпоинт")
        return df

    def daily_candles(self, secid: str, date_from: str, date_till: str) -> pd.DataFrame:
        """Дневные свечи (interval=24) за диапазон дат, все страницы."""
        df = self._paginated(
            f"{BOARD_PATH}/securities/{secid}/candles.json",
            {"from": date_from, "till": date_till, "interval": 24},
            "candles",
        )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["begin"]).dt.normalize()
        df = df[["date", "open", "high", "low", "close", "value", "volume"]]
        return df.sort_values("date").reset_index(drop=True)

    def trade_history(self, secid: str, date_from: str, date_till: str) -> pd.DataFrame:
        """Историческая статистика торгов (объёмы, число сделок) — ликвидность."""
        return self._paginated(
            f"/history{BOARD_PATH}/securities/{secid}.json",
            {"from": date_from, "till": date_till},
            "history",
        )

    def index_composition(self, date: str, index_id: str = "IMOEX") -> pd.DataFrame:
        """Состав индекса на дату — защита от ошибки выживаемости."""
        return self._paginated(
            f"/statistics/engines/stock/markets/index/analytics/{index_id}.json",
            {"date": date, "limit": 100},
            "analytics",
        )


class MarketData:
    """Кэширующая обёртка: сначала Parquet, сеть — только при промахе."""

    def __init__(self, client: MoexClient, cache: ParquetCache):
        self._client = client
        self._cache = cache

    def daily_candles(self, secid: str, date_from: str, date_till: str) -> pd.DataFrame:
        key = f"candles/{secid}_{date_from}_{date_till}"
        if self._cache.has(key):
            return self._cache.load(key)
        df = self._client.daily_candles(secid, date_from, date_till)
        # Пустой результат тоже кэшируем: бумага могла не торговаться
        # в диапазоне, и переспрашивать сеть каждый раз незачем.
        self._cache.save(key, df)
        return df

    def securities(self) -> pd.DataFrame:
        key = "securities/tqbr"
        if self._cache.has(key):
            return self._cache.load(key)
        df = self._client.securities()
        self._cache.save(key, df)
        return df

    def index_composition(self, date: str, index_id: str = "IMOEX") -> pd.DataFrame:
        key = f"index/{index_id}_{date}"
        if self._cache.has(key):
            return self._cache.load(key)
        df = self._client.index_composition(date, index_id)
        self._cache.save(key, df)
        return df

    def lot_sizes(self) -> dict[str, int]:
        """SECID → размер лота. Критично для сайзинга (фаза 3)."""
        df = self.securities()
        return {
            str(row.SECID): int(row.LOTSIZE)
            for row in df.itertuples()
            if pd.notna(row.LOTSIZE)
        }
