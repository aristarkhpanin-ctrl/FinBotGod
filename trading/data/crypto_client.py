"""Клиент Bitfinex для дневных свечей криптовалют.

Тот же принцип, что у moex_client: свечи → кэш Parquet → валидация,
повторный прогон не трогает сеть. Публичный API, без ключа.

Формат свечи Bitfinex: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME] — обратите
внимание, CLOSE идёт вторым, а не последним. Один запрос с limit=10000
и sort=1 отдаёт всю дневную историю пары по возрастанию.

Крипта дробная: на $250 нельзя купить целый BTC. Поэтому размер «лота»
берётся дробным (crypto_lot_size) так, чтобы один лот стоил порядка
$0,05–0,5 — гранулярность становится незначимой, а движок не меняется.
"""

from __future__ import annotations

import math
import time

import httpx
import pandas as pd

from trading.data.cache import ParquetCache
from trading.logging_setup import get_logger

log = get_logger("crypto")

BASE_URL = "https://api-pub.bitfinex.com"


class CryptoApiError(Exception):
    """Сетевая ошибка или некорректный ответ Bitfinex после ретраев."""


class BitfinexClient:
    def __init__(self, requests_per_second: float = 2.0, timeout_seconds: float = 15.0,
                 retries: int = 5, retry_delay_seconds: float = 1.0,
                 transport: httpx.BaseTransport | None = None):
        self._min_interval = 1.0 / requests_per_second
        self._retries = retries
        self._retry_delay = retry_delay_seconds
        self._last_request_at = 0.0
        self._http = httpx.Client(base_url=BASE_URL, timeout=timeout_seconds,
                                  transport=transport,
                                  headers={"User-Agent": "finbotgod-research/0.1"})

    def close(self) -> None:
        self._http.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def daily_candles(self, pair: str) -> pd.DataFrame:
        """Все дневные свечи пары (например, 'tBTCUSD'), по возрастанию дат."""
        path = f"/v2/candles/trade:1D:{pair}/hist"
        params = {"limit": 10000, "sort": 1}
        delay = self._retry_delay
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            if attempt > 0:
                log.warning("повтор запроса к Bitfinex", пара=pair, попытка=attempt,
                            задержка=delay, ошибка=str(last_error))
                time.sleep(delay)
                delay *= 2
            self._throttle()
            try:
                self._last_request_at = time.monotonic()
                resp = self._http.get(path, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = CryptoApiError(f"Bitfinex ответил {resp.status_code}")
                    continue
                resp.raise_for_status()
                rows = resp.json()
                if not isinstance(rows, list):
                    raise CryptoApiError(f"Неожиданный ответ Bitfinex: {str(rows)[:120]}")
                return self._to_frame(rows)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_error = e
        raise CryptoApiError(
            f"Запрос свечей {pair} не удался после {self._retries + 1} попыток: {last_error}"
        )

    @staticmethod
    def _to_frame(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["mts", "open", "close", "high", "low", "volume"])
        df["date"] = pd.to_datetime(df["mts"], unit="ms").dt.normalize()
        df["value"] = df["volume"] * df["close"]     # оборот в долларах
        df = df[["date", "open", "high", "low", "close", "value", "volume"]]
        return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


class CryptoData:
    """Кэширующая обёртка: сначала Parquet, сеть — только при промахе."""

    def __init__(self, client: BitfinexClient, cache: ParquetCache):
        self._client = client
        self._cache = cache

    def daily_candles(self, pair: str) -> pd.DataFrame:
        key = f"crypto/{pair}"
        if self._cache.has(key):
            return self._cache.load(key)
        df = self._client.daily_candles(pair)
        self._cache.save(key, df)
        return df


def crypto_lot_size(price: float, target_cost: float = 0.1) -> float:
    """Дробный размер лота: степень 10, при которой лот стоит ~target_cost $.

    Так на любой цене (BTC $60 000 или XRP $0,5) один лот стоит порядка
    десяти центов, и округление по лотам не мешает диверсификации $250.
    """
    if price <= 0:
        return 1.0
    exp = math.floor(math.log10(target_cost / price))
    return float(10.0 ** exp)
