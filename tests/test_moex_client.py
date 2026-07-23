"""Тесты клиента MOEX ISS: пагинация, кэширование, устойчивость к обрывам.

Сеть подменяется httpx.MockTransport — тесты работают без интернета.
"""

import httpx
import pandas as pd
import pytest

from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexApiError, MoexClient

CANDLE_COLUMNS = ["begin", "open", "close", "high", "low", "value", "volume"]


def candle_row(day: int):
    price = 100.0 + day
    return [
        f"2024-01-{day:02d} 00:00:00",
        price, price + 1, price + 2, price - 1, 1_000_000.0, 5000,
    ]


class FakeIss:
    """Подставной сервер ISS: 7 свечей, страницы по 3 строки."""

    def __init__(self):
        self.requests = 0
        self.all_candles = [candle_row(d) for d in range(1, 8)]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        params = dict(request.url.params)
        path = request.url.path
        if "candles.json" in path:
            start = int(params.get("start", 0))
            page = self.all_candles[start : start + 3]
            return httpx.Response(
                200, json={"candles": {"columns": CANDLE_COLUMNS, "data": page}}
            )
        if path.endswith("/securities.json") and "/history/" not in path:
            return httpx.Response(
                200,
                json={
                    "securities": {
                        "columns": ["SECID", "SHORTNAME", "LOTSIZE", "ISIN", "PREVPRICE"],
                        "data": [
                            ["SBER", "Сбербанк", 10, "RU0009029540", 312.4],
                            ["LKOH", "Лукойл", 1, "RU0009024277", 7100.0],
                        ],
                    }
                },
            )
        if "/history/" in path:
            # Пагинация через блок cursor: всего 5 строк, страницы по 2.
            start = int(params.get("start", 0))
            rows = [[f"2024-01-{d:02d}", "SBER", 100 + d] for d in range(1, 6)]
            return httpx.Response(
                200,
                json={
                    "history": {
                        "columns": ["TRADEDATE", "SECID", "NUMTRADES"],
                        "data": rows[start : start + 2],
                    },
                    "history.cursor": {
                        "columns": ["INDEX", "TOTAL", "PAGESIZE"],
                        "data": [[start, 5, 2]],
                    },
                },
            )
        return httpx.Response(404)


@pytest.fixture
def fake_iss() -> FakeIss:
    return FakeIss()


def make_client(fake: FakeIss) -> MoexClient:
    return MoexClient(
        requests_per_second=1000,  # в тестах не ждём
        retries=1,
        retry_delay_seconds=0.01,
        transport=httpx.MockTransport(fake.handler),
    )


def test_candles_pagination_collects_all_pages(fake_iss):
    client = make_client(fake_iss)
    df = client.daily_candles("SBER", "2024-01-01", "2024-01-07")
    assert len(df) == 7  # 3 + 3 + 1 — все страницы собраны
    assert list(df.columns) == ["date", "open", "high", "low", "close", "value", "volume"]
    assert df["date"].is_monotonic_increasing


def test_cursor_pagination(fake_iss):
    client = make_client(fake_iss)
    df = client.trade_history("SBER", "2024-01-01", "2024-01-05")
    assert len(df) == 5  # 2 + 2 + 1 по блоку cursor


def test_securities_contains_lotsize(fake_iss):
    client = make_client(fake_iss)
    df = client.securities()
    assert "LOTSIZE" in df.columns


def test_repeat_run_makes_zero_network_requests(fake_iss, tmp_path):
    """Критерий приёмки фазы 1: повторный запуск не трогает сеть."""
    cache = ParquetCache(tmp_path / "cache")
    market = MarketData(make_client(fake_iss), cache)

    market.daily_candles("SBER", "2024-01-01", "2024-01-07")
    market.securities()
    requests_after_first_run = fake_iss.requests
    assert requests_after_first_run > 0

    df = market.daily_candles("SBER", "2024-01-01", "2024-01-07")
    lots = market.lot_sizes()
    assert fake_iss.requests == requests_after_first_run  # ни одного нового запроса
    assert len(df) == 7
    assert lots == {"SBER": 10, "LKOH": 1}


def test_network_failure_mid_download_does_not_corrupt_cache(tmp_path):
    """Критерий приёмки фазы 1: обрыв сети посреди загрузки не портит кэш."""
    fake = FakeIss()
    cache = ParquetCache(tmp_path / "cache")

    # Сначала успешно скачиваем SBER.
    market = MarketData(make_client(fake), cache)
    good = market.daily_candles("SBER", "2024-01-01", "2024-01-07")
    hash_before = cache.data_version_hash()

    # Теперь сеть обрывается на второй странице свечей GAZP.
    calls = {"n": 0}

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise httpx.ConnectError("обрыв сети")
        return fake.handler(request)

    flaky_client = MoexClient(
        requests_per_second=1000, retries=0, retry_delay_seconds=0.01,
        transport=httpx.MockTransport(flaky_handler),
    )
    flaky_market = MarketData(flaky_client, cache)
    with pytest.raises(MoexApiError):
        flaky_market.daily_candles("GAZP", "2024-01-01", "2024-01-07")

    # Частичных данных GAZP в кэше нет, данные SBER не изменились.
    assert not cache.has("candles/GAZP_2024-01-01_2024-01-07")
    assert cache.data_version_hash() == hash_before
    pd.testing.assert_frame_equal(
        cache.load("candles/SBER_2024-01-01_2024-01-07"), good
    )


def test_retries_then_success():
    """Первый запрос падает, ретрай добивается ответа."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("временный сбой")
        return httpx.Response(
            200, json={"candles": {"columns": CANDLE_COLUMNS, "data": []}}
        )

    client = MoexClient(
        requests_per_second=1000, retries=2, retry_delay_seconds=0.01,
        transport=httpx.MockTransport(handler),
    )
    df = client.daily_candles("SBER", "2024-01-01", "2024-01-07")
    assert df.empty
    assert attempts["n"] == 2


def test_server_error_5xx_retried_then_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = MoexClient(
        requests_per_second=1000, retries=1, retry_delay_seconds=0.01,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(MoexApiError, match="не удался"):
        client.daily_candles("SBER", "2024-01-01", "2024-01-07")
