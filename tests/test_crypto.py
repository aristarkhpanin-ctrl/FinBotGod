"""Тесты крипто-клиента, дробного лота и дробных позиций портфеля."""

import httpx
import pandas as pd
import pytest

from trading.core.costs import CostModel
from trading.core.portfolio import Portfolio
from trading.data.crypto_client import BitfinexClient, crypto_lot_size
from trading.settings import load_settings


class TestBitfinexParsing:
    def test_candle_column_order(self):
        # Bitfinex: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME] — close ВТОРОЙ.
        rows = [
            [1364688000000, 92.5, 93.033, 93.75, 91.0, 3083.0],
            [1364774400000, 93.0, 95.0, 96.0, 92.0, 1000.0],
        ]

        def handler(request):
            return httpx.Response(200, json=rows)

        client = BitfinexClient(requests_per_second=1000, retry_delay_seconds=0.01,
                                transport=httpx.MockTransport(handler))
        df = client.daily_candles("tBTCUSD")
        assert list(df.columns) == ["date", "open", "high", "low", "close", "value", "volume"]
        first = df.iloc[0]
        assert first["open"] == 92.5
        assert first["close"] == 93.033      # не 93.75 (это high)
        assert first["high"] == 93.75
        assert first["low"] == 91.0
        assert first["value"] == pytest.approx(3083.0 * 93.033)   # оборот в $

    def test_retries_on_429(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": "rate"})
            return httpx.Response(200, json=[[1364688000000, 1, 2, 3, 0.5, 10]])

        client = BitfinexClient(requests_per_second=1000, retries=2,
                                retry_delay_seconds=0.01,
                                transport=httpx.MockTransport(handler))
        df = client.daily_candles("tBTCUSD")
        assert len(df) == 1 and calls["n"] == 2


class TestCryptoLotSize:
    @pytest.mark.parametrize("price", [60000.0, 3000.0, 0.5, 0.08])
    def test_lot_cost_is_small_fraction(self, price):
        lot = crypto_lot_size(price, target_cost=0.1)
        lot_cost = lot * price
        # Один лот стоит порядка десяти центов на любой цене.
        assert 0.01 <= lot_cost <= 1.0

    def test_expensive_and_cheap_coins_both_fine_grained(self):
        # На $250 позиция 25% = $62 делится на сотни лотов даже дорогой монеты.
        for price in (60000.0, 0.5):
            lot = crypto_lot_size(price)
            lots_in_position = 62.5 / (lot * price)
            assert lots_in_position > 50


class TestFractionalPortfolio:
    def costs(self, value):
        return CostModel(load_settings().costs).trade_costs(value, adv_20=1e9)

    def test_fractional_buy_and_full_sell(self):
        from datetime import date

        p = Portfolio(250.0)
        # Купить 0.001 BTC по $60000 = $60.
        p.buy("tBTCUSD", 0.001, 60000.0, self.costs(60.0), date(2024, 1, 1))
        assert p.shares_of("tBTCUSD") == pytest.approx(0.001)
        # Продать ровно столько же — позиция закрывается, шорта нет.
        p.sell("tBTCUSD", 0.001, 66000.0, self.costs(66.0), date(2024, 1, 2))
        assert "tBTCUSD" not in p.positions

    def test_float_dust_does_not_trigger_short(self):
        from datetime import date

        p = Portfolio(250.0)
        p.buy("tXMRUSD", 0.41379999999999995, 150.0, self.costs(62.0), date(2024, 1, 1))
        # Продаём чуть больше из-за float-погрешности — не должно уйти в шорт.
        p.sell("tXMRUSD", 0.4138, 150.0, self.costs(62.0), date(2024, 1, 2))
        assert p.shares_of("tXMRUSD") == 0
        assert "tXMRUSD" not in p.positions
