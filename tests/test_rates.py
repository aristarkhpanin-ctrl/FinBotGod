"""Тесты дохода на свободные деньги и исторической ставки денежного рынка."""

import math

import pandas as pd
import pytest

from trading.data.rates import (
    money_market_benchmark_series,
    money_market_growth,
    money_market_rate,
    parse_cbr_key_rate_html,
)
from trading.engine.backtest import BacktestEngine
from trading.reporting.metrics import compute_metrics
from trading.research.ledger import HypothesisLedger
from trading.settings import load_settings
from trading.strategy.examples.buy_and_hold import BuyAndHold
from tests.test_backtest import flat_candles

CBR_HTML = """
<table><tr><th>Дата</th><th>Ставка</th></tr>
<tr><td>03.01.2024</td><td>16,00</td></tr>
<tr><td>29.07.2024</td><td>18,00</td></tr>
<tr><td>28.10.2024</td><td>21,00</td></tr>
</table>
"""


class NoTrades:
    """Стратегия, которая никогда не входит в рынок: весь капитал — кэш."""

    name = "no_trades"

    def params(self):
        return {}

    def target_weights(self, data_until_t):
        return {}

    def explain_ru(self):
        return "Никогда не покупать — весь капитал в фонде денежного рынка."


class TestParsing:
    def test_parse_cbr_table(self):
        df = parse_cbr_key_rate_html(CBR_HTML)
        assert list(df["rate_pct"]) == [16.0, 18.0, 21.0]
        assert df["date"].iloc[1] == pd.Timestamp("2024-07-29")

    def test_money_market_rate_is_asof_minus_spread(self):
        df = parse_cbr_key_rate_html(CBR_HTML)
        r = money_market_rate(df, spread_pp=0.5, end="2024-12-31")
        assert r.loc["2024-07-28"] == pytest.approx(0.155)   # ещё 16% − 0,5
        assert r.loc["2024-07-29"] == pytest.approx(0.175)   # решение вступило
        assert r.loc["2024-12-31"] == pytest.approx(0.205)

    def test_growth_beyond_series_end_uses_last_rate(self):
        """Ряд ставки кончается раньше периода — дни не теряются."""
        short = pd.Series(0.10, index=pd.date_range("2024-01-01", periods=30, freq="D"))
        g = money_market_growth(short, "2024-01-01", "2024-12-31")
        assert g == pytest.approx((1 + 0.10 / 365) ** 365, rel=1e-12)

    def test_growth_known_answer(self):
        # 10% годовых, 365 дней ежедневной капитализации: (1+0.1/365)^365.
        rate = pd.Series(0.10, index=pd.date_range("2024-01-01", periods=400, freq="D"))
        g = money_market_growth(rate, "2024-01-01", "2024-12-31")
        assert g == pytest.approx((1 + 0.10 / 365) ** 365, rel=1e-12)

    def test_benchmark_series_annualizes(self):
        rate = pd.Series(0.12, index=pd.date_range("2020-01-01", "2023-01-01", freq="D"))
        bm = money_market_benchmark_series(20_000, "2020-01-01", "2022-01-01", rate)
        # Ежедневная капитализация 12%: (1 + 0,12/365)^365,25 − 1 ≈ 12,76%.
        assert bm["annual_return"] == pytest.approx((1 + 0.12 / 365) ** 365.25 - 1, rel=1e-9)
        assert bm["average_rate"] == pytest.approx(0.12)


class TestEngineCashInterest:
    def run(self, tmp_path, strategy, cash_rate):
        return BacktestEngine(
            candles={"TEST": flat_candles(10)}, lot_sizes={"TEST": 10},
            settings=load_settings(), strategy=strategy,
            ledger=HypothesisLedger(tmp_path / "ledger.sqlite"), data_hash="тест",
            cash_rate=cash_rate,
        ).run()

    def test_idle_cash_earns_money_market(self, tmp_path):
        """Стратегия «ничего не покупать» = фонд денежного рынка до копейки."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        rate = pd.Series(0.10, index=pd.date_range("2023-12-01", "2024-02-01", freq="D"))
        result = self.run(tmp_path, NoTrades(), rate)
        expected = 20_000.0
        for prev, day in zip(dates[:-1], dates[1:]):
            expected *= 1 + 0.10 * (day - prev).days / 365
        assert result.equity.iloc[-1] == pytest.approx(expected, abs=0.005)
        assert result.metrics["cash_interest"] == pytest.approx(expected - 20_000, abs=0.005)

    def test_without_rate_cash_is_zero_as_before(self, tmp_path):
        result = self.run(tmp_path, NoTrades(), None)
        assert result.equity.iloc[-1] == pytest.approx(20_000.0)
        assert result.metrics["cash_interest"] == 0.0
        assert any("НЕ приносят дохода" in t for t in result.limitations)

    def test_invested_part_does_not_earn_interest(self, tmp_path):
        """Проценты начисляются только на свободные деньги, а не на акции."""
        rate = pd.Series(0.10, index=pd.date_range("2023-12-01", "2024-02-01", freq="D"))
        idle = self.run(tmp_path / "a", NoTrades(), rate)
        invested = self.run(tmp_path / "b", BuyAndHold("TEST", 0.25), rate)
        assert invested.metrics["cash_interest"] < idle.metrics["cash_interest"]

    def test_report_compares_with_historical_rate(self, tmp_path):
        rate = pd.Series(0.10, index=pd.date_range("2023-12-01", "2024-02-01", freq="D"))
        report = self.run(tmp_path, NoTrades(), rate).report_ru
        assert "ист. ставка ЦБ" in report
        assert "Доход на свободные деньги" in report


def test_sharpe_with_rf_series_matches_constant_case():
    """Постоянная ставка, переданная рядом, даёт тот же знак избыточной доходности."""
    idx = pd.bdate_range("2024-01-01", periods=250)
    equity = pd.Series([100 * 1.0004 ** i for i in range(250)], index=idx)
    high = pd.Series(0.30, index=pd.date_range("2023-12-01", "2025-01-31", freq="D"))
    low = pd.Series(0.01, index=high.index)
    assert compute_metrics(equity, 0.0, rf_series=high)["sharpe"] < 0 < \
        compute_metrics(equity, 0.0, rf_series=low)["sharpe"]
