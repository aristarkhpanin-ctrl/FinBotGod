"""Тесты дивидендов: разбор источников, экс-даты Т+2/Т+1, зачисление в движке."""

import pandas as pd
import pytest

from trading.core.costs import CostModel
from trading.core.portfolio import Portfolio
from trading.data.dividends import ex_dividend_dates, parse_dohod, parse_smartlab
from trading.engine.backtest import BacktestEngine
from trading.research.ledger import HypothesisLedger
from trading.settings import load_settings
from trading.strategy.examples.buy_and_hold import BuyAndHold
from tests.test_backtest import flat_candles

DOHOD = """
<table><tr><td>16%</td></tr></table>
<table>
<tr><th>Дата объявления дивиденда</th><th>Дата закрытия реестра</th><th>Год для учета дивиденда</th><th>Дивиденд</th></tr>
<tr><td>n/a</td><td>20.07.2027 (прогноз)</td><td>n/a</td><td>44.53</td></tr>
<tr><td>22.04.2025</td><td>18.07.2025</td><td>2025</td><td>34.84</td></tr>
<tr><td>24.04.2023</td><td>11.05.2023</td><td>2023</td><td>25</td></tr>
</table>
"""

SMARTLAB = """
<table>
<tr><td>Выплаченные дивиденды</td></tr>
<tr><th>Тикер</th><th>дата T-1</th><th>дата отсечки</th><th>Период</th><th>дивиденд</th><th>Цена акции</th></tr>
<tr><td>POLY</td><td>05.05.2021</td><td>07.05.2021</td><td>2020</td><td>89,52₽</td><td>1 700</td></tr>
</table>
"""


class TestParsing:
    def test_dohod_skips_forecast(self):
        df = parse_dohod(DOHOD)
        assert len(df) == 2                          # прогноз отброшен
        assert df["amount"].tolist() == [34.84, 25.0]
        assert df["record_date"].iloc[0] == pd.Timestamp("2025-07-18")

    def test_smartlab_reads_last_day_with_rights(self):
        df = parse_smartlab(SMARTLAB)
        assert len(df) == 1
        assert df["amount"].iloc[0] == pytest.approx(89.52)
        assert df["last_day_with_rights"].iloc[0] == pd.Timestamp("2021-05-05")


class TestExDates:
    cal = pd.bdate_range("2021-01-01", "2026-12-31")

    def test_t_plus_1_era(self):
        # Реестр 18.07.2025 (пятница), режим Т+1: экс-дата = сам день реестра.
        divs = pd.DataFrame({"record_date": [pd.Timestamp("2025-07-18")], "amount": [34.84],
                             "last_day_with_rights": [pd.NaT]})
        ex = ex_dividend_dates(divs, self.cal)
        assert ex["ex_date"].iloc[0] == pd.Timestamp("2025-07-18")

    def test_t_plus_2_era(self):
        # Реестр 11.05.2023 (четверг), режим Т+2: последний день с правом —
        # вторник 09.05, экс-дата — среда 10.05.
        divs = pd.DataFrame({"record_date": [pd.Timestamp("2023-05-11")], "amount": [25.0],
                             "last_day_with_rights": [pd.NaT]})
        ex = ex_dividend_dates(divs, self.cal)
        assert ex["ex_date"].iloc[0] == pd.Timestamp("2023-05-10")

    def test_explicit_last_day_with_rights_wins(self):
        divs = parse_smartlab(SMARTLAB)
        ex = ex_dividend_dates(divs, self.cal)
        assert ex["ex_date"].iloc[0] == pd.Timestamp("2021-05-06")   # день после T-1


class TestEngineDividends:
    def run(self, tmp_path, strategy, ex_date, amount=10.0, **kw):
        candles = flat_candles(10)             # цена 100 ₽, лот 10 шт
        divs = {"TEST": pd.DataFrame({"ex_date": [ex_date], "amount": [amount]})}
        return BacktestEngine(
            candles={"TEST": candles}, lot_sizes={"TEST": 10}, settings=load_settings(),
            strategy=strategy, ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
            data_hash="тест", dividends=divs, **kw,
        ).run()

    def test_holder_receives_net_dividend(self, tmp_path):
        # BuyAndHold покупает 50 шт на открытии 2-го дня; экс-дата — 5-й день.
        ex = flat_candles(10)["date"].iloc[4]
        result = self.run(tmp_path, BuyAndHold("TEST", 0.25), ex)
        # 50 шт × 10 ₽ = 500 ₽, НДФЛ 13% = 65 ₽, зачислено 435 ₽.
        assert result.metrics["dividends_net"] == pytest.approx(435.0)
        assert result.metrics["dividend_tax"] == pytest.approx(65.0)
        assert result.equity.iloc[-1] == pytest.approx(19_991.875 + 435.0, abs=0.005)
        assert any("ДИВИДЕНД" in d for d in result.decisions)

    def test_buying_on_ex_date_gives_no_dividend(self, tmp_path):
        # Покупка на открытии 2-го дня, а экс-дата — тоже 2-й день: права нет.
        ex = flat_candles(10)["date"].iloc[1]
        result = self.run(tmp_path, BuyAndHold("TEST", 0.25), ex)
        assert result.metrics["dividends_net"] == 0.0


def test_short_pays_gross_dividend():
    p = Portfolio(20_000, allow_short=True)
    costs = CostModel(load_settings().costs).trade_costs(5_000, 1e9)
    p.sell("SBER", 50, 100.0, costs, pd.Timestamp("2024-01-02").date())
    credited, tax = p.receive_dividend("SBER", 10.0, 0.13)
    assert credited == pytest.approx(-500.0)     # шорт отдаёт полную сумму
    assert tax == 0.0
