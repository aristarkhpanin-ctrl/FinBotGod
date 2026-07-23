"""Тесты референсных гипотез: ранжирование, месячная ребалансировка,
исключение неторгующихся бумаг."""

import pandas as pd
import pytest

from trading.strategy.examples.monthly_ranked import (
    AbsoluteMomentum,
    CrossSectionalMomentum,
    MeanReversion,
)


def make_slice(returns: dict[str, float], n_days: int = 30,
               last_date: str = "2024-03-15") -> dict[str, pd.DataFrame]:
    """Срезы данных: каждая бумага равномерно шла к своей итоговой доходности."""
    out = {}
    dates = pd.bdate_range(end=last_date, periods=n_days)
    for secid, total in returns.items():
        daily = (1 + total) ** (1 / (n_days - 1))
        closes = [100.0 * daily ** i for i in range(n_days)]
        out[secid] = pd.DataFrame({"date": dates, "close": closes})
    return out


class TestCrossSectionalMomentum:
    def test_picks_top_by_return(self):
        s = CrossSectionalMomentum(lookback=20, top_n=2)
        data = make_slice({"AAAA": 0.30, "BBBB": 0.10, "CCCC": -0.20})
        weights = s.target_weights(data)
        assert set(weights) == {"AAAA", "BBBB"}
        assert weights["AAAA"] == pytest.approx(0.5)

    def test_monthly_rebalance_holds_weights_within_month(self):
        s = CrossSectionalMomentum(lookback=20, top_n=1)
        first = s.target_weights(
            make_slice({"AAAA": 0.30, "BBBB": 0.10}, last_date="2024-03-05")
        )
        assert set(first) == {"AAAA"}
        # Внутри того же месяца лидер сменился — веса НЕ пересчитываются.
        mid_month = s.target_weights(
            make_slice({"AAAA": 0.05, "BBBB": 0.50}, last_date="2024-03-20")
        )
        assert mid_month == first
        # Новый месяц — пересчёт.
        next_month = s.target_weights(
            make_slice({"AAAA": 0.05, "BBBB": 0.50}, last_date="2024-04-02")
        )
        assert set(next_month) == {"BBBB"}

    def test_not_trading_now_excluded(self):
        s = CrossSectionalMomentum(lookback=20, top_n=1)
        data = make_slice({"AAAA": 0.50, "BBBB": 0.10})
        # AAAA перестала торговаться неделю назад — несмотря на лучший рост.
        data["AAAA"] = data["AAAA"].iloc[:-5]
        weights = s.target_weights(data)
        assert set(weights) == {"BBBB"}

    def test_insufficient_history_excluded(self):
        s = CrossSectionalMomentum(lookback=60, top_n=2)
        weights = s.target_weights(make_slice({"AAAA": 0.30}, n_days=30))
        assert weights == {}

    def test_params_within_tz_limit(self):
        assert len(CrossSectionalMomentum(60, 3).params()) <= 3


class TestAbsoluteMomentum:
    def test_negative_leaders_filtered_out(self):
        s = AbsoluteMomentum(lookback=20, top_n=3)
        data = make_slice({"AAAA": 0.20, "BBBB": -0.05, "CCCC": -0.30})
        weights = s.target_weights(data)
        assert set(weights) == {"AAAA"}          # только растущая
        assert weights["AAAA"] == pytest.approx(1 / 3)  # доля от top_n, не 100%

    def test_all_falling_means_cash(self):
        s = AbsoluteMomentum(lookback=20, top_n=3)
        weights = s.target_weights(make_slice({"AAAA": -0.10, "BBBB": -0.20}))
        assert weights == {}


class TestMeanReversion:
    def test_picks_worst(self):
        s = MeanReversion(lookback=20, top_n=2)
        data = make_slice({"AAAA": 0.30, "BBBB": -0.10, "CCCC": -0.25})
        weights = s.target_weights(data)
        assert set(weights) == {"BBBB", "CCCC"}


class TestExplanations:
    def test_every_hypothesis_explains_itself_in_russian(self):
        for s in (
            CrossSectionalMomentum(120, 3),
            AbsoluteMomentum(120, 3),
            MeanReversion(10, 2),
        ):
            text = s.explain_ru()
            assert "Раз в месяц" in text
            assert str(s.lookback) in text
