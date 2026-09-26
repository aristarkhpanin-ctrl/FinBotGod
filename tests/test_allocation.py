"""Тесты v2: доходности фондов, симулятор распределения, модель риска."""

import numpy as np
import pandas as pd
import pytest

from trading.allocation.assets import BONDS, CASH, EQUITY, fund_returns
from trading.allocation.risk_model import drawdown_labels, risk_features, walk_forward_risk
from trading.allocation.simulator import (
    easing_regime,
    ml_risk_targets,
    simulate,
    static_targets,
    weekly,
)

DAYS = pd.bdate_range("2020-01-01", periods=300)


def make_returns(eq=0.0, bd=0.0, cash=0.0, n=len(DAYS)):
    return pd.DataFrame({EQUITY: eq, BONDS: bd, CASH: cash}, index=DAYS[:n])


class TestFundReturns:
    def test_fund_fee_and_cash_rate(self):
        idx = pd.bdate_range("2024-01-01", periods=6)
        flat = pd.Series(100.0, index=idx)
        rate = pd.Series(0.10, index=pd.date_range("2023-12-01", "2024-02-01", freq="D"))
        r = fund_returns(flat, flat, rate, equity_ter=0.0073, bond_ter=0.0)
        # Пятница → понедельник: 3 календарных дня комиссии и процентов.
        monday = pd.Timestamp("2024-01-08")
        assert r.loc[monday, EQUITY] == pytest.approx(-0.0073 * 3 / 365)
        assert r.loc[monday, CASH] == pytest.approx(0.10 * 3 / 365)
        assert r.loc[monday, BONDS] == 0.0

    def test_days_before_rate_series_are_dropped(self):
        idx = pd.bdate_range("2013-09-02", periods=20)
        flat = pd.Series(100.0, index=idx)
        rate = pd.Series(0.05, index=pd.date_range("2013-09-13", "2013-12-31", freq="D"))
        r = fund_returns(flat, flat, rate)
        assert r.index.min() > pd.Timestamp("2013-09-13")   # ставки ещё не было


class TestSimulator:
    def test_all_cash_equals_money_market(self):
        rets = make_returns(eq=0.01, cash=0.0004)
        sim = simulate(rets, static_targets(rets.index), cost_rate=0.001)
        assert sim.equity.iloc[-1] == pytest.approx(20_000 * 1.0004 ** len(rets))
        assert sim.costs_rub == 0.0          # с денег в деньги — сделок нет

    def test_execution_lag_no_lookahead(self):
        """Решение в день t зарабатывает только с доходности дня t+2."""
        rets = make_returns()
        rets.iloc[10, 0] = 0.10              # скачок акций в день 10
        rets.iloc[11, 0] = 0.10              # и в день 11
        targets = static_targets(rets.index)
        targets.iloc[9:, targets.columns.get_loc(EQUITY)] = 1.0   # решение: день 9
        targets.iloc[9:, targets.columns.get_loc(CASH)] = 0.0
        sim = simulate(rets, targets, cost_rate=0.0)
        # День 10 — сделка на закрытии, скачок дня 10 не пойман; день 11 — пойман.
        assert sim.equity.iloc[10] == pytest.approx(20_000)
        assert sim.equity.iloc[11] == pytest.approx(22_000)

    def test_switch_costs_both_sides(self):
        rets = make_returns(n=5)
        targets = static_targets(rets.index, equity=1.0)
        sim = simulate(rets, targets, cost_rate=0.001)
        # Продать 20 000 денег + купить 20 000 акций = 40 000 оборота.
        assert sim.turnover_rub == pytest.approx(40_000)
        assert sim.equity.iloc[-1] == pytest.approx(20_000 - 40)

    def test_future_targets_do_not_change_past(self):
        rets = make_returns(eq=0.002, cash=0.0003)
        a = static_targets(rets.index, equity=0.5)
        b = a.copy()
        b.iloc[200:] = [0.0, 0.0, 1.0]
        ea = simulate(rets, a, 0.001).equity
        eb = simulate(rets, b, 0.001).equity
        pd.testing.assert_series_equal(ea.iloc[:201], eb.iloc[:201])


class TestRules:
    def test_weekly_holds_decision_between_fridays(self):
        idx = pd.bdate_range("2024-01-01", periods=10)       # пн 1.01 … пт 12.01
        s = pd.Series(range(10), index=idx, dtype=float)
        w = weekly(s)
        assert np.isnan(w.iloc[0])                             # до первой пятницы решения нет
        assert w.loc["2024-01-08"] == 4.0                      # держим решение пятницы 5.01
        assert w.loc["2024-01-12"] == 9.0

    def test_easing_uses_only_past_decisions(self):
        kr = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-06-03", "2024-09-02"]),
                           "rate_pct": [16.0, 16.0, 15.0]})
        idx = pd.to_datetime(["2024-08-30", "2024-09-02", "2024-09-10"])
        e = easing_regime(kr, idx)
        assert e.tolist() == [False, True, True]

    def test_ml_rule_goes_to_cash_when_risk_high(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        proba = pd.Series([0.1] * 5 + [0.9] * 5, index=idx)
        base = pd.Series(0.3, index=idx)
        t = ml_risk_targets(proba, base, k=1.0, w_max=1.0, rest=CASH)
        assert t.loc["2024-01-05", EQUITY] == 1.0              # пятница: риск обычный
        assert t.loc["2024-01-12", EQUITY] == 0.0              # пятница: риск высокий
        assert t.loc["2024-01-12", CASH] == 1.0


class TestRiskModel:
    def test_labels_known_answer(self):
        eq = pd.Series([100, 99, 94, 100, 100, 100], index=pd.bdate_range("2024-01-01", periods=6),
                       dtype=float)
        y = drawdown_labels(eq, horizon=2, depth=0.05)
        assert y.iloc[0] == 1.0          # 94/100 − 1 = −6% в пределах 2 дней
        assert y.iloc[1] == 1.0          # 94/99 − 1 = −5,05% ≤ −5%
        assert y.iloc[2] == 0.0          # дальше только рост
        assert np.isnan(y.iloc[-1])      # будущее ещё не наступило

    def test_features_do_not_look_ahead(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2015-01-01", periods=600)
        eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), index=idx)
        bd = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 600))), index=idx)
        f1 = risk_features(eq, bd)
        eq2 = eq.copy()
        eq2.iloc[400:] *= 0.5                # меняем будущее
        f2 = risk_features(eq2, bd)
        pd.testing.assert_frame_equal(f1.iloc[:400], f2.iloc[:400])

    def test_volatility_clusters_are_predictable(self):
        """На синтетике с режимами волатильности модель должна быть лучше монетки."""
        rng = np.random.default_rng(7)
        n = 2600
        regime = (np.arange(n) // 120) % 2               # спокойно / бурно по 120 дней
        vol = np.where(regime == 1, 0.03, 0.006)
        idx = pd.bdate_range("2010-01-01", periods=n)
        eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol))), index=idx)
        bd = pd.Series(100.0, index=idx)
        fc = walk_forward_risk(risk_features(eq, bd), drawdown_labels(eq), first_test_year=2015)
        assert fc.yearly["AUC модели"].mean() > 0.7
