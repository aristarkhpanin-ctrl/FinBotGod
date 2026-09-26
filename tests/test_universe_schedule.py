"""Тесты фильтра исторического состава индекса (ошибка выживаемости)."""

import pandas as pd

from trading.engine.backtest import BacktestEngine
from trading.research.ledger import HypothesisLedger
from trading.settings import load_settings
from tests.test_backtest import flat_candles


class SeenRecorder:
    """Стратегия-шпион: запоминает, какие бумаги видела в каждый день."""

    name = "spy"

    def __init__(self):
        self.seen: dict[pd.Timestamp, set[str]] = {}

    def params(self):
        return {}

    def target_weights(self, data_until_t):
        last = max(d["date"].iloc[-1] for d in data_until_t.values())
        self.seen[last] = set(data_until_t)
        return {}

    def explain_ru(self):
        return "Ничего не покупать, только смотреть, какие бумаги доступны."


def run(tmp_path, schedule):
    spy = SeenRecorder()
    BacktestEngine(
        candles={"AAA": flat_candles(10), "BBB": flat_candles(10),
                 "_IDX": flat_candles(10)},
        lot_sizes={"AAA": 1, "BBB": 1}, settings=load_settings(), strategy=spy,
        ledger=HypothesisLedger(tmp_path / "ledger.sqlite"), data_hash="тест",
        universe_schedule=schedule,
    ).run()
    return spy.seen


def test_strategy_sees_only_index_members_on_decision_date(tmp_path):
    days = flat_candles(10)["date"]
    # До 5-го дня в индексе только AAA, с 5-го — только BBB.
    schedule = pd.Series({days.iloc[0]: frozenset({"AAA"}),
                          days.iloc[4]: frozenset({"BBB"})})
    seen = run(tmp_path, schedule)
    assert seen[days.iloc[3]] == {"AAA", "_IDX"}      # служебный ряд виден всегда
    assert seen[days.iloc[4]] == {"BBB", "_IDX"}      # срез вступил в силу в свой день


def test_before_first_snapshot_nothing_is_visible(tmp_path):
    days = flat_candles(10)["date"]
    schedule = pd.Series({days.iloc[3]: frozenset({"AAA", "BBB"})})
    seen = run(tmp_path, schedule)
    assert seen[days.iloc[1]] == {"_IDX"}
    assert seen[days.iloc[5]] == {"AAA", "BBB", "_IDX"}


def test_without_schedule_everything_is_visible(tmp_path):
    seen = run(tmp_path, None)
    assert all(v == {"AAA", "BBB", "_IDX"} for v in seen.values())
