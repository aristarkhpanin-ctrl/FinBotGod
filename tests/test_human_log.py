"""Тесты человекочитаемого журнала — фаза 7 (ТЗ, раздел 10).

Два потока (decisions.log на русском + events.jsonl), обязательная
строка сравнения с фондом ДР в каждой ежедневной сводке, ежемесячный
отчёт текстом и CSV.
"""

import json
from datetime import date

import pandas as pd
import pytest

from trading.engine.backtest import BacktestEngine
from trading.reporting.human_log import DecisionJournal, monthly_report
from trading.research.ledger import HypothesisLedger
from trading.settings import load_settings
from trading.strategy.examples.buy_and_hold import BuyAndHold
from tests.test_backtest import flat_candles


@pytest.fixture
def journal(tmp_path) -> DecisionJournal:
    return DecisionJournal(tmp_path / "logs")


class TestTwoStreams:
    def test_decision_goes_to_both_files(self, journal):
        journal.log_decision("2026-07-23  ПОКУПКА  SBER  2 лота", тикер="SBER")
        journal.close()
        decisions = (journal.dir / "decisions.log").read_text(encoding="utf-8")
        assert "ПОКУПКА  SBER" in decisions
        events = (journal.dir / "events.jsonl").read_text(encoding="utf-8")
        event = json.loads(events.strip())
        assert event["тикер"] == "SBER"
        assert "ПОКУПКА" in event["text"]
        assert "ts" in event

    def test_jsonl_is_line_delimited_json(self, journal):
        for i in range(3):
            journal.log_decision(f"решение {i}", номер=i)
        journal.close()
        lines = (journal.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["номер"] for line in lines] == [0, 1, 2]


class TestDailySummary:
    def test_benchmark_line_is_mandatory(self, journal):
        """Последняя строка каждой сводки — сравнение с фондом ДР."""
        text = journal.daily_summary(
            day=date(2026, 7, 23), equity=19_940, prev_equity=20_000,
            costs_today=12.40, costs_total=340.0, start_capital=20_000,
            risk_free_rate=0.1425, n_positions=1, max_positions=4,
        )
        assert "ИТОГИ ДНЯ" in text
        assert "Портфель: 19 940 ₽" in text
        assert "-0.30% за день" in text
        assert "Издержки за день: 12.40 ₽" in text
        assert "Издержки с начала: 340.00 ₽ (1.70% капитала)" in text
        # Обязательная последняя строка.
        assert text.strip().splitlines()[-1].strip().startswith(
            "Бенчмарк за тот же период: фонд ДР дал"
        )
        # Ставка 14,25% годовых ≈ +0,036% в день.
        assert "+0.036%" in text


class TestMonthlyReport:
    def make_equity(self):
        dates = pd.bdate_range("2024-01-01", "2024-03-29")
        # Январь вверх, февраль вниз, март вверх.
        values = []
        v = 20_000.0
        for d in dates:
            v *= 1.002 if d.month != 2 else 0.999
            values.append(v)
        return pd.Series(values, index=dates)

    def test_monthly_rows_and_benchmark_comparison(self):
        text, table = monthly_report(
            self.make_equity(), fills=[], risk_free_rate=0.1425,
            guard_events=["2024-02-15  ПРЕДОХРАНИТЕЛЬ daily_loss_limit: ..."],
        )
        assert len(table) == 3
        assert list(table["месяц"]) == ["2024-01", "2024-02", "2024-03"]
        assert "обогнал фонд ДР" in text
        assert "ПРОИГРАЛ фонду ДР" in text          # февраль был убыточным
        assert "ПРЕДОХРАНИТЕЛИ" in text             # событие попало в свой месяц
        assert table.loc[1, "предохранители"] != ""
        assert "Итого: обогнал фонд денежного рынка в" in text


class TestEngineWiring:
    def test_engine_writes_journal_files(self, tmp_path):
        journal = DecisionJournal(tmp_path / "logs")
        result = BacktestEngine(
            candles={"TEST": flat_candles(10)},
            lot_sizes={"TEST": 10},
            settings=load_settings(),
            strategy=BuyAndHold("TEST", 0.25),
            ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
            data_hash="тест",
            journal=journal,
        ).run()
        journal.close()

        decisions = (journal.dir / "decisions.log").read_text(encoding="utf-8")
        assert "ПОКУПКА" in decisions
        assert decisions.count("ИТОГИ ДНЯ") == 10   # сводка каждый день
        assert "Бенчмарк за тот же период" in decisions

        events = (journal.dir / "events.jsonl").read_text(encoding="utf-8")
        for line in events.splitlines():
            json.loads(line)   # каждая строка — валидный JSON

        assert (journal.dir / "monthly.csv").exists()
        assert "ЕЖЕМЕСЯЧНЫЙ ОТЧЁТ" in result.monthly_report_ru

    def test_daily_costs_in_summary_only_once(self, tmp_path):
        """Издержки покупки попадают в сводку своего дня, дальше — нули."""
        journal = DecisionJournal(tmp_path / "logs")
        BacktestEngine(
            candles={"TEST": flat_candles(4)},
            lot_sizes={"TEST": 10},
            settings=load_settings(),
            strategy=BuyAndHold("TEST", 0.25),
            ledger=HypothesisLedger(tmp_path / "ledger.sqlite"),
            data_hash="тест",
            journal=journal,
        ).run()
        journal.close()
        text = (journal.dir / "decisions.log").read_text(encoding="utf-8")
        assert text.count("Издержки за день: 8.12 ₽") == 1   # только день покупки
        assert "Издержки с начала: 8.12 ₽" in text
