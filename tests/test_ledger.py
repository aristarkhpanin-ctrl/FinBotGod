"""Тесты журнала гипотез — самого критичного модуля системы."""

import sqlite3

import pytest

from trading.research.ledger import HypothesisLedger


@pytest.fixture
def ledger(tmp_path) -> HypothesisLedger:
    return HypothesisLedger(tmp_path / "ledger.sqlite")


def test_record_and_read_back(ledger):
    run_hash = ledger.record(
        hypothesis="Моментум 20 дней на голубых фишках",
        source="человек",
        params={"lookback": 20, "top_n": 3},
        data_hash="abc123",
        metrics_is={"annual_return": 0.42},
        metrics_oos={"annual_return": 0.11},
    )
    record = ledger.get(run_hash)
    assert record is not None
    assert record.params == {"lookback": 20, "top_n": 3}
    assert record.metrics_oos["annual_return"] == 0.11
    assert record.source == "человек"


def test_run_hash_is_deterministic(ledger):
    """Та же конфигурация на тех же данных → тот же идентификатор."""
    kwargs = dict(
        hypothesis="Тест", source="перебор",
        params={"a": 1}, data_hash="d1",
    )
    h1 = ledger.record(**kwargs)
    h2 = ledger.record(**kwargs)
    assert h1 == h2
    assert ledger.count() == 1  # повтор не раздувает журнал


def test_different_data_different_hash(ledger):
    h1 = ledger.record(hypothesis="Тест", source="перебор", params={"a": 1}, data_hash="d1")
    h2 = ledger.record(hypothesis="Тест", source="перебор", params={"a": 1}, data_hash="d2")
    assert h1 != h2  # изменились данные — изменился прогон


def test_empty_hypothesis_rejected(ledger):
    with pytest.raises(ValueError):
        ledger.record(hypothesis="  ", source="человек", params={}, data_hash="d")


def test_invalid_source_rejected(ledger):
    with pytest.raises(sqlite3.IntegrityError):
        ledger.record(hypothesis="Тест", source="магия", params={}, data_hash="d")


def test_ledger_is_append_only(ledger):
    """У журнала нет методов изменения и удаления записей."""
    forbidden = [m for m in dir(ledger) if any(
        word in m.lower() for word in ("delete", "remove", "update", "drop", "clear")
    )]
    assert forbidden == []


def test_summary_shows_median_not_only_best(ledger):
    """Защита от множественного тестирования: медиана, не только максимум."""
    for i in range(15):
        ledger.record(
            hypothesis=f"Вариант {i}", source="перебор",
            params={"вариант": i}, data_hash="d1",
            metrics_oos={"annual_return": 0.01 * i},
        )
    text = ledger.summary_ru()
    assert "Проверено конфигураций: 15" in text
    assert "Лучшая по OOS-доходности: 14.0%" in text
    assert "Медианная по OOS: 7.0%" in text
    assert "не является" in text  # предупреждение о незначимости


def test_summary_without_metrics(ledger):
    ledger.record(hypothesis="Тест", source="человек", params={}, data_hash="d")
    assert "выводы делать не по чему" in ledger.summary_ru()


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "ledger.sqlite"
    first = HypothesisLedger(path)
    first.record(hypothesis="Тест", source="llm", params={"x": 1}, data_hash="d")
    first.close()

    second = HypothesisLedger(path)
    assert second.count() == 1
    assert second.all_runs()[0].source == "llm"
