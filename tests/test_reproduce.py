"""Тест воспроизводимости: прогон восстанавливается из журнала по хешу."""

from trading.research.ledger import HypothesisLedger
from trading.research.reproduce import describe_run


def test_describe_existing_run(tmp_path):
    ledger = HypothesisLedger(tmp_path / "ledger.sqlite")
    h = ledger.record(
        hypothesis="Моментум 120 дн., top-4",
        source="llm",
        params={"lookback": 120, "top_n": 4},
        data_hash="deadbeef",
        metrics_oos={"annual_return": -0.008},
    )
    text = describe_run(ledger, h)
    assert "Моментум 120" in text
    assert "deadbeef" in text
    assert "lookback" in text
    assert "детерминирован" in text


def test_describe_missing_run(tmp_path):
    ledger = HypothesisLedger(tmp_path / "ledger.sqlite")
    assert "не найден" in describe_run(ledger, "нетничего")
