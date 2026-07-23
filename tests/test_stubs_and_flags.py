"""Тесты заглушек будущих фаз и фича-флага ML.

Модули Части II существуют, импортируются, но честно отказываются
работать до своего времени.
"""

import pytest

from trading.ml import MlDisabledError, ensure_ml_enabled
from trading.settings import load_settings


def test_ml_disabled_by_default():
    settings = load_settings()
    assert settings.ml.enabled is False
    with pytest.raises(MlDisabledError, match="раздела 24"):
        ensure_ml_enabled(settings)


def test_ml_guard_passes_when_enabled():
    settings = load_settings()
    settings.ml.enabled = True  # только в тесте; в жизни — после фазы 8
    ensure_ml_enabled(settings)


def test_all_stub_modules_importable():
    """Структура проекта из раздела 3 ТЗ существует целиком."""
    import trading.core.events
    import trading.core.portfolio
    import trading.core.sizing
    import trading.engine.backtest
    import trading.engine.live
    import trading.engine.paper
    import trading.execution.alor
    import trading.execution.base
    import trading.execution.simulated
    import trading.features.store
    import trading.features.transforms
    import trading.ml.labeling
    import trading.ml.models
    import trading.ml.monitoring
    import trading.ml.registry
    import trading.ml.validation
    import trading.reporting.human_log
    import trading.reporting.metrics
    import trading.reporting.report
    import trading.research.llm
    import trading.risk.guards
    import trading.strategy.base  # noqa: F401


@pytest.mark.parametrize(
    "call",
    [
        lambda: __import__("trading.research.llm", fromlist=["x"]).generate_hypotheses(),
        lambda: __import__("trading.research.llm", fromlist=["x"]).text_to_features(),
    ],
)
def test_stubs_refuse_to_pretend_they_work(call):
    """Ещё не реализованные модули честно отказываются работать.

    По мере реализации Части II строки отсюда переезжают в свои тесты."""
    with pytest.raises(NotImplementedError):
        call()


def test_breakeven_header_present_in_reports():
    """Порог безубыточности печатается в каждом отчёте (ТЗ, раздел 1)."""
    from trading.reporting.report import breakeven_header_ru

    header = breakeven_header_ru(load_settings())
    assert "ПОРОГ БЕЗУБЫТОЧНОСТИ" in header
    assert "38%" in header
    assert "14.25%" in header
