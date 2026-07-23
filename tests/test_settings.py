"""Тесты конфигурации: валидация схемы и понятные ошибки на русском."""

import pytest

from trading.settings import ConfigError, load_settings, load_universe


def test_default_settings_load():
    s = load_settings()
    assert s.capital.start_amount == 20000
    assert s.mode == "backtest"
    assert s.risk.max_positions == 4
    assert s.costs.stress_multiplier == 1.0
    assert s.ml.enabled is False


def test_breakeven_rate_is_about_38_percent():
    """Ключевая цифра ТЗ: 14,25% ставка + 24% инфраструктура ≈ 38%."""
    s = load_settings()
    assert s.breakeven_rate() == pytest.approx(0.3825, abs=1e-4)


def test_universe_has_30_plus_tickers():
    tickers = load_universe()
    assert len(tickers) >= 30
    assert "SBER" in tickers
    assert len(tickers) == len(set(tickers))


def test_missing_file_gives_russian_error(tmp_path):
    with pytest.raises(ConfigError, match="не найден"):
        load_settings(tmp_path / "нет_такого.yaml")


def test_invalid_value_rejected_with_explanation(tmp_path):
    """Доля позиции 1.5 (150% портфеля) — отказ запуска, не стектрейс."""
    import yaml

    from trading.settings import DEFAULT_SETTINGS_PATH

    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["риск"]["макс_доля_одной_позиции"] = 1.5
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="не прошла проверку"):
        load_settings(bad_path)


def test_unknown_key_rejected(tmp_path):
    """Опечатка в имени параметра — ошибка, а не молчаливое игнорирование."""
    import yaml

    from trading.settings import DEFAULT_SETTINGS_PATH

    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["риск"]["макс_доля_одной_позици"] = 0.2  # опечатка: пропала буква
    bad_path = tmp_path / "typo.yaml"
    bad_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(bad_path)


def test_broken_yaml_rejected(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("капитал: [незакрытый список", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_settings(p)
