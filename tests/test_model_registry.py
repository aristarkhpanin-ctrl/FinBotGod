"""Тесты реестра моделей: проверка хеша, неизменность критериев."""

import pytest

from trading.ml.registry import (
    ModelCard,
    ModelIntegrityError,
    ModelRegistry,
    RetirementCriteria,
    RetirementCriteriaLocked,
)


def make_card(name="meta_v1"):
    return ModelCard(
        name=name, train_start="2018-01-01", train_end="2021-12-31",
        features=["mom_20", "vol_20"],
        validation_metrics={"pbo": 0.4, "deflated_sharpe": 0.6},
        retirement=RetirementCriteria(),
        code_hash="abc123",
    )


def test_register_and_load_roundtrip(tmp_path):
    reg = ModelRegistry(tmp_path)
    model = {"weights": [1, 2, 3]}          # любой pickle-совместимый объект
    reg.register("meta_v1", model, make_card())
    loaded, card = reg.load("meta_v1")
    assert loaded == model
    assert card["features"] == ["mom_20", "vol_20"]
    assert card["retirement"]["unconditional_retrain_days"] == 180


def test_tampered_model_rejected(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register("meta_v1", {"w": 1}, make_card())
    # Портим файл весов — хеш перестаёт совпадать.
    (tmp_path / "meta_v1.pkl").write_bytes("подмена".encode("utf-8"))
    with pytest.raises(ModelIntegrityError, match="не совпал"):
        reg.load("meta_v1")


def test_retirement_criteria_cannot_be_overwritten(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register("meta_v1", {"w": 1}, make_card())
    with pytest.raises(RetirementCriteriaLocked, match="задним числом"):
        reg.register("meta_v1", {"w": 2}, make_card())


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry(tmp_path).load("нет")


def test_list_models(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register("a", {"x": 1}, make_card("a"))
    reg.register("b", {"x": 2}, make_card("b"))
    assert reg.list_models() == ["a", "b"]


def test_retirement_criteria_defaults():
    c = RetirementCriteria()
    assert c.feature_psi_above == 0.25          # совпадает с порогом PSI
    assert c.model_drawdown_exceeds == 0.10
