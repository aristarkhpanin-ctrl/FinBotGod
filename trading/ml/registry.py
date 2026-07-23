"""Реестр моделей — ЧАСТЬ II (ТЗ, разделы 22–23).

Каждая обученная модель хранится вместе с: датами обучающей выборки,
списком признаков, метриками валидации, КРИТЕРИЯМИ ВЫВОДА ИЗ ЭКСПЛУАТАЦИИ
(зафиксированными ДО первого боевого решения) и хешем кода/весов.

Два жёстких правила:
* модель не грузится из pickle без проверки хеша — несовпадение = отказ
  (защита от подмены и порчи файла, раздел 23);
* критерии вывода из эксплуатации задаются при регистрации и не могут
  быть дописаны позже. Критерии, добавленные после того, как модель начала
  терять деньги, не имеют силы (раздел 22).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class ModelIntegrityError(Exception):
    """Хеш модели не совпал — файл повреждён или подменён."""


class RetirementCriteriaLocked(Exception):
    """Попытка изменить критерии вывода из эксплуатации после регистрации."""


@dataclass(frozen=True)
class RetirementCriteria:
    """Условия автоматического отключения модели, заданные ДО запуска."""

    rolling_sharpe_60d_below: float = 0.0
    model_drawdown_exceeds: float = 0.10
    feature_psi_above: float = 0.25
    wrong_filter_rate_above: float = 0.60
    unconditional_retrain_days: int = 180


@dataclass(frozen=True)
class ModelCard:
    name: str
    train_start: str
    train_end: str
    features: list[str]
    validation_metrics: dict
    retirement: RetirementCriteria
    code_hash: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class ModelRegistry:
    def __init__(self, root: str | Path = "journal/models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, name: str) -> tuple[Path, Path]:
        return self.root / f"{name}.pkl", self.root / f"{name}.card.json"

    def register(self, name: str, model, card: ModelCard) -> str:
        """Сохраняет модель и карточку. Возвращает хеш весов.

        Хеш весов включается в карточку — при загрузке он проверяется.
        """
        pkl_path, card_path = self._paths(name)
        if card_path.exists():
            raise RetirementCriteriaLocked(
                f"Модель «{name}» уже зарегистрирована. Перерегистрация под тем "
                f"же именем запрещена — критерии вывода из эксплуатации нельзя "
                f"переписать задним числом. Используйте новое имя (например, "
                f"«{name}_v2»)."
            )
        blob = pickle.dumps(model)
        weights_hash = hashlib.sha256(blob).hexdigest()
        pkl_path.write_bytes(blob)
        payload = asdict(card)
        payload["weights_hash"] = weights_hash
        card_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return weights_hash

    def load(self, name: str):
        """Загружает модель с проверкой хеша. Несовпадение — отказ."""
        pkl_path, card_path = self._paths(name)
        if not pkl_path.exists() or not card_path.exists():
            raise FileNotFoundError(f"Модель «{name}» не найдена в реестре.")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        blob = pkl_path.read_bytes()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != card["weights_hash"]:
            raise ModelIntegrityError(
                f"Хеш модели «{name}» не совпал: файл повреждён или подменён. "
                f"Запуск отклонён (ТЗ, раздел 23)."
            )
        return pickle.loads(blob), card

    def list_models(self) -> list[str]:
        return sorted(p.stem.replace(".card", "")
                      for p in self.root.glob("*.card.json"))
