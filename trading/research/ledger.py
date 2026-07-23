"""ЖУРНАЛ ВСЕХ ГИПОТЕЗ — самый критичный модуль системы (ТЗ, раздел 16).

При 500 испытаниях лучший результат выглядит блестяще, даже если ни одна
идея не работает — чистая статистика экстремальных значений. Единственная
защита — полный журнал всех прогонов.

Правила:
* каждый прогон бэктеста записывается автоматически, отключить нельзя;
* журнал только пополняется: методов изменения и удаления записей нет;
* хранится: хеш конфигурации, хеш версии данных, полный набор параметров,
  источник гипотезы (человек / LLM / перебор), метрики in-sample и
  out-of-sample, временная метка;
* сводка показывает не только лучшую конфигурацию, но и медиану по всем —
  и честное предупреждение о множественном тестировании.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

HypothesisSource = Literal["человек", "llm", "перебор"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_hash        TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    data_hash       TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('человек', 'llm', 'перебор')),
    hypothesis      TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    metrics_is_json TEXT NOT NULL,
    metrics_oos_json TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class RunRecord:
    run_hash: str
    created_at: str
    config_hash: str
    data_hash: str
    source: str
    hypothesis: str
    params: dict
    metrics_is: dict
    metrics_oos: dict


def _stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class HypothesisLedger:
    """Журнал гипотез на SQLite. Только добавление, никакого удаления."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record(
        self,
        hypothesis: str,
        source: HypothesisSource,
        params: dict,
        data_hash: str,
        metrics_is: dict | None = None,
        metrics_oos: dict | None = None,
    ) -> str:
        """Записывает прогон. Возвращает run_hash — идентификатор прогона.

        run_hash детерминирован: та же конфигурация на тех же данных даёт
        тот же идентификатор, и повторный прогон не раздувает статистику
        множественного тестирования.
        """
        if not hypothesis.strip():
            raise ValueError("Гипотеза не может быть пустой строкой.")
        config_hash = _stable_hash(params)
        run_hash = _stable_hash(
            {"config": config_hash, "data": data_hash, "hypothesis": hypothesis}
        )
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_hash, created_at, config_hash, data_hash, source,
                hypothesis, params_json, metrics_is_json, metrics_oos_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_hash, created_at, config_hash, data_hash, source, hypothesis,
                json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
                json.dumps(metrics_is or {}, ensure_ascii=False, default=str),
                json.dumps(metrics_oos or {}, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()
        return run_hash

    def all_runs(self) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT run_hash, created_at, config_hash, data_hash, source, "
            "hypothesis, params_json, metrics_is_json, metrics_oos_json "
            "FROM runs ORDER BY created_at"
        ).fetchall()
        return [
            RunRecord(
                run_hash=r[0], created_at=r[1], config_hash=r[2], data_hash=r[3],
                source=r[4], hypothesis=r[5], params=json.loads(r[6]),
                metrics_is=json.loads(r[7]), metrics_oos=json.loads(r[8]),
            )
            for r in rows
        ]

    def get(self, run_hash: str) -> RunRecord | None:
        for r in self.all_runs():
            if r.run_hash == run_hash:
                return r
        return None

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def summary_ru(self, oos_metric: str = "annual_return") -> str:
        """Сводка с поправкой на множественное тестирование (ТЗ, раздел 8).

        Показывает лучшую И медианную конфигурацию. Эту защиту от
        самообмана нельзя отключать.
        """
        runs = self.all_runs()
        values = [
            float(r.metrics_oos[oos_metric])
            for r in runs
            if oos_metric in r.metrics_oos
        ]
        n = len(runs)
        if not values:
            return (
                f"Проверено конфигураций: {n}\n"
                f"OOS-метрик пока нет — выводы делать не по чему."
            )
        best = max(values)
        median = statistics.median(values)
        lines = [
            f"Проверено конфигураций: {n}",
            f"Лучшая по OOS-доходности: {best:.1%} годовых",
            f"Медианная по OOS: {median:.1%} годовых",
        ]
        if n > 10:
            lines.append(
                f"Поправка на множественное тестирование: при {n} испытаниях "
                f"результат лучшей конфигурации сам по себе не является "
                f"статистически значимым. Доверять можно медиане, а не максимуму."
            )
        return "\n".join(lines)
