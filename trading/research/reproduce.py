"""Воспроизводимость прогонов — ЧАСТЬ II (ТЗ, раздел 23).

Любой результат из журнала гипотез восстанавливается одной командой:

    python -m trading.research.reproduce <run_hash>

Печатает полную конфигурацию прогона: гипотезу, источник, все параметры,
хеш версии данных и записанные метрики. Движок детерминирован, поэтому
та же конфигурация на тех же данных (совпадающий хеш) даёт тот же
результат — идентификатор прогона это гарантирует.

Источники случайности зафиксированы: движок не использует ГСЧ; MDA
принимает random_state; решатель логистической регрессии детерминирован.
"""

from __future__ import annotations

import argparse
import json
import sys

from trading.research.ledger import HypothesisLedger


def describe_run(ledger: HypothesisLedger, run_hash: str) -> str:
    record = ledger.get(run_hash)
    if record is None:
        return f"Прогон {run_hash} в журнале не найден."
    lines = [
        f"Прогон: {record.run_hash}",
        f"Записан: {record.created_at}",
        f"Гипотеза: {record.hypothesis}",
        f"Источник: {record.source}",
        f"Хеш конфигурации: {record.config_hash}",
        f"Хеш версии данных: {record.data_hash}",
        "",
        "Параметры:",
        json.dumps(record.params, ensure_ascii=False, indent=2),
        "",
        "Метрики in-sample:",
        json.dumps(record.metrics_is, ensure_ascii=False, indent=2),
        "",
        "Метрики out-of-sample:",
        json.dumps(record.metrics_oos, ensure_ascii=False, indent=2),
        "",
        "Чтобы получить тот же результат: запустить ту же конфигурацию на "
        "данных с хешем " + record.data_hash + " — движок детерминирован.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Восстановление прогона по хешу")
    parser.add_argument("run_hash", help="Идентификатор прогона из журнала гипотез")
    parser.add_argument("--db", default="journal/hypotheses.sqlite")
    args = parser.parse_args(argv)

    ledger = HypothesisLedger(args.db)
    print(describe_run(ledger, args.run_hash))
    ledger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
