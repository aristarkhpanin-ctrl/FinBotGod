"""Логирование: два потребителя — человек и машина.

* консоль / ``logs/decisions.log`` — человекочитаемые строки на русском;
* ``logs/events.jsonl`` — структурированный JSON для анализа.

Полноценный журнал решений — фаза 7. Здесь минимальная база, чтобы
предупреждения валидации данных с первого дня писались в оба потока,
а не терялись.
"""

from __future__ import annotations

import logging
from pathlib import Path

import structlog


def setup_logging(log_dir: str | Path = "logs", console_level: int = logging.INFO) -> None:
    """Настраивает structlog: консоль (текст) + events.jsonl (JSON)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))

    jsonl = logging.FileHandler(log_dir / "events.jsonl", encoding="utf-8")
    jsonl.setLevel(logging.DEBUG)
    jsonl.setFormatter(logging.Formatter("%(message)s"))
    # Обработчик пишет только строки, которые structlog уже отрендерил в JSON.
    jsonl.addFilter(lambda record: getattr(record, "is_json", False))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = [console]

    json_logger = logging.getLogger("events_jsonl")
    json_logger.setLevel(logging.DEBUG)
    json_logger.handlers = [jsonl]
    json_logger.propagate = False

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.add_log_level,
            _fanout_processor,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def _fanout_processor(logger, method_name, event_dict):
    """Дублирует событие в events.jsonl в виде JSON, консоли отдаёт текст."""
    import json

    json_line = json.dumps(event_dict, ensure_ascii=False, default=str)
    record = logging.LogRecord(
        name="events_jsonl", level=logging.INFO, pathname="", lineno=0,
        msg=json_line, args=(), exc_info=None,
    )
    record.is_json = True
    logging.getLogger("events_jsonl").handle(record)

    event = event_dict.pop("event", "")
    extras = " ".join(
        f"{k}={v}" for k, v in event_dict.items() if k not in ("timestamp", "level")
    )
    ts = event_dict.get("timestamp", "")
    return f"{ts}  {event}" + (f"  [{extras}]" if extras else "")


def get_logger(name: str = "trading"):
    return structlog.get_logger(name)
