"""Локальный кэш данных в формате Parquet.

Правила:
* скачанное однажды больше не запрашивается из сети;
* запись атомарна: данные пишутся во временный файл и переименовываются
  одним системным вызовом. Обрыв сети или падение процесса посреди
  загрузки не может испортить уже сохранённый кэш;
* содержимое кэша хешируется — хеш версии данных входит в журнал гипотез.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pandas as pd


class ParquetCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if ".." in key or key.startswith(("/", "\\")):
            raise ValueError(f"Недопустимый ключ кэша: {key!r}")
        return self.root / f"{key}.parquet"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def save(self, key: str, df: pd.DataFrame) -> None:
        """Атомарная запись: temp-файл в том же каталоге + os.replace."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                df.to_parquet(f)
            os.replace(tmp_name, path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def keys(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root)).removesuffix(".parquet")
            for p in self.root.rglob("*.parquet")
        )

    def data_version_hash(self) -> str:
        """Хеш содержимого кэша. Изменились данные — изменился хеш.

        Участвует в идентификаторе прогона в журнале гипотез: результат,
        полученный на других данных, — это другой результат.
        """
        h = hashlib.sha256()
        for key in self.keys():
            path = self._path(key)
            h.update(key.encode("utf-8"))
            h.update(path.read_bytes())
        return h.hexdigest()[:16]
