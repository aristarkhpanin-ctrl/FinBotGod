"""Тесты кэша: атомарность записи и стабильность хеша версии данных."""

import pandas as pd
import pytest

from trading.data.cache import ParquetCache


@pytest.fixture
def cache(tmp_path) -> ParquetCache:
    return ParquetCache(tmp_path / "cache")


def sample_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "close": [100.0 + i for i in range(n)],
        }
    )


def test_roundtrip(cache):
    df = sample_df()
    cache.save("candles/SBER_test", df)
    assert cache.has("candles/SBER_test")
    loaded = cache.load("candles/SBER_test")
    pd.testing.assert_frame_equal(loaded, df)


def test_missing_key(cache):
    assert not cache.has("candles/нет_такого")


def test_keys_listing(cache):
    cache.save("candles/SBER", sample_df())
    cache.save("securities/tqbr", sample_df())
    assert cache.keys() == ["candles/SBER", "securities/tqbr"]


def test_data_version_hash_changes_with_content(cache):
    cache.save("candles/SBER", sample_df())
    h1 = cache.data_version_hash()
    cache.save("candles/GAZP", sample_df(3))
    h2 = cache.data_version_hash()
    assert h1 != h2  # изменились данные — изменился хеш


def test_interrupted_write_does_not_corrupt_cache(cache):
    """Обрыв посреди записи не портит уже сохранённые данные."""
    good = sample_df()
    cache.save("candles/SBER", good)

    class ExplodingFrame(pd.DataFrame):
        def to_parquet(self, *args, **kwargs):
            raise ConnectionError("обрыв сети посреди записи")

    with pytest.raises(ConnectionError):
        cache.save("candles/SBER", ExplodingFrame(sample_df(3)))

    # Старые данные целы, временных файлов не осталось.
    pd.testing.assert_frame_equal(cache.load("candles/SBER"), good)
    leftovers = list(cache.root.rglob("*.tmp"))
    assert leftovers == []


def test_path_traversal_rejected(cache):
    with pytest.raises(ValueError):
        cache.save("../побег", sample_df())
