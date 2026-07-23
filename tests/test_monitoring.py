"""Тесты мониторинга: CUSUM, PSI, KS на синтетике с известным ответом."""

import numpy as np
import pandas as pd
import pytest

from trading.ml.monitoring import (
    cusum_events,
    feature_drift_psi,
    ks_drift,
    population_stability_index,
)


class TestCusum:
    def test_detects_level_shift(self):
        # Ряд стоит на 0, затем прыгает на 10 — CUSUM обязан среагировать.
        s = pd.Series([0.0] * 20 + [10.0] * 20)
        events = cusum_events(s, threshold=3.0)
        assert len(events) >= 1
        assert events[0][1] == 1              # сдвиг вверх

    def test_quiet_series_no_events(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 0.1, 200))
        events = cusum_events(s, threshold=5.0)
        assert len(events) == 0

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError):
            cusum_events(pd.Series([1.0, 2.0]), threshold=-1)


class TestPsi:
    def test_no_drift_low_psi(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(0, 1, 5000)
        assert population_stability_index(a, b) < 0.1

    def test_strong_drift_high_psi(self):
        rng = np.random.default_rng(2)
        train = rng.normal(0, 1, 5000)
        shifted = rng.normal(3, 1, 5000)      # сильно сдвинутое распределение
        assert population_stability_index(train, shifted) > 0.25

    def test_feature_drift_table(self):
        rng = np.random.default_rng(3)
        train = pd.DataFrame({"stable": rng.normal(0, 1, 2000),
                              "drifting": rng.normal(0, 1, 2000)})
        live = pd.DataFrame({"stable": rng.normal(0, 1, 2000),
                             "drifting": rng.normal(2.5, 1, 2000)})
        table = feature_drift_psi(train, live)
        assert table.iloc[0]["feature"] == "drifting"    # сильнее дрейф — выше
        assert bool(table[table["feature"] == "drifting"]["strong_drift"].iloc[0])
        assert not bool(table[table["feature"] == "stable"]["strong_drift"].iloc[0])


class TestKs:
    def test_same_distribution_no_drift(self):
        rng = np.random.default_rng(4)
        assert not ks_drift(rng.normal(0, 1, 1000), rng.normal(0, 1, 1000))["drift"]

    def test_different_distribution_drift(self):
        rng = np.random.default_rng(5)
        result = ks_drift(rng.normal(0, 1, 1000), rng.normal(1, 1, 1000))
        assert result["drift"]
        assert result["pvalue"] < 0.05
