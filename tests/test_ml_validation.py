"""Тесты валидации без утечки: PurgedKFold, эмбарго, PBO, дефлированный Шарп.

Синтетика с заранее известным ответом — как требует ТЗ (раздел 25).
"""

import numpy as np
import pandas as pd
import pytest

from trading.ml.validation import (
    ForbiddenPracticeError,
    PurgedKFold,
    cscv_pbo,
    forbid_plain_kfold,
    forbid_shuffle_split,
)
from trading.reporting.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)


class TestForbiddenPractices:
    def test_shuffle_split_raises(self):
        with pytest.raises(ForbiddenPracticeError, match="утечка"):
            forbid_shuffle_split(shuffle=True)
        forbid_shuffle_split(shuffle=False)  # не бросает

    def test_plain_kfold_raises(self):
        with pytest.raises(ForbiddenPracticeError, match="PurgedKFold"):
            forbid_plain_kfold()


class TestPurgedKFold:
    def make_t1(self, n=100, horizon=5):
        # Наблюдение i начинается в день i, метка закрывается через horizon дней.
        idx = pd.RangeIndex(n)
        return pd.Series(idx + horizon, index=idx)

    def test_requires_t1(self):
        with pytest.raises(ValueError, match="t1 обязателен"):
            PurgedKFold(n_splits=5, t1=None)

    def test_all_test_indices_covered_once(self):
        t1 = self.make_t1()
        cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
        test_sets = [set(test) for _, test in cv.split()]
        union = set().union(*test_sets)
        assert union == set(range(100))
        # тестовые фолды не пересекаются
        assert sum(len(s) for s in test_sets) == 100

    def test_no_overlap_between_train_and_test(self):
        t1 = self.make_t1()
        cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
        for train, test in cv.split():
            assert set(train).isdisjoint(set(test))

    def test_purging_removes_overlapping_labels(self):
        """Наблюдения, чья метка перекрывает тест, исключаются из обучения."""
        t1 = self.make_t1(n=100, horizon=5)
        cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
        folds = list(cv.split())
        train, test = folds[2]                 # средний фолд: тест 40..59
        test_start, test_end = test.min(), test.max()
        for i in train:
            label_end = t1.iloc[i]
            overlaps = (i <= test_end) and (label_end >= test_start)
            assert not overlaps, f"наблюдение {i} с меткой до {label_end} перекрывает тест"

    def test_embargo_removes_buffer_after_test(self):
        t1 = self.make_t1(n=100, horizon=0)   # без горизонта — виден только эмбарго
        no_embargo = list(PurgedKFold(5, t1, embargo_pct=0.0).split())
        with_embargo = list(PurgedKFold(5, t1, embargo_pct=0.10).split())
        # На среднем фолде эмбарго уносит ~10 наблюдений справа.
        train_no = set(no_embargo[2][0])
        train_emb = set(with_embargo[2][0])
        assert len(train_no) > len(train_emb)


class TestCscvPbo:
    def test_random_strategies_give_high_pbo(self):
        """Чисто случайные стратегии: лучшая на обучении случайна на проверке,
        PBO должен быть около 0,5."""
        rng = np.random.default_rng(42)
        perf = pd.DataFrame(rng.normal(0, 1, size=(256, 10)))
        result = cscv_pbo(perf, n_partitions=8)
        assert 0.35 < result["pbo"] < 0.65

    def test_one_genuinely_better_strategy_gives_low_pbo(self):
        """Одна стратегия стабильно лучше остальных → низкий PBO."""
        rng = np.random.default_rng(0)
        perf = rng.normal(0, 1, size=(256, 10))
        perf[:, 0] += 1.5   # конфигурация 0 стабильно прибыльнее
        result = cscv_pbo(pd.DataFrame(perf), n_partitions=8)
        assert result["pbo"] < 0.1

    def test_odd_partitions_rejected(self):
        with pytest.raises(ValueError, match="чётным"):
            cscv_pbo(pd.DataFrame(np.zeros((10, 3))), n_partitions=7)


class TestDeflatedSharpe:
    def test_per_period_sharpe(self):
        r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.01])
        assert per_period_sharpe(r) == pytest.approx(
            r.mean() / r.std(ddof=1), rel=1e-9
        )

    def test_psr_high_for_strong_track_record(self):
        rng = np.random.default_rng(1)
        good = rng.normal(0.002, 0.01, size=1000)   # стабильно положительный
        assert probabilistic_sharpe_ratio(good, sr_star=0.0) > 0.95

    def test_psr_low_for_noise(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(0.0, 0.01, size=200)
        assert probabilistic_sharpe_ratio(noise, sr_star=0.05) < 0.5

    def test_expected_max_sharpe_grows_with_trials(self):
        s10 = expected_max_sharpe(0.1, 10)
        s500 = expected_max_sharpe(0.1, 500)
        assert s500 > s10 > 0

    def test_deflated_sharpe_penalizes_many_trials(self):
        """Тот же track record при 500 испытаниях менее значим, чем при 5.

        Разброс Шарпов держим одинаковым (тот же набор, повторённый), чтобы
        изолировать эффект ЧИСЛА испытаний."""
        rng = np.random.default_rng(3)
        best = rng.normal(0.0015, 0.01, size=1000)
        base = np.linspace(-0.1, 0.1, 5)      # разброс Шарпов, std фиксирован
        few = deflated_sharpe_ratio(best, base)                  # 5 испытаний
        many = deflated_sharpe_ratio(best, np.tile(base, 100))   # 500, тот же std
        assert many["expected_max_sharpe_null"] > few["expected_max_sharpe_null"]
        assert many["deflated_sharpe"] < few["deflated_sharpe"]

    def test_deflated_sharpe_verdict_text(self):
        rng = np.random.default_rng(4)
        noise = rng.normal(0.0, 0.01, size=300)
        result = deflated_sharpe_ratio(noise, list(rng.normal(0, 0.05, 300)))
        assert "n_trials" in result
        assert "гипотеза" in result["verdict_ru"].lower() or "случай" in result["verdict_ru"].lower()
