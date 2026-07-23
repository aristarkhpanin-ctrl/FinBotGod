"""Тесты модельного слоя: логистический мета-фильтр, explain, MDA."""

import numpy as np
import pandas as pd
import pytest

from trading.ml.feature_importance import mda_importance
from trading.ml.models.linear import LogisticMetaModel
from trading.ml.validation import PurgedKFold


def separable_data(n=400, seed=0):
    """Признак f_good разделяет классы, f_noise — чистый шум."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    f_good = y + rng.normal(0, 0.5, n)      # несёт сигнал
    f_noise = rng.normal(0, 1, n)           # шум
    X = pd.DataFrame({"f_good": f_good, "f_noise": f_noise})
    return X, pd.Series(y)


class TestLogisticMetaModel:
    def test_learns_separable_signal(self):
        X, y = separable_data()
        m = LogisticMetaModel(list(X.columns))
        m.fit(X.to_numpy(), y.to_numpy())
        proba = m.predict_proba(X.to_numpy())[:, 1]
        acc = ((proba > 0.5).astype(int) == y.to_numpy()).mean()
        assert acc > 0.8

    def test_predict_before_fit_raises(self):
        m = LogisticMetaModel(["a", "b"])
        with pytest.raises(RuntimeError, match="не обучена"):
            m.predict_proba(np.zeros((1, 2)))

    def test_importance_ranks_signal_above_noise(self):
        X, y = separable_data()
        m = LogisticMetaModel(list(X.columns))
        m.fit(X.to_numpy(), y.to_numpy())
        imp = m.feature_importance()
        assert imp["f_good"] > imp["f_noise"]
        assert abs(sum(imp.values()) - 1.0) < 1e-9

    def test_explain_is_russian_and_actionable(self):
        X, y = separable_data()
        m = LogisticMetaModel(list(X.columns))
        m.fit(X.to_numpy(), y.to_numpy())
        text = m.explain(X.to_numpy()[0])
        assert "Вероятность" in text
        assert "f_good" in text
        assert ("действовать" in text) or ("пропустить" in text)

    def test_sample_weight_accepted(self):
        X, y = separable_data()
        m = LogisticMetaModel(list(X.columns))
        w = np.ones(len(y))
        m.fit(X.to_numpy(), y.to_numpy(), sample_weight=w)   # не падает


class TestMdaImportance:
    def test_mda_finds_useful_feature(self):
        X, y = separable_data(n=600)
        t1 = pd.Series(y.index + 1, index=y.index)   # непересекающиеся метки
        cv = PurgedKFold(n_splits=4, t1=t1, embargo_pct=0.0)
        result = mda_importance(
            lambda: LogisticMetaModel(list(X.columns)), X, y, cv
        )
        top = result.iloc[0]
        assert top["feature"] == "f_good"
        assert top["mda_mean"] > 0          # ломать полезный признак вредит
        noise_row = result[result["feature"] == "f_noise"].iloc[0]
        assert top["mda_mean"] > noise_row["mda_mean"]
