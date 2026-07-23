"""Мета-фильтрованный моментум — ЧАСТЬ II (схема мета-разметки, раздел 18).

Первичная модель (правило): кросс-секционный моментум предлагает купить
top_n бумаг-лидеров. Вторичная модель (обученная логистическая регрессия)
для каждой предложенной бумаги оценивает вероятность, что сигнал сработает,
и отсеивает слабые. Отфильтрованные слоты остаются в кэше — под 14,25%
безрисковой ставкой это высокая планка, поэтому не входить в плохой сигнал
часто выгоднее, чем входить.

Ключевое требование мета-разметки: вторичной модели НЕ нужно предсказывать
направление рынка (задача, где почти все проигрывают) — только отфильтровать
заведомо плохие сигналы первичной модели.

Курс USD/RUB и индекс IMOEX передаются в срезе данных под служебными
ключами (не торгуются) — из них строятся межрыночные признаки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.ml.features import FEATURE_NAMES, bet_features, features_frame

USD_KEY = "_USDRUB"
IMOEX_KEY = "_IMOEX"
_SERVICE = {USD_KEY, IMOEX_KEY}


class MetaFilteredMomentum:
    """Моментум, сигналы которого фильтрует обученная мета-модель.

    ``meta_model`` — объект с predict_proba(X)->[:,1]; если None, фильтр
    выключен и стратегия эквивалентна чистому моментуму (для сравнения).
    ``threshold`` — минимальная вероятность срабатывания, чтобы взять сделку.
    """

    def __init__(self, lookback: int, top_n: int, meta_model=None,
                 threshold: float = 0.5):
        self.lookback = lookback
        self.top_n = top_n
        self.meta_model = meta_model
        self.threshold = threshold
        self.name = f"meta_momentum_{lookback}_{top_n}_{threshold}"
        self._last_month: tuple[int, int] | None = None
        self._weights: dict[str, float] = {}

    def params(self) -> dict:
        return {"lookback": self.lookback, "top_n": self.top_n,
                "threshold": self.threshold, "filtered": self.meta_model is not None}

    def target_weights(self, data_until_t: dict[str, pd.DataFrame]) -> dict[str, float]:
        stocks = {k: v for k, v in data_until_t.items() if k not in _SERVICE}
        if not stocks:
            return {}
        t = max(df["date"].iloc[-1] for df in stocks.values() if not df.empty)
        month = (t.year, t.month)
        if month == self._last_month:
            return self._weights
        self._last_month = month

        # Первичная модель: ранжирование по моментуму.
        returns = {}
        for secid, df in stocks.items():
            if df.empty or df["date"].iloc[-1] != t or len(df) <= self.lookback:
                continue
            past = float(df["close"].iloc[-1 - self.lookback])
            if past > 0:
                returns[secid] = float(df["close"].iloc[-1]) / past - 1
        candidates = sorted(returns, key=returns.get, reverse=True)[: self.top_n]
        if not candidates:
            self._weights = {}
            return self._weights

        weight = 1.0 / self.top_n     # фикс. доля; отсев уходит в кэш
        if self.meta_model is None:
            self._weights = {s: weight for s in candidates}
            return self._weights

        # Вторичная модель: фильтрация сигналов.
        usd = data_until_t.get(USD_KEY)
        imoex = data_until_t.get(IMOEX_KEY)
        rows, keys = [], []
        for secid in candidates:
            rows.append(bet_features(
                stocks[secid]["close"].reset_index(drop=True),
                usd["close"].reset_index(drop=True) if usd is not None else None,
                imoex["close"].reset_index(drop=True) if imoex is not None else None,
            ))
            keys.append(secid)
        X = features_frame(rows)
        keep = {}
        if not X.dropna().empty:
            valid = X.notna().all(axis=1)
            if valid.any():
                proba = np.full(len(X), np.nan)
                proba[valid.values] = self.meta_model.predict_proba(
                    X[valid].to_numpy()
                )[:, 1]
                for secid, p in zip(keys, proba):
                    if not np.isnan(p) and p >= self.threshold:
                        keep[secid] = weight
        self._weights = keep
        return self._weights

    def explain_ru(self) -> str:
        if self.meta_model is None:
            return (f"Моментум {self.lookback} дн., top-{self.top_n}, "
                    f"без мета-фильтра.")
        return (
            f"Моментум {self.lookback} дн., top-{self.top_n}: мета-модель "
            f"отсеивает сигналы с вероятностью успеха ниже {self.threshold:.0%}, "
            f"отсев — в кэш."
        )
