"""ML-архитектура с приоритетом сохранения капитала.

Принципы (постановка заказчика):
1. Направление предсказывает обученная ML-модель, а не правило.
2. Актив покупается, ТОЛЬКО если калиброванная уверенность модели в его
   росте не ниже порога (по умолчанию 80%).
3. Быть в рынке не обязательно: нет уверенных активов — сидим в кэше.
4. Главное — не потерять, а не заработать.
5. Свободный кэш допустим и является нормой, а не недостатком.

Модель — единая для всех активов: на входе point-in-time признаки актива,
на выходе вероятность того, что длинная позиция закроется в плюс (по
разметке тройным барьером). Стратегия отбирает активы с вероятностью выше
порога, берёт до max_positions самых уверенных равными долями, остальное —
кэш. Модель передаётся обученной (обучение — строго на прошлом, вне теста).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.ml.features import FEATURE_NAMES, bet_features, features_frame

USD_KEY = "_USDRUB"
IMOEX_KEY = "_IMOEX"
_SERVICE = {USD_KEY, IMOEX_KEY}


class MLConfidenceLong:
    """Лонг только при уверенности модели ≥ threshold, иначе кэш.

    ``model`` — обученная модель с predict_proba(X)->[:,1] (вероятность
    роста). ``threshold`` — минимальная уверенность (0.80). ``max_positions``
    — сколько самых уверенных активов держать; доля каждого = 1/max_positions,
    поэтому при нехватке уверенных активов часть портфеля остаётся в кэше.
    ``rebalance`` — 'month' или 'bar'.
    """

    def __init__(self, model, threshold: float = 0.80, max_positions: int = 6,
                 rebalance: str = "month"):
        self.model = model
        self.threshold = threshold
        self.max_positions = max_positions
        self.rebalance = rebalance
        self.name = f"ml_confidence_{int(threshold*100)}"
        self._last_month: tuple[int, int] | None = None
        self._weights: dict[str, float] = {}
        self.last_cash_share = 1.0     # для отчётности: доля кэша в решении

    def params(self) -> dict:
        return {"threshold": self.threshold, "max_positions": self.max_positions,
                "model": type(self.model).__name__ if self.model else None}

    def target_weights(self, data_until_t: dict[str, pd.DataFrame]) -> dict[str, float]:
        stocks = {k: v for k, v in data_until_t.items() if k not in _SERVICE}
        if not stocks or self.model is None:
            return {}
        t = max(df["date"].iloc[-1] for df in stocks.values() if not df.empty)
        if self.rebalance == "month":
            month = (t.year, t.month)
            if month == self._last_month:
                return self._weights
            self._last_month = month

        usd = data_until_t.get(USD_KEY)
        imoex = data_until_t.get(IMOEX_KEY)
        rows, keys = [], []
        for secid, df in stocks.items():
            if df.empty or df["date"].iloc[-1] != t or len(df) < 130:
                continue
            rows.append(bet_features(
                df["close"].reset_index(drop=True),
                usd["close"].reset_index(drop=True) if usd is not None else None,
                imoex["close"].reset_index(drop=True) if imoex is not None else None,
            ))
            keys.append(secid)
        if not rows:
            self._weights = {}
            self.last_cash_share = 1.0
            return self._weights

        X = features_frame(rows)
        valid = X.notna().all(axis=1).to_numpy()
        confidences = {}
        if valid.any():
            proba = self.model.predict_proba(X[valid].to_numpy())[:, 1]
            valid_keys = [k for k, v in zip(keys, valid) if v]
            for secid, p in zip(valid_keys, proba):
                if p >= self.threshold:
                    confidences[secid] = float(p)

        # Берём до max_positions самых уверенных; доля каждого = 1/max_positions.
        chosen = sorted(confidences, key=confidences.get, reverse=True)[: self.max_positions]
        weight = 1.0 / self.max_positions
        self._weights = {secid: weight for secid in chosen}
        self.last_cash_share = 1.0 - weight * len(chosen)
        return self._weights

    def explain_ru(self) -> str:
        return (
            f"Покупать только активы, в росте которых модель уверена ≥ "
            f"{self.threshold:.0%}; держать до {self.max_positions} самых "
            f"уверенных равными долями, остальное — в кэше. Нет уверенных — "
            f"весь портфель в кэше."
        )
