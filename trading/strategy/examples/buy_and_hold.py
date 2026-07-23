"""Референсная стратегия «купил и держи» — для теста известного ответа.

Это не торговая идея, а эталон для проверки движка (ТЗ, раздел 7):
результат такой стратегии на одной бумаге обязан до копейки совпадать
с ручным расчётом.
"""

from __future__ import annotations

import pandas as pd


class BuyAndHold:
    def __init__(self, secid: str, weight: float = 0.25):
        self.name = f"buy_and_hold_{secid}"
        self.secid = secid
        self.weight = weight

    def params(self) -> dict:
        return {"secid": self.secid, "weight": self.weight}

    def target_weights(self, data_until_t: dict[str, pd.DataFrame]) -> dict[str, float]:
        if self.secid not in data_until_t:
            return {}
        return {self.secid: self.weight}

    def explain_ru(self) -> str:
        return (
            f"Купить {self.secid} на {self.weight:.0%} портфеля в первый "
            f"день и держать до конца."
        )
