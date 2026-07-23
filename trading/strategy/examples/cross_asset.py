"""Межрыночная гипотеза: курс доллара как ведущий сигнал для экспортёров.

Идея заказчика: движение одного актива предсказывает другой с задержкой
(«упала нефть → через месяц упали акции нефтяников»). Прямого чистого ряда
Brent на Мосбирже нет (только фьючерсы, требующие сшивки), поэтому берём
самый сильный доступный межрыночный сигнал российского рынка — курс
USD/RUB. Экономика: нефтяники и металлурги продают за валюту, а отчитываются
в рублях. Слабеющий рубль (USD растёт) поднимает их рублёвую выручку, и
с некоторым лагом — котировки.

Сигнал строится ТОЛЬКО по данным до дня T (курс на закрытии T известен),
исполнение — по открытию T+1. Утечки будущего нет.

Ряд курса передаётся стратегии внутри среза данных под ключом SIGNAL_KEY;
торгуются только реальные бумаги-экспортёры, сам курс не торгуется.
"""

from __future__ import annotations

import pandas as pd

SIGNAL_KEY = "_USDRUB"

# Экспортёры: выручка в валюте, отчётность в рублях. Нефть/газ и металлурги/
# горнодобыча/химия — то, на что слабый рубль влияет напрямую.
EXPORTERS = [
    "LKOH", "ROSN", "SNGS", "SNGSP", "TATN", "TATNP", "GAZP", "NVTK", "TRNFP",
    "GMKN", "NLMK", "MAGN", "CHMF", "PLZL", "ALRS", "RUAL", "PHOR",
]


class CrossAssetExporters:
    """Держать корзину экспортёров, когда рубль слабел, иначе — в кэш.

    Параметры (2, лимит ТЗ — 3):
    * ``lookback`` — за сколько торговых дней мерить движение курса;
    * ``lag`` — сдвиг: сигнал берётся не по последний день, а с задержкой
      lag дней. Так проверяется гипотеза «курс двинулся раньше, акции
      реагируют позже». lag=0 — одновременно.
    """

    direction = +1   # +1 лонг экспортёров; переопределяется в шорт-версии

    def __init__(self, lookback: int, lag: int = 0, basket: list[str] | None = None):
        if lookback < 2:
            raise ValueError("lookback должен быть не меньше 2 дней")
        if lag < 0:
            raise ValueError("lag не может быть отрицательным")
        self.lookback = lookback
        self.lag = lag
        self.basket = basket or EXPORTERS
        self.name = f"cross_usd_exporters_{lookback}_{lag}"
        self._last_month: tuple[int, int] | None = None
        self._weights: dict[str, float] = {}

    def params(self) -> dict:
        return {"lookback": self.lookback, "lag": self.lag}

    def _fx_signal(self, fx: pd.DataFrame) -> float | None:
        """Доходность курса за окно lookback, сдвинутое на lag назад."""
        closes = fx["close"].to_numpy()
        need = self.lookback + self.lag + 1
        if len(closes) < need:
            return None
        end = closes[-1 - self.lag]
        start = closes[-1 - self.lag - self.lookback]
        if start <= 0:
            return None
        return end / start - 1

    def _decide(self, fx_return: float) -> bool:
        """Открывать ли корзину экспортёров. Лонг: да, если рубль слабел."""
        return fx_return > 0

    def target_weights(self, data_until_t: dict[str, pd.DataFrame]) -> dict[str, float]:
        fx = data_until_t.get(SIGNAL_KEY)
        if fx is None or fx.empty:
            return {}
        t = max(
            df["date"].iloc[-1] for k, df in data_until_t.items()
            if k != SIGNAL_KEY and not df.empty
        )
        month = (t.year, t.month)
        if month == self._last_month:
            return self._weights
        self._last_month = month

        signal = self._fx_signal(fx)
        if signal is None or not self._decide(signal):
            self._weights = {}
            return self._weights

        # Корзина — только те экспортёры, что реально торгуются на дату T.
        tradeable = [
            s for s in self.basket
            if s in data_until_t and not data_until_t[s].empty
            and data_until_t[s]["date"].iloc[-1] == t
        ]
        if not tradeable:
            self._weights = {}
            return self._weights
        weight = self.direction * (1.0 / len(tradeable))
        self._weights = {s: weight for s in tradeable}
        return self._weights

    def explain_ru(self) -> str:
        move = "слабел" if self.direction > 0 else "укреплялся"
        act = "покупать" if self.direction > 0 else "шортить"
        lag_txt = f" (со сдвигом {self.lag} дн.)" if self.lag else ""
        return (
            f"Если рубль {move} за последние {self.lookback} дней{lag_txt} — "
            f"{act} корзину экспортёров равными долями, иначе быть в кэше."
        )


class CrossAssetExportersShort(CrossAssetExporters):
    """Зеркало: шортить экспортёров, когда рубль УКРЕПЛЯЛСЯ (USD падал)."""

    direction = -1

    def __init__(self, lookback: int, lag: int = 0, basket: list[str] | None = None):
        super().__init__(lookback, lag, basket)
        self.name = f"cross_usd_exporters_short_{lookback}_{lag}"

    def _decide(self, fx_return: float) -> bool:
        return fx_return < 0   # рубль укреплялся — давим экспортёров в шорт
