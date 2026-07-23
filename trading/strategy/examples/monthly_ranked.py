"""Три референсные гипотезы для walk-forward валидации.

Источник гипотез: LLM (Claude) — так они и помечаются в журнале гипотез.
Каждая гипотеза объяснима одной строкой и имеет 2 оптимизируемых
параметра (лимит по ТЗ — 3).

Общий каркас: раз в календарный месяц пересчитать ранжирование бумаг
по доходности за lookback торговых дней и держать top_n бумаг равными
долями до следующего месяца. Ранжирование считается только по данным,
которые движок передал стратегии (срез по день T) — доступа к будущему
нет архитектурно.

Бумаги, не торгующиеся на дату решения (пауза, делистинг), из
кандидатов исключаются: купить их всё равно нельзя.
"""

from __future__ import annotations

import pandas as pd


class _MonthlyRanked:
    """Каркас: месячная ребалансировка по ранжированию доходностей.

    ``direction`` = +1 — лонг (покупаем выбранных), −1 — шорт (продаём
    выбранных без покрытия; движок должен быть с allow_short=True)."""

    direction = +1

    def __init__(self, lookback: int, top_n: int, rebalance: str = "month"):
        if lookback < 2:
            raise ValueError("lookback должен быть не меньше 2 баров")
        if top_n < 1:
            raise ValueError("top_n должен быть не меньше 1")
        if rebalance not in ("month", "bar"):
            raise ValueError("rebalance: 'month' (раз в календарный месяц) "
                             "или 'bar' (каждый бар — для старших таймфреймов)")
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance = rebalance
        self._last_month: tuple[int, int] | None = None
        self._weights: dict[str, float] = {}

    def params(self) -> dict:
        return {"lookback": self.lookback, "top_n": self.top_n}

    def _select(self, returns: dict[str, float]) -> list[str]:
        raise NotImplementedError

    def target_weights(self, data_until_t: dict[str, pd.DataFrame]) -> dict[str, float]:
        if not data_until_t:
            return {}
        t = max(df["date"].iloc[-1] for df in data_until_t.values())
        # На старших таймфреймах (bar) ребалансируем каждый бар: бар и есть
        # шаг таймфрейма. На дневных данных (month) — раз в календарный месяц.
        if self.rebalance == "month":
            month = (t.year, t.month)
            if month == self._last_month:
                return self._weights
            self._last_month = month

        returns: dict[str, float] = {}
        for secid, df in data_until_t.items():
            if df["date"].iloc[-1] != t:
                continue  # бумага сейчас не торгуется — не кандидат
            if len(df) <= self.lookback:
                continue  # истории не хватает для оценки
            past = float(df["close"].iloc[-1 - self.lookback])
            if past > 0:
                returns[secid] = float(df["close"].iloc[-1]) / past - 1

        selected = self._select(returns)
        weight = self.direction * (1.0 / self.top_n) if selected else 0.0
        self._weights = {secid: weight for secid in selected}
        return self._weights


class CrossSectionalMomentum(_MonthlyRanked):
    """Гипотеза 1: лидеры роста продолжают расти (относительный моментум)."""

    def __init__(self, lookback: int, top_n: int):
        super().__init__(lookback, top_n)
        self.name = f"momentum_{lookback}_{top_n}"

    def _select(self, returns: dict[str, float]) -> list[str]:
        return sorted(returns, key=returns.get, reverse=True)[: self.top_n]

    def explain_ru(self) -> str:
        return (
            f"Раз в месяц держать {self.top_n} бумаг с наибольшим ростом за "
            f"{self.lookback} торговых дней, равными долями."
        )


class AbsoluteMomentum(_MonthlyRanked):
    """Гипотеза 2: моментум с фильтром — падающие лидеры не покупаются.

    Как гипотеза 1, но в портфель попадают только бумаги с положительной
    доходностью за период. Если таких нет — весь портфель в кэше
    (а кэш у нас под 14,25% безрисковой альтернативой — высокая планка)."""

    def __init__(self, lookback: int, top_n: int):
        super().__init__(lookback, top_n)
        self.name = f"abs_momentum_{lookback}_{top_n}"

    def _select(self, returns: dict[str, float]) -> list[str]:
        ranked = sorted(returns, key=returns.get, reverse=True)
        return [s for s in ranked if returns[s] > 0][: self.top_n]

    def explain_ru(self) -> str:
        return (
            f"Раз в месяц держать до {self.top_n} бумаг с наибольшим ростом за "
            f"{self.lookback} дней, но только растущие; нет растущих — сидеть в кэше."
        )


class MeanReversion(_MonthlyRanked):
    """Гипотеза 3: перепроданные бумаги отскакивают (возврат к среднему)."""

    def __init__(self, lookback: int, top_n: int):
        super().__init__(lookback, top_n)
        self.name = f"mean_reversion_{lookback}_{top_n}"

    def _select(self, returns: dict[str, float]) -> list[str]:
        return sorted(returns, key=returns.get)[: self.top_n]

    def explain_ru(self) -> str:
        return (
            f"Раз в месяц покупать {self.top_n} бумаг с наибольшим падением за "
            f"{self.lookback} торговых дней — ставка на отскок."
        )


class ShortMomentum(_MonthlyRanked):
    """Шорт-гипотеза 1: падающие продолжают падать — шортить худших.

    В шорт попадают только бумаги с ОТРИЦАТЕЛЬНОЙ доходностью за период:
    шортить растущую бумагу в надежде на разворот — другая гипотеза
    (см. ShortOverbought). Нет падающих — портфель в кэше."""

    direction = -1

    def __init__(self, lookback: int, top_n: int):
        super().__init__(lookback, top_n)
        self.name = f"short_momentum_{lookback}_{top_n}"

    def _select(self, returns: dict[str, float]) -> list[str]:
        ranked = sorted(returns, key=returns.get)   # худшие первыми
        return [s for s in ranked if returns[s] < 0][: self.top_n]

    def explain_ru(self) -> str:
        return (
            f"Раз в месяц шортить {self.top_n} бумаг с наибольшим падением за "
            f"{self.lookback} торговых дней — ставка на продолжение падения."
        )


class ShortOverbought(_MonthlyRanked):
    """Шорт-гипотеза 2: перегретые лидеры роста откатываются — шортить их."""

    direction = -1

    def __init__(self, lookback: int, top_n: int):
        super().__init__(lookback, top_n)
        self.name = f"short_overbought_{lookback}_{top_n}"

    def _select(self, returns: dict[str, float]) -> list[str]:
        ranked = sorted(returns, key=returns.get, reverse=True)  # лидеры первыми
        return [s for s in ranked if returns[s] > 0][: self.top_n]

    def explain_ru(self) -> str:
        return (
            f"Раз в месяц шортить {self.top_n} бумаг с наибольшим ростом за "
            f"{self.lookback} торговых дней — ставка на откат перегретых."
        )
