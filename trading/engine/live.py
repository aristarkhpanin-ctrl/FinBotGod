"""Боевой режим — ФАЗА 9 (заглушка).

Включается только после 3 месяцев бумажной торговли без необъяснимых
расхождений с бэктестом. Та же логика, что в backtest и paper, —
различается только адаптер исполнения.
"""

from __future__ import annotations


class LiveEngine:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Боевой режим — фаза 9. Модуль ещё не реализован.")
