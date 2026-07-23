"""Единая шина событий — ФАЗА 4 (заглушка).

Один поток событий для backtest / paper / live: если стратегия ведёт
себя по-разному в разных режимах — это баг, а не особенность.
"""

from __future__ import annotations


class EventBus:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Шина событий — фаза 4. Модуль ещё не реализован.")
