"""НЕЗАВИСИМЫЙ риск-слой — ФАЗА 6 (заглушка).

Не зависит от логики стратегии, работает даже при грубых ошибках в ней.
Предохранители (ТЗ, раздел 9): max_position_pct, max_positions, whitelist,
max_order_value (остановка системы), max_orders_per_day (остановка),
daily_loss_limit, max_drawdown_from_peak. Последние два — по фактической
стоимости портфеля от брокера, не по внутреннему расчёту.
"""

from __future__ import annotations


class RiskGuards:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Риск-слой — фаза 6. Модуль ещё не реализован.")
