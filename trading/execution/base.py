"""Абстракция исполнения — единый интерфейс для трёх режимов.

backtest, paper и live используют одну и ту же логику стратегии, сайзинга
и риска. Различается ТОЛЬКО адаптер исполнения. Контракт закладывается
сейчас, реализации — в фазах 4 (simulated) и 9 (alor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Order:
    client_order_id: str          # детерминирован: дата + тикер + намерение
    secid: str
    side: Literal["buy", "sell"]
    shares: int
    limit_price: float            # только лимитные заявки, рыночные запрещены


class ExecutionAdapter(Protocol):
    def submit(self, order: Order) -> None: ...
    def cancel_all(self) -> None: ...
    def positions(self) -> dict[str, int]: ...
    def cash(self) -> float: ...
