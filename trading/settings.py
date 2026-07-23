"""Загрузка и валидация конфигурации.

Файл ``trading/config/settings.yaml`` — панель управления заказчика.
Ключи в YAML — на русском языке. Здесь каждая секция описана моделью
pydantic: опечатка, лишний ключ или значение вне допустимого диапазона
приводят к отказу запуска с понятным сообщением на русском, а не к
падению со стектрейсом.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_SETTINGS_PATH = Path(__file__).parent / "config" / "settings.yaml"
DEFAULT_UNIVERSE_PATH = Path(__file__).parent / "config" / "universe.yaml"


class ConfigError(Exception):
    """Ошибка конфигурации с человекочитаемым объяснением."""


class _Section(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CapitalConfig(_Section):
    start_amount: float = Field(alias="стартовая_сумма", gt=0)
    currency: str = Field(alias="валюта", default="RUB")


class RiskConfig(_Section):
    max_position_pct: float = Field(alias="макс_доля_одной_позиции", gt=0, le=1)
    max_positions: int = Field(alias="макс_число_позиций", ge=1)
    daily_loss_limit: float = Field(alias="дневной_лимит_убытка", gt=0, le=1)
    max_drawdown_from_peak: float = Field(alias="макс_просадка_от_пика", gt=0, le=1)
    max_order_value: float = Field(alias="макс_сумма_одной_заявки", gt=0)
    max_orders_per_day: int = Field(alias="макс_заявок_в_день", ge=1)


class CostConfig(_Section):
    broker_commission_pct: float = Field(alias="комиссия_брокера_пct", ge=0)
    broker_commission_min: float = Field(alias="минимальная_комиссия_руб", ge=0)
    exchange_fee_pct: float = Field(alias="комиссия_биржи_пct", ge=0)
    half_spread_pct: float = Field(alias="половина_спреда_пct", ge=0)
    base_slippage_pct: float = Field(alias="базовое_проскальзывание_пct", ge=0)
    impact_coefficient: float = Field(alias="коэффициент_влияния", ge=0)
    adv_warning_share: float = Field(alias="предупреждение_доля_adv", gt=0, le=1)
    adv_reject_share: float = Field(alias="отказ_доля_adv", gt=0, le=1)
    stress_multiplier: float = Field(alias="стресс_множитель", ge=1)
    ndfl_rate: float = Field(alias="ндфл_ставка", ge=0, le=1)


class BenchmarkConfig(_Section):
    risk_free_rate: float = Field(alias="безрисковая_ставка", ge=0, le=1)
    infra_cost_rub_year: float = Field(alias="инфраструктура_руб_год", ge=0)


class DataConfig(_Section):
    cache_dir: str = Field(alias="каталог_кэша")
    history_start: str = Field(alias="начало_истории")
    requests_per_second: float = Field(alias="запросов_в_секунду", gt=0, le=3)
    timeout_seconds: float = Field(alias="таймаут_секунд", gt=0)
    retries: int = Field(alias="число_ретраев", ge=0)
    price_jump_threshold: float = Field(alias="порог_скачка_цены", gt=0)


class MlConfig(_Section):
    enabled: bool = False


class Settings(_Section):
    capital: CapitalConfig = Field(alias="капитал")
    mode: Literal["backtest", "paper", "live"] = Field(alias="режим")
    risk: RiskConfig = Field(alias="риск")
    costs: CostConfig = Field(alias="издержки")
    benchmark: BenchmarkConfig = Field(alias="бенчмарк")
    data: DataConfig = Field(alias="данные")
    ml: MlConfig = Field(alias="ml", default_factory=MlConfig)

    def breakeven_rate(self) -> float:
        """Порог безубыточности: безрисковая ставка + инфраструктура к капиталу.

        Это та самая цифра ~38% годовых из ТЗ. Она обязана печататься
        в каждом отчёте.
        """
        infra_share = self.benchmark.infra_cost_rub_year / self.capital.start_amount
        return self.benchmark.risk_free_rate + infra_share


def _explain_validation_error(err: ValidationError) -> str:
    lines = ["Конфигурация не прошла проверку. Исправьте settings.yaml:"]
    for e in err.errors():
        where = " → ".join(str(p) for p in e["loc"])
        lines.append(f"  • параметр «{where}»: {e['msg']} (получено: {e.get('input')!r})")
    return "\n".join(lines)


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Читает settings.yaml. Ошибка → ConfigError с объяснением на русском."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Файл {path} не является корректным YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"Файл {path} пуст или имеет неверную структуру.")
    try:
        return Settings.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_explain_validation_error(e)) from e


def load_universe(path: str | Path = DEFAULT_UNIVERSE_PATH) -> list[str]:
    """Читает белый список тикеров из universe.yaml."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Файл универсума не найден: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tickers = raw.get("тикеры") if isinstance(raw, dict) else None
    if not tickers or not isinstance(tickers, list):
        raise ConfigError(f"В файле {path} нет списка «тикеры».")
    cleaned = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if len(cleaned) != len(set(cleaned)):
        dupes = sorted({t for t in cleaned if cleaned.count(t) > 1})
        raise ConfigError(f"В универсуме дублируются тикеры: {', '.join(dupes)}")
    return cleaned
