"""Рыночные профили — демонстрация переносимости системы на другие рынки.

Вся защита от самообмана (walk-forward, честные издержки, PBO,
дефлированный Шарп, тесты на утечку) рыночно-агностична. От рынка к рынку
меняются лишь ПАРАМЕТРЫ: безрисковая ставка, структура издержек, стоимость
инфраструктуры, лотность, торговый календарь. Здесь они собраны в профили,
чтобы посчитать, как меняется ключевая экономика — порог безубыточности
и стоимость оборота — при переносе тех же стратегий на другой рынок.

ВАЖНО: числа для США и крипты — оценки автора на 2025 год, не котировки
из API (данные этих рынков в текущем окружении недоступны). Их легко
поправить. Реальный бэктест возможен только после подключения источников
данных этих рынков.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketProfile:
    name: str
    currency: str
    risk_free_rate: float          # безрисковая альтернатива (фонд/облигации)
    capital: float                 # стартовый капитал в валюте рынка
    infra_cost_per_year: float     # инфраструктура (данные, VPS, комиссии за ведение)
    broker_commission_pct: float   # % за сторону сделки
    exchange_fee_pct: float        # % биржи за сторону
    half_spread_pct: float         # половина спреда ликвидной бумаги, %
    base_slippage_pct: float       # базовое проскальзывание, %
    has_lots: bool                 # торговля лотами (иначе — доли/штуки)
    calendar: str
    notes_ru: str

    def breakeven_rate(self) -> float:
        """Порог безубыточности = безрисковая ставка + инфраструктура/капитал.
        Та же формула, что в settings.py для Мосбиржи."""
        return self.risk_free_rate + self.infra_cost_per_year / self.capital

    def round_trip_cost_pct(self) -> float:
        """Издержки кругового оборота (купил+продал) в % от суммы, без
        учёта проскальзывания от размера (только базовое)."""
        one_side = (self.broker_commission_pct + self.exchange_fee_pct
                    + self.half_spread_pct + self.base_slippage_pct)
        return 2 * one_side


# Мосбиржа — как в боевом конфиге (settings.yaml).
MOEX = MarketProfile(
    name="Мосбиржа (акции)",
    currency="RUB",
    risk_free_rate=0.1425,
    capital=20_000,
    infra_cost_per_year=4_800,
    broker_commission_pct=0.05,
    exchange_fee_pct=0.01,
    half_spread_pct=0.05,
    base_slippage_pct=0.05,
    has_lots=True,
    calendar="T+1, дневные свечи, будни",
    notes_ru="Высокая ключевая ставка и большая доля инфраструктуры в "
             "капитале дают самый высокий порог безубыточности.",
)

# США — розничный брокер с нулевой комиссией (оценки на 2025).
US_STOCKS = MarketProfile(
    name="США (акции)",
    currency="USD",
    risk_free_rate=0.045,          # 3-мес. векселя казначейства ~4,5%
    capital=250,                   # ~эквивалент 20 000 ₽
    infra_cost_per_year=0.0,       # бесплатные брокер и котировки
    broker_commission_pct=0.0,     # zero-commission (Schwab, IBKR Lite)
    exchange_fee_pct=0.0,
    half_spread_pct=0.01,          # узкий спред крупных бумаг
    base_slippage_pct=0.02,
    has_lots=False,                # штучные (и дробные) акции — нет проблемы лота
    calendar="T+1, будни, часы биржи США",
    notes_ru="Низкая ставка и почти нулевые издержки → низкий порог. Но "
             "рынок эффективнее: простые ценовые правила давно арбитражены. "
             "Правило PDT ограничивает счёт < $25k тремя дневными сделками "
             "за 5 дней.",
)

# Криптовалюты — спотовая биржа (оценки на 2025).
CRYPTO = MarketProfile(
    name="Криптовалюты (спот)",
    currency="USDT",
    risk_free_rate=0.045,          # доходность стейблкоина/векселей ~4,5%
    capital=250,
    infra_cost_per_year=0.0,       # бесплатные публичные API
    broker_commission_pct=0.10,    # тейкер-комиссия ~0,1% за сторону
    exchange_fee_pct=0.0,
    half_spread_pct=0.02,          # BTC/ETH узко; альткоины кратно шире
    base_slippage_pct=0.05,
    has_lots=False,                # дробные объёмы — нет проблемы лота
    calendar="24/7, без выходных и аукционов",
    notes_ru="Низкий порог, но комиссия за сделку выше, ликвидность "
             "альткоинов тонкая, выживаемость выборки крайне плохая (тысячи "
             "мёртвых монет), а нестационарность экстремальна.",
)

ALL_PROFILES = [MOEX, US_STOCKS, CRYPTO]
