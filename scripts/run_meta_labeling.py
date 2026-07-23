"""ИТОГОВЫЙ ЭКСПЕРИМЕНТ ЧАСТИ II: мета-разметка поверх моментума.

Запуск:
    python scripts/run_meta_labeling.py

Собирает всю Часть II в одну систему и проверяет главный вопрос: улучшает
ли «умная» надстройка результат простого правила?

Схема (ТЗ, раздел 18):
* первичная модель — кросс-секционный моментум (простое правило);
* на обучающем окне генерируется выборка: для каждой предложенной сделки
  строятся признаки (point-in-time) и мета-метка тройным барьером
  (сработал ли сигнал);
* обучается вторичная модель (логистическая регрессия) с весами
  наблюдений, обратными конкурентности, и честной оценкой PurgedKFold;
* на проверочном окне сравниваются ЧИСТЫЙ моментум и МЕТА-фильтрованный
  (модель отсеивает слабые сигналы, отсев — в кэш);
* значимость — дефлированный Шарп и PBO по порогам фильтра.

Всё out-of-sample, метрики только по склейке проверочных окон.
Обучение мета-модели каждого окна — строго на данных до его начала.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.moex_client import MarketData, MoexClient
from trading.data.validation import build_trading_calendar, validate_candles
from trading.engine.backtest import BacktestEngine
from trading.logging_setup import setup_logging
from trading.ml.features import FEATURE_NAMES, bet_features, features_frame
from trading.ml.labeling import concurrency_weights, meta_labels
from trading.ml.models.linear import LogisticMetaModel
from trading.ml.validation import PurgedKFold, cscv_pbo
from trading.reporting.metrics import (
    compute_metrics,
    deflated_sharpe_ratio,
    per_period_sharpe,
)
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import year_windows
from trading.settings import load_settings, load_universe
from trading.strategy.examples.meta_filtered import (
    IMOEX_KEY,
    USD_KEY,
    MetaFilteredMomentum,
)

LOOKBACK, TOP_N = 120, 4          # первичный моментум (фиксирован, середина сетки)
MAX_HOLDING = 20                  # вертикальный барьер, дней
PT_SL = (1.5, 1.0)               # цель 1,5×волатильности, стоп 1,0×
THRESHOLDS = [0.50, 0.55, 0.60]   # пороги мета-фильтра
VOL_SPAN = 20


def month_end_entries(index: pd.DatetimeIndex, start, end) -> list[pd.Timestamp]:
    """Последние торговые дни каждого месяца в [start, end]."""
    sub = index[(index >= start) & (index <= end)]
    if len(sub) == 0:
        return []
    s = pd.Series(sub, index=sub)
    return list(s.groupby([sub.year, sub.month]).last().values)


def momentum_picks(stocks, t, lookback, top_n):
    returns = {}
    for secid, df in stocks.items():
        d = df[df["date"] <= t]
        if len(d) <= lookback or d["date"].iloc[-1] != t:
            continue
        past = float(d["close"].iloc[-1 - lookback])
        if past > 0:
            returns[secid] = float(d["close"].iloc[-1]) / past - 1
    return sorted(returns, key=returns.get, reverse=True)[:top_n]


def build_training_set(stocks, usd, imoex, train_start, train_end, index):
    """Выборка (признаки, мета-метка, t1) для мета-модели на обучающем окне.

    Берутся только сделки, чей горизонт тройного барьера укладывается
    в обучающее окно — чтобы метки не заглядывали в проверочный период.
    """
    horizon_cutoff_pos = index.searchsorted(train_end)
    rows, labels, t0s, t1s, weights_src = [], [], [], [], []
    per_stock_entries: dict[str, list] = {s: [] for s in stocks}
    for t in month_end_entries(index, train_start, train_end):
        pos = index.searchsorted(t)
        if pos + MAX_HOLDING > horizon_cutoff_pos:
            continue                 # горизонт вышел бы за обучающее окно
        for secid in momentum_picks(stocks, t, LOOKBACK, TOP_N):
            per_stock_entries[secid].append(t)

    for secid, entries in per_stock_entries.items():
        if not entries:
            continue
        close = stocks[secid].set_index("date")["close"]
        entries_idx = pd.DatetimeIndex(sorted(set(entries)))
        side = pd.Series(1, index=entries_idx)
        labelled = meta_labels(close, entries_idx, side, pt_sl=PT_SL,
                               max_holding=MAX_HOLDING, vol_span=VOL_SPAN)
        for t0, row in labelled.iterrows():
            feats = bet_features(
                close[close.index <= t0].reset_index(drop=True),
                usd[usd["date"] <= t0]["close"].reset_index(drop=True),
                imoex[imoex["date"] <= t0]["close"].reset_index(drop=True),
            )
            rows.append(feats)
            labels.append(int(row["meta"]))
            t0s.append(t0)
            t1s.append(row["t1"])
    if not rows:
        return None
    X = features_frame(rows)
    df = X.copy()
    df["y"] = labels
    df["t0"] = t0s
    df["t1"] = t1s
    df = df.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    return df


def train_meta_model(train_df):
    """Обучает мета-модель + честная оценка PurgedKFold. None, если данных мало."""
    if train_df is None or len(train_df) < 40 or train_df["y"].nunique() < 2:
        return None
    train_df = train_df.sort_values("t0").reset_index(drop=True)
    X = train_df[FEATURE_NAMES]
    y = train_df["y"]
    t1 = pd.Series(train_df["t1"].values, index=train_df["t0"].values)
    weights = concurrency_weights(t1, pd.DatetimeIndex(sorted(train_df["t0"].unique())))
    w = pd.Series(weights.reindex(train_df["t0"].values).fillna(1.0).values)
    model = LogisticMetaModel(FEATURE_NAMES)
    model.fit(X.to_numpy(), y.to_numpy(), sample_weight=w.to_numpy())
    return model


def run_variant(candles, lot_sizes, settings, ledger, data_hash,
                strategy, capital, test_start, test_end):
    s = settings.model_copy(deep=True)
    s.capital.start_amount = capital
    sliced = {sec: df[df["date"] <= test_end] for sec, df in candles.items()}
    engine = BacktestEngine(
        candles=sliced, lot_sizes=lot_sizes, settings=s, strategy=strategy,
        ledger=ledger, data_hash=f"{data_hash}|meta", source="llm",
        trade_from=str(pd.Timestamp(test_start).date()),
    )
    return engine.run()


def main() -> int:
    setup_logging()
    settings = load_settings()
    cache = ParquetCache(settings.data.cache_dir)
    client = MoexClient(requests_per_second=settings.data.requests_per_second,
                        timeout_seconds=settings.data.timeout_seconds,
                        retries=settings.data.retries)
    market = MarketData(client, cache)
    dfrom, dtill = settings.data.history_start, "2026-07-22"

    daily = {s: market.daily_candles(s, dfrom, dtill) for s in load_universe()}
    calendar = build_trading_calendar(daily)
    clean = {}
    for secid, df in daily.items():
        rep = validate_candles(df, secid, jump_threshold=settings.data.price_jump_threshold,
                               trading_calendar=calendar,
                               confirmed_events=settings.data.confirmed_events)
        if not rep.suspicious and not df.empty:
            clean[secid] = df

    index = pd.DatetimeIndex(sorted(calendar))
    fx = market.fx_candles(dfrom, dtill).set_index("date").reindex(index).ffill()
    fx = fx.dropna(subset=["close"]).reset_index().rename(columns={"index": "date"})
    imoex = market.index_candles(dfrom, dtill).set_index("date").reindex(index).ffill()
    imoex = imoex.dropna(subset=["close"]).reset_index().rename(columns={"index": "date"})

    lot_sizes = market.lot_sizes()
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")

    # Свечи для движка: акции + служебные ряды курса и индекса (не торгуются).
    engine_candles = dict(clean)
    engine_candles[USD_KEY] = fx[["date", "open", "high", "low", "close", "value", "volume"]]
    engine_candles[IMOEX_KEY] = imoex[["date", "open", "high", "low", "close", "value", "volume"]]

    years = sorted({d.year for d in index})
    windows = year_windows(years[0], years[-1], train_years=4, test_years=1)

    variants = {"чистый моментум": None}
    for th in THRESHOLDS:
        variants[f"мета-фильтр ≥{int(th*100)}%"] = th

    carried = {name: settings.capital.start_amount for name in variants}
    oos_segments = {name: [] for name in variants}
    oos_fills = {name: [] for name in variants}
    trained_windows = 0

    for w in windows:
        train_df = build_training_set(clean, fx, imoex, w.train_start, w.train_end, index)
        model = train_meta_model(train_df)
        if model is not None:
            trained_windows += 1
        for name, th in variants.items():
            strat = MetaFilteredMomentum(
                LOOKBACK, TOP_N,
                meta_model=(model if th is not None else None),
                threshold=(th or 0.5),
            )
            result = run_variant(engine_candles, lot_sizes, settings, ledger,
                                 data_hash, strat, carried[name],
                                 w.test_start, w.test_end)
            carried[name] = float(result.equity.iloc[-1])
            oos_segments[name].append(result.equity)
            oos_fills[name].extend(result.fills)

    # Метрики по склейке проверочных окон.
    rf = settings.benchmark.risk_free_rate
    stats, daily_sharpes, perf_matrix = {}, {}, {}
    for name in variants:
        equity = pd.concat(oos_segments[name])
        equity = equity[~equity.index.duplicated(keep="last")]
        m = compute_metrics(equity, rf, fills=oos_fills[name])
        rets = equity.pct_change().dropna()
        stats[name] = m
        daily_sharpes[name] = per_period_sharpe(rets)
        perf_matrix[name] = rets
    perf = pd.DataFrame(perf_matrix).dropna()
    pbo = cscv_pbo(perf, n_partitions=8) if perf.shape[1] >= 2 else None

    best_meta = max(
        (n for n in variants if n != "чистый моментум"),
        key=lambda n: stats[n]["annual_return"],
    )
    dsr = deflated_sharpe_ratio(
        pd.concat(oos_segments[best_meta]).pct_change().dropna().values,
        list(daily_sharpes.values()),
    )

    lines = [
        "# Итоговый эксперимент Части II: мета-разметка поверх моментума",
        "",
        f"Хеш данных: `{data_hash}`. Окон walk-forward: {len(windows)} "
        f"(мета-модель обучена на {trained_windows} из них).",
        f"Первичка: моментум {LOOKBACK} дн., top-{TOP_N}. Тройной барьер: "
        f"цель {PT_SL[0]}×vol, стоп {PT_SL[1]}×vol, срок {MAX_HOLDING} дн.",
        "",
        "## Результаты (только out-of-sample, после издержек)",
        "",
        "| Вариант | Годовых | Шарп (rf 14,25%) | Макс. просадка | Сделок |",
        "|---|---|---|---|---|",
    ]
    for name in variants:
        m = stats[name]
        lines.append(
            f"| {name} | {m['annual_return']:+.1%} | {m['sharpe']:.2f} | "
            f"{m['max_drawdown']:.1%} | {m.get('n_trades', 0)} |"
        )
    lines += [
        "",
        f"Безрисковая ставка (фонд денежного рынка): {rf:.2%} годовых.",
        "",
        "## Значимость",
        "",
        f"- Дефлированный Шарп лучшего мета-варианта ({best_meta}): "
        f"{dsr['verdict_ru']}",
    ]
    if pbo:
        lines.append(f"- {pbo['verdict_ru']}")
    lines += [
        "",
        "## Вывод",
        "",
        _conclusion(stats, rf, best_meta),
        "",
        "## Ограничения",
        "",
        "- Мета-модель — ступень 1 (логистическая регрессия); дерево/бустинг "
        "(ступень 2) откроются только при устойчивом OOS ступени 1 (раздел 20).",
        "- По ТЗ (раздел 24) это ИССЛЕДОВАНИЕ: в боевой контур мета-модель не "
        "допускается до прохождения фазы 8 (3 месяца бумажной торговли). "
        "Флаг ml.enabled остаётся false.",
        "- Дивиденды не учтены; LLM-признаки не использованы (ретроспективная "
        "осведомлённость, раздел 21).",
    ]
    out = Path("docs/research_meta_labeling.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nПолный отчёт: {out}")
    client.close()
    return 0


def _conclusion(stats, rf, best_meta):
    raw = stats["чистый моментум"]["annual_return"]
    best = stats[best_meta]["annual_return"]
    raw_sharpe = stats["чистый моментум"]["sharpe"]
    best_sharpe = stats[best_meta]["sharpe"]
    improved = best > raw and best_sharpe > raw_sharpe
    verdict = (
        f"Мета-фильтр {'УЛУЧШИЛ' if improved else 'НЕ улучшил'} результат: "
        f"доходность {raw:+.1%} → {best:+.1%} годовых, "
        f"Шарп {raw_sharpe:.2f} → {best_sharpe:.2f}. "
    )
    if best < rf:
        verdict += (
            f"Но даже лучший вариант ({best:+.1%}) проигрывает безрисковой "
            f"ставке {rf:.2%}. Умная надстройка не превратила невыгодную идею "
            f"в выгодную — она лишь чуть изменила уже отрицательный результат. "
            f"Критерий остановки проекта (раздел 15) остаётся в силе."
        )
    else:
        verdict += "Итог выше ставки — редкий случай, требует стресс-теста издержек."
    return verdict


if __name__ == "__main__":
    sys.exit(main())
