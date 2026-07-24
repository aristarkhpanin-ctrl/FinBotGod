"""ML-архитектура «уверенность ≥ 80% или кэш» на крипте (walk-forward).

Запуск:
    python scripts/run_ml_confidence.py

Реализует постановку заказчика:
* направление предсказывает обученная калиброванная модель, не правило;
* актив покупается только при уверенности ≥ порога (80% основной);
* нет уверенных активов — портфель в кэше (кэш допустим и нормален);
* цель — не потерять: смотрим на просадку и долю времени в кэше, а не
  только на доходность.

Схема честная: для каждого окна walk-forward модель обучается СТРОГО на
прошлом (генерация выборки тройным барьером + признаки), метки не
заглядывают за конец обучения; на проверочном окне модель только
предсказывает. BTC играет роль «индекса» (рыночный признак).

Пороги 0.80 (основной), а также 0.60/0.70 для контекста и PBO. Для «не
потерять» ключевые цифры — просадка, доля кэша и сравнение с фондом
денежного рынка (безрисковой альтернативой).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from trading.data.cache import ParquetCache
from trading.data.crypto_client import BitfinexClient, CryptoData, crypto_lot_size
from trading.data.validation import build_trading_calendar, validate_candles
from trading.engine.backtest import BacktestEngine
from trading.logging_setup import setup_logging
from trading.ml.features import FEATURE_NAMES, bet_features, features_frame
from trading.ml.labeling import concurrency_weights, meta_labels
from trading.ml.models.linear import CalibratedLogisticModel
from trading.ml.validation import cscv_pbo
from trading.reporting.metrics import (
    annualize,
    compute_metrics,
    deflated_sharpe_ratio,
    per_period_sharpe,
)
from trading.research.ledger import HypothesisLedger
from trading.research.walkforward import year_windows
from trading.settings import DEFAULT_SETTINGS_PATH, load_universe, load_settings
from trading.strategy.examples.ml_confidence import IMOEX_KEY, MLConfidenceLong

MAX_HOLDING = 20
PT_SL = (1.5, 1.0)          # цель 1,5×vol, стоп 1,0×vol
VOL_SPAN = 20
THRESHOLDS = [0.60, 0.70, 0.80]
MAX_POSITIONS = 6


def month_ends(index, start, end):
    sub = index[(index >= start) & (index <= end)]
    if len(sub) == 0:
        return []
    ser = pd.Series(sub, index=sub)
    return list(ser.groupby([sub.year, sub.month]).last().values)


def build_training_set(coins, btc, train_start, train_end, index):
    """Выборка (признаки, метка роста, t1) по ВСЕМ активам на всех датах
    ребалансировки обучающего окна. Метка = 1, если длинная позиция закрылась
    в плюс (тройной барьер). Горизонт не выходит за конец обучения."""
    cutoff = index.searchsorted(train_end)
    per_coin = {}
    for t in month_ends(index, train_start, train_end):
        pos = index.searchsorted(t)
        if pos + MAX_HOLDING > cutoff:
            continue
        for secid, df in coins.items():
            d = df[df["date"] <= t]
            if len(d) >= 130 and d["date"].iloc[-1] == t:
                per_coin.setdefault(secid, []).append(t)

    rows, labels, t0s, t1s = [], [], [], []
    btc_close = btc.set_index("date")["close"] if btc is not None else None
    for secid, entries in per_coin.items():
        close = coins[secid].set_index("date")["close"]
        idx = pd.DatetimeIndex(sorted(set(entries)))
        lab = meta_labels(close, idx, pd.Series(1, index=idx), pt_sl=PT_SL,
                          max_holding=MAX_HOLDING, vol_span=VOL_SPAN)
        for t0, row in lab.iterrows():
            rows.append(bet_features(
                close[close.index <= t0].reset_index(drop=True),
                None,
                btc_close[btc_close.index <= t0].reset_index(drop=True)
                if btc_close is not None else None,
            ))
            labels.append(int(row["meta"]))
            t0s.append(t0)
            t1s.append(row["t1"])
    if not rows:
        return None
    df = features_frame(rows)
    df["y"] = labels
    df["t0"] = t0s
    df["t1"] = t1s
    return df.dropna(subset=FEATURE_NAMES).reset_index(drop=True)


def train_model(train_df):
    if train_df is None or len(train_df) < 60 or train_df["y"].nunique() < 2:
        return None
    train_df = train_df.sort_values("t0").reset_index(drop=True)
    X, y = train_df[FEATURE_NAMES], train_df["y"]
    t1 = pd.Series(train_df["t1"].values, index=train_df["t0"].values)
    w = concurrency_weights(t1, pd.DatetimeIndex(sorted(train_df["t0"].unique())))
    sw = pd.Series(w.reindex(train_df["t0"].values).fillna(1.0).values)
    model = CalibratedLogisticModel(FEATURE_NAMES)
    model.fit(X.to_numpy(), y.to_numpy(), sample_weight=sw.to_numpy())
    return model


def main() -> int:
    setup_logging()
    cfg = DEFAULT_SETTINGS_PATH.parent
    settings = load_settings(cfg / "settings_crypto.yaml")
    tickers = load_universe(cfg / "crypto_universe.yaml", uppercase=False)
    cache = ParquetCache(settings.data.cache_dir)
    market = CryptoData(BitfinexClient(), cache)
    frames = {t: market.daily_candles(t) for t in tickers}
    calendar = build_trading_calendar(frames)
    clean = {t: df for t, df in frames.items()
             if not validate_candles(df, t, jump_threshold=settings.data.price_jump_threshold,
                                     trading_calendar=calendar).suspicious and not df.empty}
    btc = clean.get("tBTCUSD")
    index = pd.DatetimeIndex(sorted(calendar))
    lot_sizes = {t: crypto_lot_size(float(df["close"].median())) for t, df in clean.items()}
    data_hash = cache.data_version_hash()
    ledger = HypothesisLedger("journal/hypotheses.sqlite")

    # BTC как «индекс» (рыночный признак) — служебный ряд, не торгуется.
    engine_candles = dict(clean)
    if btc is not None:
        engine_candles[IMOEX_KEY] = btc[["date", "open", "high", "low", "close", "value", "volume"]]

    years = sorted({d.year for d in index})
    windows = year_windows(years[0], years[-1], 4, 1)

    carried = {th: settings.capital.start_amount for th in THRESHOLDS}
    oos_seg = {th: [] for th in THRESHOLDS}
    oos_fills = {th: [] for th in THRESHOLDS}
    cash_shares = {th: [] for th in THRESHOLDS}
    trained = 0

    for w in windows:
        model = train_model(build_training_set(clean, btc, w.train_start, w.train_end, index))
        if model is not None:
            trained += 1
        for th in THRESHOLDS:
            strat = MLConfidenceLong(model, threshold=th, max_positions=MAX_POSITIONS)
            s = settings.model_copy(deep=True)
            s.capital.start_amount = carried[th]
            s.risk.max_order_value = settings.risk.max_order_value * (
                carried[th] / settings.capital.start_amount)
            sliced = {sec: df[df["date"] <= w.test_end] for sec, df in engine_candles.items()}
            res = BacktestEngine(
                candles=sliced, lot_sizes=lot_sizes, settings=s, strategy=strat,
                ledger=ledger, data_hash=f"{data_hash}|mlconf|{th}", source="llm",
                trade_from=str(pd.Timestamp(w.test_start).date()),
            ).run()
            carried[th] = float(res.equity.iloc[-1])
            oos_seg[th].append(res.equity)
            oos_fills[th].extend(res.fills)

    rf = settings.benchmark.risk_free_rate
    stats, perf = {}, {}
    for th in THRESHOLDS:
        eq = pd.concat(oos_seg[th])
        eq = eq[~eq.index.duplicated(keep="last")]
        stats[th] = compute_metrics(eq, rf, fills=oos_fills[th])
        perf[th] = eq.pct_change().dropna()
    perf_df = pd.DataFrame(perf).dropna()
    pbo = cscv_pbo(perf_df, n_partitions=8) if perf_df.shape[1] >= 2 else None
    best_th = max(THRESHOLDS, key=lambda t: stats[t]["annual_return"])
    dsr = deflated_sharpe_ratio(perf[best_th].values,
                                [per_period_sharpe(perf[t]) for t in THRESHOLDS])

    # Бенчмарки на всём OOS-периоде.
    oos_start = min(pd.concat(oos_seg[th]).index[0] for th in THRESHOLDS)
    oos_end = max(pd.concat(oos_seg[th]).index[-1] for th in THRESHOLDS)
    btc_ann = None
    if btc is not None:
        seg = btc[(btc["date"] >= oos_start) & (btc["date"] <= oos_end)]
        if len(seg) >= 2:
            btc_ann = annualize(float(seg["close"].iloc[-1]) / float(seg["close"].iloc[0]),
                                (seg["date"].iloc[-1] - seg["date"].iloc[0]).days)

    lines = [
        "# ML-архитектура «уверенность ≥ 80% или кэш» (крипта)",
        "",
        f"Хеш данных: `{data_hash}`. Окон: {len(windows)} (модель обучена на "
        f"{trained}). Модель: калиброванная логистическая регрессия, "
        f"признаки point-in-time, BTC как рыночный сигнал. Цель — не потерять.",
        "",
        "Направление задаёт ТОЛЬКО модель; актив покупается лишь при "
        "уверенности не ниже порога, иначе — кэш.",
        "",
        "## Результаты по порогам уверенности (OOS, после издержек)",
        "",
        "| Порог | Годовых | Итог $ | Макс. просадка | Шарп | Сделок |",
        "|---|---|---|---|---|---|",
    ]
    for th in THRESHOLDS:
        m = stats[th]
        lines.append(
            f"| ≥{int(th*100)}% | {m['annual_return']:+.1%} | "
            f"${m['end_equity']:.0f} | {m['max_drawdown']:.1%} | "
            f"{m['sharpe']:.2f} | {m.get('n_trades', 0)} |"
        )
    lines += [
        "",
        f"Безрисковая альтернатива (фонд денежного рынка): {rf:.1%} годовых. "
        f"Купил-и-держи BTC: " + (f"{btc_ann:+.1%} годовых" if btc_ann else "н/д") + ".",
        "",
        "ВАЖНО: в симуляции свободный кэш НЕ приносит дохода (0%), тогда как "
        "фонд денежного рынка даёт 4,5%. Поэтому «сидеть в кэше» здесь — это "
        "0% номинально; честное сравнение цели «не потерять» — именно с "
        "фондом денежного рынка.",
        "",
        "## Значимость",
        "",
        f"- Порог 80% как основной: {stats[0.80]['annual_return']:+.1%} годовых, "
        f"просадка {stats[0.80]['max_drawdown']:.1%}.",
        f"- Дефлированный Шарп лучшего порога (≥{int(best_th*100)}%): {dsr['verdict_ru']}",
    ]
    if pbo:
        lines.append(f"- {pbo['verdict_ru']}")
    lines += [
        "",
        "## Ограничения",
        "",
        "- Выживаемость выборки (только живые на 2026 монеты) завышает результат.",
        "- Калибровка приблизительная (Platt на отложенном хвосте обучения); "
        "при высоком пороге 80% уверенных сигналов может быть очень мало.",
        "- Ступень 1 (логистическая регрессия); дерево/бустинг — ступень 2, "
        "открывается только при устойчивом OOS ступени 1 (ТЗ, раздел 20).",
        "- ml.enabled остаётся false: это исследование, не боевой контур "
        "(ТЗ, раздел 24).",
    ]
    out = Path("docs/research_ml_confidence.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nПолный отчёт: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
