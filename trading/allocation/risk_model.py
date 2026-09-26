"""Модель риска: вероятность просадки индекса акций в ближайший месяц.

Вопрос модели: «упадёт ли индекс ниже сегодняшнего уровня на D% или больше
хотя бы раз в следующие H торговых дней?». Это не прогноз направления
(он на Мосбирже не удался, AUC ≈ 0,48), а прогноз риска: периоды высокой
волатильности идут кластерами, и это видно заранее.

Признаки — только из прошлого (данные по день t включительно), все
нормализуются внутри обучения (без этого L2-регуляризация «душит» мелкие
по масштабу признаки — ошибка, найденная раньше). Обучение — раз в год на
всей доступной истории; метки, чей горизонт заходит в проверочный год,
из обучения удаляются (очистка — защита от утечки будущего).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading.ml.models.linear import CalibratedLogisticModel

FEATURE_NAMES_RU = {
    "vol20": "волатильность за месяц",
    "vol60": "волатильность за квартал",
    "vol_ratio": "волатильность месяца к годовой",
    "mom20": "изменение индекса за месяц",
    "mom120": "изменение индекса за полгода",
    "dd250": "удаление от годового максимума",
    "ma200": "отклонение от 200-дневной средней",
    "bond_mom60": "изменение облигаций за квартал",
    "bond_vol20": "волатильность облигаций за месяц",
}


def risk_features(equity: pd.Series, bonds: pd.Series) -> pd.DataFrame:
    """Признаки на день t — только по данным до t включительно."""
    eq = equity.sort_index()
    bd = bonds.sort_index().reindex(eq.index.union(bonds.index)).ffill().reindex(eq.index)
    lr = np.log(eq).diff()
    blr = np.log(bd).diff()
    vol250 = lr.rolling(250).std()
    f = pd.DataFrame(index=eq.index)
    f["vol20"] = lr.rolling(20).std() * math.sqrt(252)
    f["vol60"] = lr.rolling(60).std() * math.sqrt(252)
    f["vol_ratio"] = lr.rolling(20).std() / vol250
    f["mom20"] = eq / eq.shift(20) - 1
    f["mom120"] = eq / eq.shift(120) - 1
    f["dd250"] = eq / eq.rolling(250).max() - 1
    f["ma200"] = eq / eq.rolling(200).mean() - 1
    f["bond_mom60"] = bd / bd.shift(60) - 1
    f["bond_vol20"] = blr.rolling(20).std() * math.sqrt(252)
    return f


def drawdown_labels(equity: pd.Series, horizon: int = 21, depth: float = 0.05) -> pd.Series:
    """1 — если в следующие ``horizon`` дней индекс хоть раз опустится на
    ``depth`` ниже уровня дня t. Последние ``horizon`` дней — NaN (будущее
    ещё не наступило)."""
    eq = equity.sort_index()
    values = eq.to_numpy()
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n - horizon):
        fwd_min = values[i + 1: i + 1 + horizon].min()
        out[i] = 1.0 if fwd_min / values[i] - 1 <= -depth else 0.0
    return pd.Series(out, index=eq.index, name="просадка")


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Площадь под ROC-кривой (вероятность, что опасный день получит оценку
    выше спокойного). 0,5 — модель не лучше монетки."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def explain_risk_ru(model, x, proba: float, base_rate: float) -> str:
    """Решение модели одной строкой: вероятность и три главные причины."""
    contrib = model.contributions(x)
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:3]
    parts = [f"{FEATURE_NAMES_RU.get(k, k)} ({'повышает' if v > 0 else 'снижает'} риск)"
             for k, v in top]
    return (f"Вероятность просадки: {proba:.0%} (обычно {base_rate:.0%}). "
            f"Главное: {'; '.join(parts)}.")


@dataclass
class RiskForecast:
    proba: pd.Series                         # OOS-вероятность просадки на каждый день
    base_rate: pd.Series                     # доля опасных дней в обучении на момент t
    yearly: pd.DataFrame                     # год → AUC модели, AUC «только волатильность», частота
    coefficients: dict[int, dict[str, float]] = field(default_factory=dict)
    explanations: dict[int, str] = field(default_factory=dict)


def walk_forward_risk(
    features: pd.DataFrame,
    labels: pd.Series,
    first_test_year: int,
    horizon: int = 21,
    min_train_rows: int = 750,
) -> RiskForecast:
    """Ежегодное переобучение: модель года Y учится только на днях, чья
    метка полностью известна до начала года Y, и предсказывает каждый день Y."""
    data = features.join(labels.rename("y"))
    valid_x = data[features.columns].notna().all(axis=1)
    years = sorted({d.year for d in data.index if d.year >= first_test_year})
    proba = pd.Series(np.nan, index=data.index)
    base = pd.Series(np.nan, index=data.index)
    rows, coefs, expl = [], {}, {}
    idx = data.index
    for year in years:
        test_mask = (idx.year == year) & valid_x.to_numpy()
        if not test_mask.any():
            continue
        first_test_pos = int(np.argmax(idx.year == year))
        # Очистка: метка дня i закрывается на i + horizon — должна быть < начала года.
        train_mask = np.zeros(len(idx), dtype=bool)
        train_mask[: max(first_test_pos - horizon, 0)] = True
        train_mask &= valid_x.to_numpy() & data["y"].notna().to_numpy()
        if train_mask.sum() < min_train_rows:
            continue
        X_tr = data.loc[train_mask, features.columns].to_numpy()
        y_tr = data.loc[train_mask, "y"].to_numpy().astype(int)
        model = CalibratedLogisticModel(list(features.columns))
        model.fit(X_tr, y_tr)
        X_te = data.loc[test_mask, features.columns].to_numpy()
        p = model.predict_proba(X_te)[:, 1]
        proba[test_mask] = p
        base[test_mask] = y_tr.mean()
        y_te = data.loc[test_mask, "y"]
        known = y_te.notna().to_numpy()
        rows.append({
            "год": year,
            "AUC модели": auc(y_te[known].to_numpy(), p[known]) if known.any() else np.nan,
            "AUC только волатильности": (
                auc(y_te[known].to_numpy(), data.loc[test_mask, "vol20"].to_numpy()[known])
                if known.any() else np.nan
            ),
            "частота просадок в году": float(y_te[known].mean()) if known.any() else np.nan,
            "частота в обучении": float(y_tr.mean()),
            "калибровка отключена": bool(getattr(model, "calibration_rejected", False)),
            "дней": int(test_mask.sum()),
        })
        coefs[year] = model.feature_importance()
        expl[year] = explain_risk_ru(model, X_te[-1], float(p[-1]), float(y_tr.mean()))
    yearly = pd.DataFrame(rows).set_index("год") if rows else pd.DataFrame()
    return RiskForecast(proba=proba.dropna(), base_rate=base.dropna(), yearly=yearly,
                        coefficients=coefs, explanations=expl)
