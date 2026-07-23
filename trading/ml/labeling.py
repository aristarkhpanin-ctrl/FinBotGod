"""Разметка данных — ЧАСТЬ II (ТЗ, раздел 18).

Тройной барьер вместо фиксированного горизонта: от точки входа ставятся
три барьера — верхний (цель по прибыли), нижний (стоп-лосс), вертикальный
(максимальный срок). Метка = барьер, которого цена коснулась первой.
Барьеры масштабируются волатильностью на момент входа, а не фиксированными
процентами.

Мета-разметка: первичная модель (простое правило) задаёт направление,
вторичная ML-модель отвечает лишь «стоит ли действовать по этому сигналу»
(бинарно). Эту схему ТЗ велит реализовать первой — она проще предсказания
направления и с лучшим соотношением сигнал/шум.

Веса наблюдений: метки тройного барьера перекрываются во времени и не
независимы. Вес наблюдения обратен конкурентности (сколько меток
одновременно используют бары его горизонта).

Реализация своя, с тестами на синтетике с известным ответом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """Оценка дневной волатильности — EWM-стандартное отклонение доходностей.

    Используется для масштабирования барьеров: на спокойном рынке барьеры
    уже, на волатильном — шире.
    """
    returns = close.pct_change()
    return returns.ewm(span=span).std()


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] = (1.0, 1.0),
) -> pd.DataFrame:
    """Первое касание барьера для каждого события.

    ``events`` — DataFrame, индекс = время входа, колонки:
      * ``vertical`` — время вертикального барьера (макс. срок удержания);
      * ``target``   — масштаб барьеров (обычно волатильность на входе);
      * ``side``     — направление ставки (+1 лонг, −1 шорт); опционально,
        для мета-разметки. Без него барьеры симметричны (обычная разметка).
    ``pt_sl`` — множители верхнего и нижнего барьеров относительно target.

    Возвращает DataFrame: ``t1`` (время первого касания), ``ret``
    (доходность на момент касания, с учётом side), ``label`` (какой барьер:
    +1 верхний, −1 нижний, 0 вертикальный).
    """
    out = pd.DataFrame(index=events.index, columns=["t1", "ret", "label"], dtype=object)
    has_side = "side" in events.columns
    for t0, ev in events.iterrows():
        path = close[t0: ev["vertical"]]
        if len(path) < 2:
            continue
        side = ev["side"] if has_side else 1.0
        returns = (path / close[t0] - 1) * side
        up = pt_sl[0] * ev["target"]
        dn = -pt_sl[1] * ev["target"]

        t_up = returns[returns >= up].index.min() if (returns >= up).any() else pd.NaT
        t_dn = returns[returns <= dn].index.min() if (returns <= dn).any() else pd.NaT
        t_vert = path.index[-1]

        touches = {"up": t_up, "dn": t_dn, "vert": t_vert}
        first = min((t for t in touches.values() if pd.notna(t)), default=t_vert)
        if first == t_up:
            label = 1
        elif first == t_dn:
            label = -1
        else:
            label = 0
        out.loc[t0, "t1"] = first
        out.loc[t0, "ret"] = float(returns.loc[first])
        out.loc[t0, "label"] = label
    return out.dropna(subset=["t1"])


def triple_barrier_labels(
    close: pd.Series,
    events_index: pd.Index,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_holding: int = 10,
    vol_span: int = 20,
) -> pd.DataFrame:
    """Удобная обёртка: строит барьеры для набора точек входа.

    ``max_holding`` — вертикальный барьер в барах. Волатильность на входе
    берётся как EWM-оценка на момент t0.
    """
    vol = daily_volatility(close, span=vol_span)
    idx = close.index
    events = pd.DataFrame(index=events_index)
    verticals = []
    for t0 in events_index:
        pos = idx.get_loc(t0)
        vpos = min(pos + max_holding, len(idx) - 1)
        verticals.append(idx[vpos])
    events["vertical"] = verticals
    events["target"] = vol.reindex(events_index).values
    events = events.dropna(subset=["target"])
    events = events[events["target"] > 0]
    return apply_triple_barrier(close, events, pt_sl)


def meta_labels(
    close: pd.Series,
    events_index: pd.Index,
    side: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_holding: int = 10,
    vol_span: int = 20,
) -> pd.DataFrame:
    """Мета-метки: сработал ли сигнал первичной модели.

    ``side`` — направление ставки первичной модели для каждого события
    (+1 лонг, −1 шорт). Мета-метка ``meta`` = 1, если позиция закрылась
    в плюс (в свою сторону), иначе 0. Именно её предсказывает вторичная
    модель — «действовать по сигналу или нет».
    """
    vol = daily_volatility(close, span=vol_span)
    idx = close.index
    events = pd.DataFrame(index=events_index)
    events["side"] = side.reindex(events_index).values
    verticals = []
    for t0 in events_index:
        pos = idx.get_loc(t0)
        verticals.append(idx[min(pos + max_holding, len(idx) - 1)])
    events["vertical"] = verticals
    events["target"] = vol.reindex(events_index).values
    events = events.dropna(subset=["target", "side"])
    events = events[events["target"] > 0]

    result = apply_triple_barrier(close, events, pt_sl)
    result["meta"] = (result["ret"] > 0).astype(int)
    result["side"] = events["side"].reindex(result.index)
    return result


def concurrency_weights(t1: pd.Series, close_index: pd.Index) -> pd.Series:
    """Веса наблюдений, обратные конкурентности перекрывающихся меток.

    ``t1`` — Series: индекс = время входа события, значение = время закрытия
    метки. Вес события = среднее по барам его горизонта от 1/(число меток,
    активных на этом баре). Веса нормируются к среднему 1.
    """
    if t1.empty:
        return pd.Series(dtype=float)
    # Сколько меток активно на каждом баре.
    concurrency = pd.Series(0, index=close_index, dtype=float)
    for t0, t1_i in t1.items():
        concurrency.loc[t0:t1_i] += 1
    concurrency = concurrency.replace(0, np.nan)

    weights = pd.Series(index=t1.index, dtype=float)
    for t0, t1_i in t1.items():
        span = concurrency.loc[t0:t1_i]
        weights.loc[t0] = (1.0 / span).mean()
    weights = weights / weights.mean()   # средний вес = 1
    return weights
