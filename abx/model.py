"""Модель прогноза: Dixon–Coles (базовая сила) + логистическая регрессия на признаках
(усталость, форма, мотивация, погода...) + смешивание с рынком.

Всё обучается «шагами вперёд» (walk-forward): прогноз на матч делается только
по данным, которые были доступны до него. Так бэктест честный.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CACHE, cfg
from .data import add_xg_proxy
from .features import build_features
from .odds import fair_power
from .ratings import elo_history, fit_dc, markets_from_matrix, score_matrix
from .weather import attach_weather

log = logging.getLogger(__name__)

FEATS_1X2 = ["dc_lph", "dc_lpa", "elo_diff", "ppg_diff", "xgd_diff", "luck_diff", "streak_diff",
             "rest_diff", "load_diff", "travel_km", "h_title_race", "a_title_race",
             "h_releg_fight", "a_releg_fight", "h_nothing", "a_nothing", "season_progress",
             "w_temp", "w_rain", "w_wind", "is_laliga"]
FEATS_OU = ["dc_lpo", "dc_total", "xg_sum", "w_temp", "w_rain", "w_wind", "rest_sum",
            "h_nothing", "a_nothing", "is_laliga"]


# ------------------------------------------------------------------ этап 1: DC walk-forward


def dc_walk_forward(df: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """λ хозяев/гостей для каждого матча по модели, подогнанной до этого матча.
    extra — будущие матчи (для них используется последняя подгонка на «сейчас»)."""
    m = cfg()["model"]
    rows = []
    for league, g in df.groupby("league"):
        model, last_fit = None, None
        fut = extra[extra.league == league] if extra is not None and len(extra) else None
        targets = pd.concat([g, fut]) if fut is not None else g
        for idx, r in targets.iterrows():
            day = r.date.normalize()
            if last_fit is None or (day - last_fit).days >= 7:
                fitted = fit_dc(g, day, m["half_life_days"], m["history_days"],
                                m["goals_weight"], m["ridge"])
                if fitted is not None:
                    model, last_fit = fitted, day
            if model is None:
                continue
            lh, la = model.lambdas(r.home, r.away)
            rows.append((idx, lh, la, model.rho))
    out = pd.DataFrame(rows, columns=["idx", "dc_lh", "dc_la", "dc_rho"]).set_index("idx")
    probs = [markets_from_matrix(score_matrix(a, b, c)) for a, b, c in
             zip(out.dc_lh, out.dc_la, out.dc_rho)]
    p = pd.DataFrame(probs, index=out.index).add_prefix("dc_")
    return out.join(p)


# ------------------------------------------------------------------ сборка признаков


def assemble(played: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """Полная таблица: матчи (+будущие) с DC, Elo, признаками, погодой, рынком."""
    played = add_xg_proxy(played)
    if upcoming is not None and len(upcoming):
        upcoming = upcoming.copy()
        upcoming.index = range(played.index.max() + 1, played.index.max() + 1 + len(upcoming))
    full = pd.concat([played, upcoming]) if upcoming is not None and len(upcoming) else played.copy()
    full = attach_weather(full, cfg()["model"]["use_weather"], cfg()["model"].get("weather_seasons", 3))

    dc = dc_walk_forward(played, upcoming)
    elo = elo_history(played)
    feats = build_features(played, upcoming)
    full = full.join(dc).join(feats)
    full[["elo_h", "elo_a"]] = elo
    cur = elo.attrs["current"]
    if upcoming is not None and len(upcoming):
        for idx in upcoming.index:
            full.loc[idx, "elo_h"] = cur.get((full.at[idx, "league"], full.at[idx, "home"]), 1450)
            full.loc[idx, "elo_a"] = cur.get((full.at[idx, "league"], full.at[idx, "away"]), 1450)

    eps = 1e-6
    full["dc_lph"] = np.log((full.dc_p_h + eps) / (full.dc_p_d + eps))
    full["dc_lpa"] = np.log((full.dc_p_a + eps) / (full.dc_p_d + eps))
    full["dc_lpo"] = np.log((full.dc_p_o25 + eps) / (full.dc_p_u25 + eps))
    full["dc_total"] = full.dc_lh + full.dc_la
    full["elo_diff"] = full.elo_h + 60 - full.elo_a
    full["ppg_diff"] = full.h_ppg - full.a_ppg
    full["xgd_diff"] = full.h_xgd - full.a_xgd
    full["luck_diff"] = full.h_luck - full.a_luck
    full["streak_diff"] = full.h_streak - full.a_streak
    full["load_diff"] = full.h_load14 - full.a_load14
    full["rest_sum"] = full.h_rest + full.a_rest
    full["xg_sum"] = full.h_xgd.abs() + full.a_xgd.abs()
    full["is_laliga"] = (full.league == "LaLiga").astype(float)

    # Рыночные вероятности (без маржи, степенной метод) по среднему коэффициенту
    mk = fair_power(full[["avg_h", "avg_d", "avg_a"]].values)
    full["mk_p_h"], full["mk_p_d"], full["mk_p_a"] = mk.T
    mo = fair_power(full[["avg_o25", "avg_u25"]].values)
    full["mk_p_o25"], full["mk_p_u25"] = mo.T
    # Справедливые вероятности на закрытии линии — для расчёта CLV
    cl = fair_power(full[["cl_h", "cl_d", "cl_a"]].values)
    full["cl_p_h"], full["cl_p_d"], full["cl_p_a"] = cl.T
    clo = fair_power(full[["cl_o25", "cl_u25"]].values)
    full["cl_p_o25"], full["cl_p_u25"] = clo.T
    return full


def _clean(X: pd.DataFrame, med: pd.Series | None = None):
    med = X.median() if med is None else med
    return X.fillna(med).fillna(0.0), med


# ------------------------------------------------------------------ этап 2


def stage2_walk_forward(full: pd.DataFrame, min_train: int = 600) -> pd.DataFrame:
    """Переобучает LR раз в месяц на всех прошлых матчах с готовым DC-прогнозом."""
    w = cfg()["model"]["market_weight"]
    data = full[full.dc_lh.notna()].copy()
    is_played = data.hg.notna()
    months = data.date.dt.to_period("M")
    out = pd.DataFrame(index=data.index, columns=["m_p_h", "m_p_d", "m_p_a", "m_p_o25"], dtype=float)
    for mth in sorted(months.unique()):
        target = data[months == mth]
        train = data[is_played & (data.date < target.date.min())]
        if len(train) < min_train:
            continue
        X, med = _clean(train[FEATS_1X2])
        y = train.result.map({"H": 0, "D": 1, "A": 2}).values
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        clf.fit(X, y)
        Xt, _ = _clean(target[FEATS_1X2], med)
        out.loc[target.index, ["m_p_h", "m_p_d", "m_p_a"]] = clf.predict_proba(Xt)

        Xo, medo = _clean(train[FEATS_OU])
        yo = ((train.hg + train.ag) > 2.5).astype(int).values
        clo = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        clo.fit(Xo, yo)
        Xot, _ = _clean(target[FEATS_OU], medo)
        out.loc[target.index, "m_p_o25"] = clo.predict_proba(Xot)[:, 1]
    data = data.join(out)
    # Если LR ещё не обучена (начало истории) — берём DC
    for k in ("h", "d", "a", "o25"):
        data[f"m_p_{k}"] = data[f"m_p_{k}"].fillna(data[f"dc_p_{k}"])
    data["m_p_u25"] = 1 - data.m_p_o25
    # Смешивание с рынком (рынок — степенной метод без маржи; модель — DC+признаки)
    for k in ("h", "d", "a", "o25", "u25"):
        mk = data[f"mk_p_{k}"]
        data[f"p_{k}"] = np.where(mk.notna(), w * data[f"m_p_{k}"] + (1 - w) * mk, data[f"m_p_{k}"])
    s = data[["p_h", "p_d", "p_a"]].sum(axis=1)
    for k in ("h", "d", "a"):
        data[f"p_{k}"] /= s
    return data


# ------------------------------------------------------------------ метрики


def evaluate(pred: pd.DataFrame, since_season: str | None = None) -> pd.DataFrame:
    """Log-loss и Brier для модели, DC, рынка и смеси (меньше = лучше)."""
    d = pred[pred.hg.notna() & pred.mk_p_h.notna()]
    if since_season:
        d = d[d.season >= since_season]
    y = d.result.map({"H": 0, "D": 1, "A": 2}).values
    Y = np.eye(3)[y]
    rows = {}
    for name, cols in {"Dixon–Coles": ["dc_p_h", "dc_p_d", "dc_p_a"],
                       "Модель (DC+признаки)": ["m_p_h", "m_p_d", "m_p_a"],
                       "Рынок (средний кэф)": ["mk_p_h", "mk_p_d", "mk_p_a"],
                       "Итог (смесь)": ["p_h", "p_d", "p_a"]}.items():
        P = np.clip(d[cols].values.astype(float), 1e-9, 1)
        P = P / P.sum(axis=1, keepdims=True)
        rows[name] = {"log_loss": float(-np.mean(np.log(P[np.arange(len(y)), y]))),
                      "brier": float(np.mean(((P - Y) ** 2).sum(axis=1))),
                      "accuracy": float(np.mean(P.argmax(axis=1) == y)),
                      "matches": len(y)}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ кэш


def _key(played: pd.DataFrame, upcoming: pd.DataFrame | None) -> str:
    h = hashlib.md5()
    h.update(json.dumps(cfg()["model"], sort_keys=True).encode())
    h.update(str(len(played)).encode())
    h.update(str(played.date.max()).encode())
    if upcoming is not None and len(upcoming):
        h.update(pd.util.hash_pandas_object(upcoming[["home", "away", "date"]], index=False).values.tobytes())
    return h.hexdigest()[:16]


def predictions(played: pd.DataFrame, upcoming: pd.DataFrame | None = None,
                use_cache: bool = True) -> pd.DataFrame:
    key = _key(played, upcoming)
    path = CACHE / f"pred_{key}.pkl"
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    full = assemble(played, upcoming)
    pred = stage2_walk_forward(full)
    for old in CACHE.glob("pred_*.pkl"):
        old.unlink()
    path.write_bytes(pickle.dumps(pred))
    return pred
