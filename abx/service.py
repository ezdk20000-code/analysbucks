"""Высокоуровневые функции для CLI, дашборда и Telegram-бота."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import betting, data, ledger, model
from .config import cfg
from .features import apply_manual, manual_adjustments
from .odds import fair_power
from .ratings import elo_history, fit_dc, markets_from_matrix, score_matrix, top_scores
from .staking import Staker

log = logging.getLogger(__name__)


def update(download: bool = True) -> dict:
    """Скачать свежие данные и рассчитать завершившиеся ставки."""
    if download:
        data.download()
    played = data.load_matches()
    pred = predictions()
    closing = pred.set_index("match_id")
    settled = ledger.auto_settle(played, closing)
    return {"матчей в базе": len(played), "рассчитано ставок": len(settled)}


def predictions(use_cache: bool = True) -> pd.DataFrame:
    played = data.load_matches()
    upcoming = data.load_fixtures(played)
    return model.predictions(played, upcoming if len(upcoming) else None, use_cache=use_cache)


def upcoming_predictions(pred: pd.DataFrame | None = None) -> pd.DataFrame:
    pred = predictions() if pred is None else pred
    up = pred[pred.hg.isna()].copy()
    return up.sort_values("date")


def current_bank() -> float:
    b = ledger.balance()
    return b["свободно"] if b["капитал"] > 0 else float(cfg()["backtest"]["start_bank"])


def recommendations(pred: pd.DataFrame | None = None, bank: float | None = None) -> dict:
    """Value-ординары и экспрессы на ближайшие матчи с рекомендуемой суммой."""
    up = upcoming_predictions(pred)
    bank = current_bank() if bank is None else bank
    if up.empty:
        return {"singles": pd.DataFrame(), "express": pd.DataFrame(), "bank": bank}
    c = betting.candidates(up)
    singles = betting.filter_singles(c, cfg()["betting"])
    st = Staker(cfg()["staking"], bank)
    if len(singles):
        stakes = [st.size(p, o, bank).stake for p, o in zip(singles.p, singles.odds)]
        singles = singles.assign(stake=st.apply_exposure_cap(stakes, bank))
        singles = singles[singles.stake > 0]
    ex = pd.DataFrame()
    e = cfg()["express"]
    if e.get("enabled"):
        ex = betting.build_expresses(c, e)
        if len(ex):
            ex = ex.assign(stake=[st.size(p, o, bank, int(n)).stake
                                  for p, o, n in zip(ex.p, ex.odds, ex.legs)])
            ex = ex[ex.stake > 0]
            ex["legs_data"] = [_legs_for(c, mid) for mid in ex.match_id]
    return {"singles": singles, "express": ex, "bank": bank}


def _legs_for(c: pd.DataFrame, express_id: str) -> list[dict]:
    legs = []
    cand = c.sort_values("edge", ascending=False)
    for mid in express_id.split("+"):
        r = cand[(cand.match_id == mid) & (cand.market == "1X2")].iloc[0]
        legs.append(leg_dict(r))
    return legs


def leg_dict(r) -> dict:
    return {"match_id": r.match_id, "league": r.league, "date": pd.Timestamp(r.date).isoformat(),
            "home": r.home, "away": r.away, "market": r.market, "sel": r.sel,
            "odds": float(r.odds), "p": float(r.p)}


def match_forecast(home: str, away: str, league: str | None = None,
                   when: pd.Timestamp | None = None) -> dict:
    """Быстрый прогноз любой пары по текущим рейтингам (без коэффициентов)."""
    played = data.add_xg_proxy(data.load_matches())
    if league is None:
        last = played[played.home.isin([home, away]) | played.away.isin([home, away])]
        if last.empty:
            raise KeyError(f"Не нашёл команды: {home}, {away}")
        league = last.league.iloc[-1]
    when = when or pd.Timestamp.now()
    g = played[played.league == league]
    m = cfg()["model"]
    dc = fit_dc(g, when, m["half_life_days"], m["history_days"], m["goals_weight"], m["ridge"])
    lh, la = dc.lambdas(home, away)
    lh, la, notes = apply_manual(lh, la, home, away, when, manual_adjustments())
    mat = score_matrix(lh, la, dc.rho)
    mk = markets_from_matrix(mat)
    elo = elo_history(played).attrs["current"]
    return {"league": league, "home": home, "away": away, "xg_home": lh, "xg_away": la, **mk,
            "fair_odds": {k: round(1 / v, 2) for k, v in mk.items() if v > 0},
            "top_scores": top_scores(mat, 5),
            "elo_home": elo.get((league, home)), "elo_away": elo.get((league, away)),
            "notes": notes}


def team_names() -> dict[str, list[str]]:
    played = data.load_matches()
    cur = played[played.season == played.season.max()]
    return {lg: sorted(set(g.home) | set(g.away)) for lg, g in cur.groupby("league")}


def strength_table(league: str) -> pd.DataFrame:
    played = data.add_xg_proxy(data.load_matches())
    g = played[played.league == league]
    m = cfg()["model"]
    dc = fit_dc(g, pd.Timestamp.now(), m["half_life_days"], m["history_days"],
                m["goals_weight"], m["ridge"])
    cur = set(g[g.season == g.season.max()].home)
    t = dc.table()
    t = t[t.team.isin(cur)].reset_index(drop=True)
    elo = elo_history(played).attrs["current"]
    t["elo"] = [round(elo.get((league, x), np.nan)) for x in t.team]
    lh_avg = np.exp(dc.mu + dc.home)
    t["xG за (vs средний)"] = np.round(lh_avg * np.exp(t.attack), 2)
    t["xG против"] = np.round(np.exp(dc.mu) * np.exp(-t.defence), 2)
    return t.round(3)


def backtest(overrides: dict | None = None, since: str | None = None, mc: bool = True,
             pred: pd.DataFrame | None = None) -> dict:
    """Бэктест стратегии. overrides: {'betting': {...}, 'staking': {...}, 'express': {...},
    'mode': 'singles'|'express'|'both'}."""
    o = overrides or {}
    b = {**cfg()["betting"], **o.get("betting", {})}
    s = {**cfg()["staking"], **o.get("staking", {})}
    e = {**cfg()["express"], **o.get("express", {})}
    bt = {**cfg()["backtest"], **o.get("backtest", {})}
    mode = o.get("mode", "both")
    pred = predictions() if pred is None else pred
    since = since or str(bt["start_season"])
    hist = pred[pred.hg.notna() & (pred.season >= since)]
    c = betting.candidates(hist, b)
    parts = []
    if mode in ("singles", "both"):
        parts.append(betting.filter_singles(c, b).assign(legs=1))
    if mode in ("express", "both"):
        ex = betting.build_expresses(c, e)
        if len(ex):
            parts.append(ex)
    bets = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if bets.empty:
        return {"summary": {"ставок": 0}, "curve": pd.DataFrame(), "bets": bets, "mc": {}}
    res = betting.simulate(bets, bt["start_bank"], s)
    out = {"summary": res.summary, "curve": res.curve, "bets": res.bets}
    out["by_season"] = (res.bets.groupby("season")
                        .apply(lambda g: pd.Series({"ставок": len(g),
                                                    "ROI флэт %": round(100 * np.where(g.win == 1, g.odds - 1, -1).mean(), 2),
                                                    "прибыль": round(g.pnl.sum(), 2)}),
                               include_groups=False))
    out["mc"] = betting.monte_carlo(res.bets, s, bt["start_bank"], bt["monte_carlo_runs"]) if mc else {}
    return out


def market_check() -> pd.DataFrame:
    """Калибровка: насколько честны вероятности рынка (степенной метод) по корзинам."""
    played = data.load_matches()
    P = fair_power(played[["avg_h", "avg_d", "avg_a"]].values)
    W = np.column_stack([played.result == "H", played.result == "D", played.result == "A"])
    p, w = P.ravel(), W.ravel()
    ok = ~np.isnan(p)
    bins = pd.cut(p[ok], [0, .1, .2, .3, .4, .5, .6, .7, .8, 1])
    return (pd.DataFrame({"p": p[ok], "w": w[ok]}).groupby(bins, observed=True)
            .agg(прогноз=("p", "mean"), факт=("w", "mean"), n=("w", "size")).round(3))


__all__ = ["update", "predictions", "upcoming_predictions", "recommendations", "match_forecast",
           "team_names", "strength_table", "backtest", "current_bank", "leg_dict"]
