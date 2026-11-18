"""Признаки матча до его начала: форма, усталость, переезды, мотивация, «психология», погода.

Всё считается только по информации, известной ДО матча (без заглядывания в будущее).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import MANUAL
from .teams import haversine_km

FORM_N = 6


def _team_log(df: pd.DataFrame) -> pd.DataFrame:
    """Длинная таблица: одна строка = команда в матче."""
    h = pd.DataFrame({"i": df.index, "date": df.date, "league": df.league, "season": df.season,
                      "team": df.home, "opp": df.away, "is_home": 1,
                      "gf": df.hg, "ga": df.ag, "xf": df.hxg, "xa": df.axg})
    a = pd.DataFrame({"i": df.index, "date": df.date, "league": df.league, "season": df.season,
                      "team": df.away, "opp": df.home, "is_home": 0,
                      "gf": df.ag, "ga": df.hg, "xf": df.axg, "xa": df.hxg})
    return pd.concat([h, a]).sort_values(["date", "i"]).reset_index(drop=True)


def build_features(played: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """Возвращает признаки для всех строк played (+upcoming), индекс сохраняется.

    upcoming — будущие матчи без счёта; для них форма/таблица берутся на текущий момент.
    """
    allm = played.copy()
    allm["_played"] = True
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["_played"] = False
        up.index = range(allm.index.max() + 1, allm.index.max() + 1 + len(up))
        for c in ("hxg", "axg", "hg", "ag"):
            if c not in up:
                up[c] = np.nan
        allm = pd.concat([allm, up])
    allm = allm.sort_values("date", kind="stable")

    feats: dict[int, dict] = {}
    # Состояние по командам
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    dates: dict[str, list] = defaultdict(list)
    streak: dict[str, int] = defaultdict(int)          # >0 серия без поражений-побед, <0 без побед
    table: dict[tuple, dict] = defaultdict(lambda: {"pts": 0, "gd": 0, "gp": 0})
    season_teams: dict[tuple, set] = defaultdict(set)
    for _, r in allm.iterrows():
        season_teams[(r.league, r.season)].update((r.home, r.away))

    for idx, r in allm.iterrows():
        f: dict = {}
        key = (r.league, r.season)
        n_teams = max(len(season_teams[key]), 2)
        total_games = 2 * (n_teams - 1)
        # Таблица на момент матча
        standings = sorted(
            ((t, table[(r.league, r.season, t)]) for t in season_teams[key]),
            key=lambda x: (-x[1]["pts"], -x[1]["gd"]))
        pos = {t: i + 1 for i, (t, _) in enumerate(standings)}
        pts_at = [s["pts"] for _, s in standings]

        for side, team, opp in (("h", r.home, r.away), ("a", r.away, r.home)):
            hh = list(hist[team])[-FORM_N:]
            if hh:
                f[f"{side}_ppg"] = np.mean([g[0] for g in hh])
                f[f"{side}_gd"] = np.mean([g[1] - g[2] for g in hh])
                f[f"{side}_xgd"] = np.mean([g[3] - g[4] for g in hh])
                # «Удача»: голы минус xG-прокси за 10 игр — склонна возвращаться к нулю
                f[f"{side}_luck"] = np.mean([(g[1] - g[3]) - (g[2] - g[4]) for g in hist[team]])
            else:
                f[f"{side}_ppg"] = f[f"{side}_gd"] = f[f"{side}_xgd"] = f[f"{side}_luck"] = np.nan
            ds = dates[team]
            f[f"{side}_rest"] = min((r.date - ds[-1]).days, 14) if ds else 14
            f[f"{side}_load14"] = sum(1 for d in ds[-6:] if (r.date - d).days <= 14)
            f[f"{side}_streak"] = streak[team]
            st = table[(r.league, r.season, team)]
            gp = st["gp"]
            remaining = max(total_games - gp, 0)
            f[f"{side}_pos"] = pos.get(team, n_teams)
            if gp >= 3 and len(pts_at) >= 18:
                top, ucl, releg = pts_at[0], pts_at[3], pts_at[n_teams - 4]
                p = st["pts"]
                reach = 3 * remaining
                f[f"{side}_title_race"] = float(top - p <= reach * 0.35 and pos[team] <= 4)
                f[f"{side}_releg_fight"] = float(p - releg <= reach * 0.35 and pos[team] >= n_teams - 7)
                europe = pts_at[6]
                f[f"{side}_nothing"] = float(europe - p > reach * 0.5 and p - releg > reach * 0.5)
                f[f"{side}_ucl_gap"] = p - ucl
            else:
                f[f"{side}_title_race"] = f[f"{side}_releg_fight"] = f[f"{side}_nothing"] = 0.0
                f[f"{side}_ucl_gap"] = 0.0
            f["season_progress"] = gp / total_games if total_games else 0
        f["rest_diff"] = f["h_rest"] - f["a_rest"]
        f["travel_km"] = haversine_km(r.home, r.away)
        feats[idx] = f

        if r._played:
            for team, gf, ga, xf, xa in ((r.home, r.hg, r.ag, r.hxg, r.axg),
                                         (r.away, r.ag, r.hg, r.axg, r.hxg)):
                pts = 3 if gf > ga else 1 if gf == ga else 0
                hist[team].append((pts, gf, ga, xf, xa))
                dates[team].append(r.date)
                s = streak[team]
                if pts == 3:
                    streak[team] = s + 1 if s > 0 else 1
                elif pts == 0:
                    streak[team] = s - 1 if s < 0 else -1
                else:
                    streak[team] = 0
                t = table[(r.league, r.season, team)]
                t["pts"] += pts
                t["gd"] += gf - ga
                t["gp"] += 1

    out = pd.DataFrame.from_dict(feats, orient="index")
    out["travel_km"] = out["travel_km"].fillna(out["travel_km"].median())
    return out


def manual_adjustments() -> pd.DataFrame:
    """Ручные поправки (травмы, мораль, смена тренера, ротация перед еврокубками).

    Файл data/manual/team_news.csv:
      date_from,date_to,team,attack_mult,defence_mult,note
      2026-11-20,2026-12-05,Arsenal,0.92,1.05,"Травма Сака + ротация перед ЛЧ"
    attack_mult <1 — команда будет забивать меньше, defence_mult >1 — пропускать больше.
    """
    path = MANUAL / "team_news.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date_from", "date_to", "team", "attack_mult",
                                     "defence_mult", "note"])
    return pd.read_csv(path, parse_dates=["date_from", "date_to"])


def apply_manual(lh: float, la: float, home: str, away: str, when: pd.Timestamp,
                 adj: pd.DataFrame) -> tuple[float, float, list[str]]:
    notes: list[str] = []
    if adj is None or adj.empty:
        return lh, la, notes
    act = adj[(adj.date_from <= when) & (adj.date_to >= when)]
    for _, a in act.iterrows():
        am = float(a.get("attack_mult", 1) or 1)
        dm = float(a.get("defence_mult", 1) or 1)
        if a.team == home:
            lh *= am
            la *= dm
        elif a.team == away:
            la *= am
            lh *= dm
        else:
            continue
        notes.append(f"{a.team}: {a.get('note', '')}")
    return lh, la, notes
