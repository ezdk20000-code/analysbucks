"""Рейтинги силы команд: Elo и модель Dixon–Coles (Пуассон с поправкой на низкие счета)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# ---------------------------------------------------------------- Elo


def elo_history(df: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0,
                carry: float = 0.75) -> pd.DataFrame:
    """Elo до каждого матча. Между сезонами рейтинг регрессирует к 1500 (carry),
    новички лиги стартуют со среднего рейтинга вылетевших команд."""
    out_h, out_a = np.empty(len(df)), np.empty(len(df))
    ratings: dict[tuple[str, str], float] = {}
    last_season: dict[str, str] = {}
    for league, g in df.groupby("league", sort=False):
        r: dict[str, float] = {}
        season = None
        for idx, row in g.iterrows():
            if row.season != season:
                if season is not None:
                    teams_now = set(g[g.season == row.season].home) | set(g[g.season == row.season].away)
                    gone = [t for t in r if t not in teams_now]
                    newcomer = np.mean([r[t] for t in gone]) if gone else 1450.0
                    r = {t: 1500 + carry * (v - 1500) for t, v in r.items()}
                    for t in teams_now:
                        r.setdefault(t, 1500 + carry * (newcomer - 1500))
                season = row.season
            rh, ra = r.setdefault(row.home, 1500.0), r.setdefault(row.away, 1500.0)
            out_h[df.index.get_loc(idx)], out_a[df.index.get_loc(idx)] = rh, ra
            exp_h = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
            s = 1.0 if row.hg > row.ag else 0.5 if row.hg == row.ag else 0.0
            gd = abs(row.hg - row.ag)
            mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
            delta = k * mult * (s - exp_h)
            r[row.home], r[row.away] = rh + delta, ra - delta
        for t, v in r.items():
            ratings[(league, t)] = v
            last_season[t] = season
    res = pd.DataFrame({"elo_h": out_h, "elo_a": out_a}, index=df.index)
    res.attrs["current"] = ratings
    return res


# ---------------------------------------------------------------- Dixon–Coles


def dc_tau(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-6, None)


@dataclass
class DCModel:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home: float
    mu: float
    rho: float
    fitted_at: pd.Timestamp | None = None
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.teams)}

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Ожидаемые голы. Неизвестная команда (новичок) — чуть ниже среднего."""
        weak = np.percentile(self.attack, 25), np.percentile(self.defence, 25)
        ah, dh = (self.attack[self.index[home]], self.defence[self.index[home]]) if home in self.index else weak
        aa, da = (self.attack[self.index[away]], self.defence[self.index[away]]) if away in self.index else weak
        lh = math.exp(self.mu + self.home + ah - da)
        la = math.exp(self.mu + aa - dh)
        return lh, la

    def table(self) -> pd.DataFrame:
        return (pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})
                .assign(strength=lambda d: d.attack + d.defence)
                .sort_values("strength", ascending=False).reset_index(drop=True))


def fit_dc(df: pd.DataFrame, at: pd.Timestamp, half_life: float = 240, history: int = 1100,
           goals_weight: float = 0.6, ridge: float = 0.02) -> DCModel | None:
    """Подгоняет рейтинги атаки/защиты по матчам ДО даты `at` с экспоненциальным
    затуханием веса. Целевая метрика — смесь голов и xG-прокси."""
    d = df[(df.date < at) & (df.date >= at - pd.Timedelta(days=history))]
    if len(d) < 100:
        return None
    teams = sorted(set(d.home) | set(d.away))
    idx = {t: i for i, t in enumerate(teams)}
    hi = d.home.map(idx).values
    ai = d.away.map(idx).values
    age = (at - d.date).dt.days.values
    w = 0.5 ** (age / half_life)
    yh = goals_weight * d.hg.values + (1 - goals_weight) * d.hxg.values
    ya = goals_weight * d.ag.values + (1 - goals_weight) * d.axg.values
    n = len(teams)

    def nll(p):
        att, de, home, mu = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
        lh_log = mu + home + att[hi] - de[ai]
        la_log = mu + att[ai] - de[hi]
        lh, la = np.exp(lh_log), np.exp(la_log)
        ll = w * (yh * lh_log - lh + ya * la_log - la)
        f = -ll.sum() + ridge * w.sum() / n * (att @ att + de @ de)
        rh, ra = w * (yh - lh), w * (ya - la)
        g_att = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * w.sum() / n * att
        g_de = (np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * w.sum() / n * de
        return f, np.concatenate([g_att, g_de, [-rh.sum(), -(rh.sum() + ra.sum())]])

    p0 = np.zeros(2 * n + 2)
    p0[2 * n], p0[2 * n + 1] = 0.25, math.log(max(d.ag.mean(), 0.5))
    res = minimize(nll, p0, jac=True, method="L-BFGS-B")
    att, de = res.x[:n], res.x[n:2 * n]
    att, de = att - att.mean(), de - de.mean()
    home, mu = res.x[2 * n], res.x[2 * n + 1]

    # rho — по реальным голам при найденных λ
    lh = np.exp(mu + home + att[hi] - de[ai])
    la = np.exp(mu + att[ai] - de[hi])
    x, y = d.hg.values.astype(int), d.ag.values.astype(int)

    def rho_nll(r):
        return -(w * np.log(dc_tau(x, y, lh, la, r))).sum()

    rho = minimize_scalar(rho_nll, bounds=(-0.25, 0.15), method="bounded").x
    return DCModel(teams, att, de, float(home), float(mu), float(rho), at)


def score_matrix(lh: float, la: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def markets_from_matrix(m: np.ndarray) -> dict[str, float]:
    i, j = np.indices(m.shape)
    tot = i + j
    return {
        "p_h": float(m[i > j].sum()), "p_d": float(m[i == j].sum()), "p_a": float(m[i < j].sum()),
        "p_o25": float(m[tot > 2].sum()), "p_u25": float(m[tot <= 2].sum()),
        "p_btts": float(m[1:, 1:].sum()),
    }


def top_scores(m: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    flat = np.argsort(m, axis=None)[::-1][:n]
    return [(f"{a}:{b}", float(m[a, b])) for a, b in zip(*np.unravel_index(flat, m.shape))]
