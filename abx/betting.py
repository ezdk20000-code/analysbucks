"""Отбор ставок (ординары и экспрессы) и бэктест с реальным управлением банком.

Стратегии:
  value    — ординары: ставим, если p * лучший_кэф - 1 > min_edge.
  favs     — ординары на фаворитов (кэф в коридоре) с положительным перевесом.
  express  — экспресс из N событий с положительным перевесом, по одному на матч,
             в пределах игровой недели. Перевесы перемножаются: (1+e1)(1+e2)... - 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import cfg
from .staking import Staker

SEL = {  # рынок/исход -> (колонка вероятности, суффикс кэфа, колонка закрытия)
    ("1X2", "H"): ("p_h", "h", "cl_p_h"),
    ("1X2", "D"): ("p_d", "d", "cl_p_d"),
    ("1X2", "A"): ("p_a", "a", "cl_p_a"),
    ("OU25", "O"): ("p_o25", "o25", "cl_p_o25"),
    ("OU25", "U"): ("p_u25", "u25", "cl_p_u25"),
}
SEL_RU = {"H": "П1", "D": "X", "A": "П2", "O": "ТБ 2.5", "U": "ТМ 2.5"}


def outcome(market: str, sel: str, hg: float, ag: float) -> float | None:
    """1 — выигрыш, 0 — проигрыш, None — матч не сыгран."""
    if pd.isna(hg) or pd.isna(ag):
        return None
    if market == "1X2":
        res = "H" if hg > ag else "D" if hg == ag else "A"
        return float(res == sel)
    tot = hg + ag
    return float(tot > 2.5) if sel == "O" else float(tot < 2.5)


def candidates(pred: pd.DataFrame, bcfg: dict | None = None, price: str | None = None) -> pd.DataFrame:
    """Все возможные ставки с вероятностью, лучшим кэфом и перевесом."""
    b = bcfg or cfg()["betting"]
    price = price or b["price_source"]
    rows = []
    for (market, sel), (pcol, osuf, clcol) in SEL.items():
        if market not in b["markets"]:
            continue
        col = f"{price}_{osuf}"
        if col not in pred:
            continue
        d = pred[["match_id", "league", "season", "date", "home", "away", "hg", "ag"]].copy()
        d["market"], d["sel"] = market, sel
        d["p"] = pred[pcol]
        d["p_market"] = pred[f"mk_{pcol}"] if f"mk_{pcol}" in pred else np.nan
        d["odds"] = pred[col]
        d["avg_odds"] = pred.get(f"avg_{osuf}")
        d["p_close"] = pred.get(clcol)
        rows.append(d)
    c = pd.concat(rows, ignore_index=True).dropna(subset=["p", "odds"])
    c["edge"] = c.p * c.odds - 1
    c["win"] = [outcome(m, s, h, a) for m, s, h, a in zip(c.market, c.sel, c.hg, c.ag)]
    # CLV: насколько наш кэф лучше справедливого кэфа на закрытии линии
    c["clv"] = c.odds * c.p_close - 1
    return c


def filter_singles(c: pd.DataFrame, b: dict) -> pd.DataFrame:
    s = c[(c.edge > b["min_edge"]) & (c.edge < b["max_edge"])
          & (c.odds >= b["min_odds"]) & (c.odds <= b["max_odds"])]
    if b.get("one_bet_per_match", True):
        s = s.sort_values("edge", ascending=False).drop_duplicates("match_id")
    return s.sort_values("date")


def build_expresses(c: pd.DataFrame, e: dict) -> pd.DataFrame:
    """Экспрессы по игровым неделям: берём N лучших по перевесу исходов из разных матчей."""
    legs = c[(c.edge > e["min_leg_edge"]) & (c.odds >= e["leg_min_odds"])
             & (c.odds <= e["leg_max_odds"]) & c.market.isin(e.get("markets", ["1X2"]))]
    legs = legs.sort_values("edge", ascending=False).drop_duplicates("match_id")
    legs = legs.assign(week=legs.date.dt.to_period("W-MON"))
    out = []
    n = e["legs"]
    for wk, g in legs.groupby("week"):
        g = g.nlargest(n * e.get("max_per_week", 1), "edge")
        for k in range(len(g) // n):
            pick = g.iloc[k * n:(k + 1) * n]
            p = float(np.prod(pick.p))
            odds = float(np.prod(pick.odds))
            wins = pick.win.tolist()
            win = None if any(w is None or pd.isna(w) for w in wins) else float(all(wins))
            out.append({
                "match_id": "+".join(pick.match_id), "date": pick.date.min(),
                "settle_date": pick.date.max(), "league": "/".join(sorted(set(pick.league))),
                "season": pick.season.iloc[0], "market": f"EXPRESS{n}",
                "sel": " | ".join(f"{h}-{a}: {SEL_RU[s]} @{o:.2f}" for h, a, s, o in
                                  zip(pick.home, pick.away, pick.sel, pick.odds)),
                "p": p, "odds": odds, "edge": p * odds - 1, "win": win, "legs": n,
                "clv": float(np.prod(pick.odds * pick.p_close) - 1) if pick.p_close.notna().all() else np.nan,
            })
    return pd.DataFrame(out)


# ------------------------------------------------------------------ симуляция


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    curve: pd.DataFrame
    summary: dict


def simulate(bets: pd.DataFrame, start_bank: float, scfg: dict) -> BacktestResult:
    """Хронологическая симуляция: ставки одного дня размещаются одновременно от текущего
    банка (с лимитом дневной нагрузки), расчёт — в день последнего события."""
    bets = bets.dropna(subset=["win"]).copy()
    if "settle_date" not in bets:
        bets["settle_date"] = bets["date"]
    bets["settle_date"] = bets["settle_date"].fillna(bets["date"])
    if "legs" not in bets:
        bets["legs"] = 1
    bets["legs"] = bets["legs"].fillna(1).astype(int)
    bets = bets.sort_values("date").reset_index(drop=True)
    staker = Staker(scfg, start_bank)
    bank = start_bank
    open_bets: list[tuple[pd.Timestamp, float, float]] = []  # (settle, stake, payout)
    stakes = np.zeros(len(bets))
    curve = []
    days = bets.date.dt.normalize()
    for day in sorted(days.unique()):
        # расчёт закрытых ставок
        still = []
        for sd, st, pay in open_bets:
            if sd.normalize() < day:
                bank += pay
            else:
                still.append((sd, st, pay))
        open_bets = still
        staker.update_peak(bank + sum(st for _, st, _ in open_bets))
        idx = bets.index[days == day]
        plans = [staker.size(bets.at[i, "p"], bets.at[i, "odds"], bank, bets.at[i, "legs"]).stake
                 for i in idx]
        plans = staker.apply_exposure_cap(plans, bank)
        for i, st in zip(idx, plans):
            if st <= 0 or st > bank:
                continue
            bank -= st
            stakes[i] = st
            pay = st * bets.at[i, "odds"] if bets.at[i, "win"] == 1 else 0.0
            open_bets.append((bets.at[i, "settle_date"], st, pay))
        curve.append({"date": day, "bank": bank + sum(st for _, st, _ in open_bets)})
    for _, st, pay in open_bets:
        bank += pay
    if curve:
        curve.append({"date": curve[-1]["date"] + pd.Timedelta(days=1), "bank": bank})
    bets["stake"] = stakes
    bets["pnl"] = np.where(bets.win == 1, bets.stake * (bets.odds - 1), -bets.stake)
    placed = bets[bets.stake > 0]
    cv = pd.DataFrame(curve)
    return BacktestResult(placed, cv, summarize(placed, cv, start_bank))


def summarize(placed: pd.DataFrame, curve: pd.DataFrame, start_bank: float) -> dict:
    if placed.empty:
        return {"ставок": 0}
    peak = curve.bank.cummax()
    dd = (1 - curve.bank / peak).max()
    flat_roi = float(np.where(placed.win == 1, placed.odds - 1, -1).mean())
    years = max((curve.date.max() - curve.date.min()).days / 365.25, 1e-9)
    final = float(curve.bank.iloc[-1])
    return {
        "ставок": int(len(placed)),
        "выигрышей %": round(100 * placed.win.mean(), 1),
        "средний кэф": round(float(placed.odds.mean()), 2),
        "оборот": round(float(placed.stake.sum()), 2),
        "прибыль": round(float(placed.pnl.sum()), 2),
        "ROI (доходность на оборот) %": round(100 * placed.pnl.sum() / placed.stake.sum(), 2),
        "ROI флэтом %": round(100 * flat_roi, 2),
        "старт банка": start_bank,
        "итог банка": round(final, 2),
        "рост банка, раз": round(final / start_bank, 3),
        "среднегодовой рост %": round(100 * ((final / start_bank) ** (1 / years) - 1), 2) if final > 0 else -100.0,
        "макс. просадка %": round(100 * float(dd), 1),
        "средний CLV %": round(100 * float(placed.clv.mean()), 2) if placed.clv.notna().any() else None,
    }


def monte_carlo(placed: pd.DataFrame, scfg: dict, start_bank: float, runs: int = 2000,
                seed: int = 7) -> dict:
    """Перемешиваем порядок ставок (бутстрэп) и смотрим разброс исходов при той же
    стратегии ставок. Показывает, насколько результат зависит от удачи."""
    if placed.empty:
        return {}
    rng = np.random.default_rng(seed)
    p, o, w = placed.p.values, placed.odds.values, placed.win.values
    legs = placed.legs.values if "legs" in placed else np.ones(len(p), int)
    st = Staker(scfg, start_bank)
    finals, dds = [], []
    for _ in range(runs):
        idx = rng.integers(0, len(p), len(p))
        bank, peak, mdd = start_bank, start_bank, 0.0
        st.peak = start_bank
        for i in idx:
            stake = st.size(p[i], o[i], bank, int(legs[i])).stake
            if stake <= 0 or stake > bank:
                continue
            bank += stake * (o[i] - 1) if w[i] == 1 else -stake
            peak = max(peak, bank)
            st.update_peak(bank)
            mdd = max(mdd, 1 - bank / peak)
        finals.append(bank / start_bank)
        dds.append(mdd)
    finals, dds = np.array(finals), np.array(dds)
    return {
        "медиана роста банка, раз": round(float(np.median(finals)), 3),
        "5% худших: рост, раз": round(float(np.percentile(finals, 5)), 3),
        "5% лучших: рост, раз": round(float(np.percentile(finals, 95)), 3),
        "вероятность закончить в минусе %": round(100 * float((finals < 1).mean()), 1),
        "вероятность просадки >30% %": round(100 * float((dds > 0.3).mean()), 1),
        "вероятность потерять половину банка %": round(100 * float((finals < 0.5).mean()), 1),
    }
