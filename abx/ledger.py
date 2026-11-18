"""Учёт ставок и банка (SQLite, файл data/ledger.db).

Хранит: движения банка (депозит/вывод/корректировка), все ставки (ординары и
экспрессы), их статус, выплату, прибыль и CLV. Ставки рассчитываются автоматически
по результатам football-data или вручную.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .config import LEDGER_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS bank_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,            -- deposit | withdraw | adjust
    amount REAL NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'single',  -- single | express
    mode TEXT NOT NULL DEFAULT 'paper',   -- paper (виртуально) | real
    bookmaker TEXT,
    legs_json TEXT NOT NULL,       -- список событий: match_id, league, date, home, away, market, sel, odds, p
    odds REAL NOT NULL,            -- итоговый кэф (для экспресса — произведение)
    stake REAL NOT NULL,
    p_model REAL,                  -- наша вероятность прохода
    edge REAL,                     -- ожидаемый перевес p*k-1
    status TEXT NOT NULL DEFAULT 'open',  -- open | won | lost | void
    payout REAL,
    pnl REAL,
    clv REAL,
    settled_at TEXT,
    note TEXT
);
"""


@contextmanager
def conn(path=None):
    c = sqlite3.connect(path or LEDGER_DB)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ банк


def bank_event(kind: str, amount: float, note: str = "", path=None) -> None:
    if kind not in ("deposit", "withdraw", "adjust"):
        raise ValueError("kind: deposit | withdraw | adjust")
    with conn(path) as c:
        c.execute("INSERT INTO bank_events (ts, kind, amount, note) VALUES (?,?,?,?)",
                  (_now(), kind, float(amount), note))


def balance(mode: str | None = None, path=None) -> dict:
    """Банк = депозиты - выводы + корректировки + прибыль рассчитанных - открытые ставки."""
    with conn(path) as c:
        ev = c.execute("SELECT kind, SUM(amount) s FROM bank_events GROUP BY kind").fetchall()
        q = "SELECT status, SUM(stake) st, SUM(pnl) pnl, COUNT(*) n FROM bets"
        args: tuple = ()
        if mode:
            q += " WHERE mode=?"
            args = (mode,)
        bets = c.execute(q + " GROUP BY status", args).fetchall()
    e = {r["kind"]: r["s"] or 0 for r in ev}
    capital = e.get("deposit", 0) - e.get("withdraw", 0) + e.get("adjust", 0)
    settled_pnl = sum((r["pnl"] or 0) for r in bets if r["status"] != "open")
    open_stake = sum((r["st"] or 0) for r in bets if r["status"] == "open")
    n_open = sum(r["n"] for r in bets if r["status"] == "open")
    total = capital + settled_pnl
    return {"капитал": round(capital, 2), "прибыль": round(settled_pnl, 2),
            "банк": round(total, 2), "в игре": round(open_stake, 2),
            "свободно": round(total - open_stake, 2), "открытых ставок": n_open}


# ------------------------------------------------------------------ ставки


def place_bet(legs: list[dict], stake: float, mode: str = "paper", bookmaker: str = "",
              note: str = "", odds: float | None = None, path=None) -> int:
    """legs — список событий. Для ординара — один элемент.
    Обязательные поля события: home, away, market (1X2|OU25), sel (H|D|A|O|U), odds.
    Желательно: match_id, league, date, p (наша вероятность)."""
    if not legs:
        raise ValueError("Пустая ставка")
    total_odds = odds or float(pd.Series([l["odds"] for l in legs]).prod())
    p = None
    if all(l.get("p") is not None for l in legs):
        p = float(pd.Series([l["p"] for l in legs]).prod())
    edge = p * total_odds - 1 if p else None
    for l in legs:
        if isinstance(l.get("date"), (pd.Timestamp, datetime)):
            l["date"] = pd.Timestamp(l["date"]).isoformat()
    with conn(path) as c:
        cur = c.execute(
            "INSERT INTO bets (created_at, kind, mode, bookmaker, legs_json, odds, stake, p_model, edge, note)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_now(), "express" if len(legs) > 1 else "single", mode, bookmaker,
             json.dumps(legs, ensure_ascii=False, default=str), total_odds, float(stake), p, edge, note))
        return int(cur.lastrowid)


def settle_bet(bet_id: int, status: str, payout: float | None = None, path=None) -> None:
    """Ручной расчёт: status = won | lost | void (возврат)."""
    with conn(path) as c:
        row = c.execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone()
        if row is None:
            raise KeyError(f"Ставка {bet_id} не найдена")
        stake, odds = row["stake"], row["odds"]
        if payout is None:
            payout = stake * odds if status == "won" else stake if status == "void" else 0.0
        c.execute("UPDATE bets SET status=?, payout=?, pnl=?, settled_at=? WHERE id=?",
                  (status, payout, payout - stake, _now(), bet_id))


def delete_bet(bet_id: int, path=None) -> None:
    with conn(path) as c:
        c.execute("DELETE FROM bets WHERE id=?", (bet_id,))


def auto_settle(results: pd.DataFrame, closing: pd.DataFrame | None = None, path=None) -> list[int]:
    """Рассчитывает открытые ставки по сыгранным матчам. results — load_matches().
    closing — таблица прогнозов с колонками cl_p_* для расчёта CLV (необязательно)."""
    from .betting import SEL, outcome

    res = results.set_index("match_id")
    by_teams = results.assign(d=results.date.dt.strftime("%Y-%m-%d")).set_index(["home", "away", "d"])
    settled = []
    with conn(path) as c:
        rows = c.execute("SELECT * FROM bets WHERE status='open'").fetchall()
    for r in rows:
        legs = json.loads(r["legs_json"])
        outs, clv_prod = [], 1.0
        for l in legs:
            mid = l.get("match_id")
            m = None
            if mid and mid in res.index:
                m = res.loc[mid]
            else:
                key = (l["home"], l["away"], str(l.get("date", ""))[:10])
                if key in by_teams.index:
                    m = by_teams.loc[key]
            if isinstance(m, pd.DataFrame):
                m = m.iloc[0]
            if m is None:
                outs.append(None)
                continue
            outs.append(outcome(l["market"], l["sel"], m["hg"], m["ag"]))
            pc = SEL[(l["market"], l["sel"])][2]
            if closing is not None and mid in closing.index and pd.notna(closing.at[mid, pc]):
                clv_prod *= l["odds"] * closing.at[mid, pc]
            else:
                clv_prod = float("nan")
        if any(o == 0 for o in outs):
            status = "lost"
        elif outs and all(o == 1 for o in outs):
            status = "won"
        else:
            continue
        settle_bet(r["id"], status, path=path)
        if clv_prod == clv_prod:  # не NaN
            with conn(path) as c:
                c.execute("UPDATE bets SET clv=? WHERE id=?", (clv_prod - 1, r["id"]))
        settled.append(r["id"])
    return settled


def bets_df(status: str | None = None, path=None) -> pd.DataFrame:
    with conn(path) as c:
        q = "SELECT * FROM bets" + (" WHERE status=?" if status else "") + " ORDER BY id DESC"
        df = pd.read_sql_query(q, c, params=(status,) if status else None)
    if df.empty:
        return df
    df["legs"] = df.legs_json.map(json.loads)
    df["описание"] = df.legs.map(describe_legs)
    return df


def describe_legs(legs: list[dict]) -> str:
    from .betting import SEL_RU
    return " | ".join(f"{l['home']}–{l['away']}: {SEL_RU.get(l['sel'], l['sel'])} @{l['odds']:.2f}"
                      for l in legs)


def stats(path=None) -> dict:
    df = bets_df(path=path)
    if df.empty:
        return {"ставок": 0}
    s = df[df.status.isin(["won", "lost"])]
    out = {"всего ставок": len(df), "рассчитано": len(s), "открыто": int((df.status == "open").sum())}
    if len(s):
        out.update({
            "выигрышей %": round(100 * (s.status == "won").mean(), 1),
            "оборот": round(s.stake.sum(), 2),
            "прибыль": round(s.pnl.sum(), 2),
            "ROI %": round(100 * s.pnl.sum() / s.stake.sum(), 2),
            "средний кэф": round(s.odds.mean(), 2),
            "средний CLV %": round(100 * s.clv.mean(), 2) if s.clv.notna().any() else None,
            "ожидаемая прибыль (по модели)": round((s.stake * s.edge.fillna(0)).sum(), 2),
        })
        for kind, g in s.groupby("kind"):
            out[f"ROI {kind} %"] = round(100 * g.pnl.sum() / g.stake.sum(), 2)
    return out


def bank_history(path=None) -> pd.DataFrame:
    """Динамика банка по времени (для графика)."""
    with conn(path) as c:
        ev = pd.read_sql_query("SELECT ts, amount * CASE kind WHEN 'withdraw' THEN -1 ELSE 1 END AS delta "
                               "FROM bank_events", c)
        bt = pd.read_sql_query("SELECT settled_at AS ts, pnl AS delta FROM bets WHERE status!='open'", c)
    h = pd.concat([ev, bt]).dropna()
    if h.empty:
        return pd.DataFrame(columns=["ts", "bank"])
    h["ts"] = pd.to_datetime(h.ts)
    h = h.sort_values("ts")
    h["bank"] = h.delta.cumsum()
    return h[["ts", "bank"]]
