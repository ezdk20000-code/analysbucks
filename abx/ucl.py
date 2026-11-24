"""Лига чемпионов: прогноз матчей тура.

Почему здесь нет «своей» модели силы, как для АПЛ/Ла Лиги: в ЛЧ играют клубы из 15+
лиг, и сравнивать их между собой можно только по еврокубковым матчам, которых мало.
Наше исследование показало, что рынок точнее модели даже внутри одной лиги, поэтому
основа прогноза — консенсус рынка без маржи (несколько букмекеров + биржа Kalshi).

Сверху система добавляет то, что рынок может недооценивать, и показывает это флагами:
  • отдых перед матчем и следующий матч в чемпионате (риск ротации);
  • форма во внутреннем чемпионате (очки, разница мячей, удары в створ);
  • положение в таблице ЛЧ: кому матч важнее (топ-8 / плей-офф / вылет).

Если вписать в data/ucl/my_odds.csv кэфы своего букмекера, система посчитает перевес
и размер ставки по Келли — так же, как для АПЛ/Ла Лиги.
"""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from . import data
from .config import DATA, RAW, cfg
from .odds import fair_power
from .staking import Staker

UCL = DATA / "ucl"
UCL.mkdir(exist_ok=True)
SEASON = "2627"
ARTICLE = ("https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000"
           "--2026-27-champions-league-all-the-league-phase-fixtures-a/")

# Название UEFA -> (файл football-data, название в нём). None — бесплатных данных нет.
TEAMS: dict[str, tuple[str, str] | None] = {
    "Arsenal": ("E0", "Arsenal"), "Aston Villa": ("E0", "Aston Villa"),
    "Manchester City": ("E0", "Man City"), "Liverpool": ("E0", "Liverpool"),
    "Manchester United": ("E0", "Man United"),
    "Real Madrid": ("SP1", "Real Madrid"), "Barcelona": ("SP1", "Barcelona"),
    "Atlético de Madrid": ("SP1", "Ath Madrid"), "Villarreal": ("SP1", "Villarreal"),
    "Real Betis": ("SP1", "Betis"),
    "Borussia Dortmund": ("D1", "Dortmund"), "Bayern München": ("D1", "Bayern Munich"),
    "Leipzig": ("D1", "RB Leipzig"), "Stuttgart": ("D1", "Stuttgart"),
    "Napoli": ("I1", "Napoli"), "Inter": ("I1", "Inter"), "Roma": ("I1", "Roma"),
    "Como": ("I1", "Como"),
    "Lens": ("F1", "Lens"), "Lille": ("F1", "Lille"), "Paris Saint-Germain": ("F1", "Paris SG"),
    "Feyenoord": ("N1", "Feyenoord"), "PSV Eindhoven": ("N1", "PSV Eindhoven"),
    "Porto": ("P1", "Porto"), "Sporting CP": ("P1", "Sp Lisbon"),
    "Club Brugge": ("B1", "Club Brugge"),
    "Galatasaray": ("T1", "Galatasaray"), "Fenerbahçe": ("T1", "Fenerbahce"),
    "AEK Athens": ("G1", "AEK"),
    "LASK": ("AUT", "LASK"),
    "Bodø/Glimt": ("NOR", "Bodo/Glimt"), "Viking": ("NOR", "Viking"),
    "Slavia Praha": None, "Slovan Bratislava": None, "Sabah": None, "Shakhtar Donetsk": None,
}
DOMESTIC = ["E0", "SP1", "D1", "I1", "F1", "N1", "P1", "B1", "T1", "G1"]
EXTRA = ["AUT", "NOR"]  # «новые» лиги football-data (один файл на все сезоны)


# ------------------------------------------------------------------ данные UEFA


def parse_uefa_article(page: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Из статьи UEFA «все матчи общего этапа»: (расписание, результаты)."""
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    t = _html.unescape(re.sub(r"<[^>]+>", "\n", t))
    t = t.replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", l).strip() for l in t.split("\n")]
    lines = [l for l in lines if l and not l.startswith("©")]
    fixtures, results = [], []
    md, day, section = None, None, None
    date_re = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) (\d+ \w+ \d{4})$")
    for i, l in enumerate(lines):
        if l == "Fixtures":
            section = "fx"
        elif l == "Results":
            section = "res"
        m = re.match(r"^Matchday (\d)$", l)
        if m:
            md = int(m.group(1))
            continue
        m = date_re.match(l)
        if m:
            day = pd.to_datetime(m.group(2), format="%d %B %Y")
            continue
        if section is None or md is None or day is None:
            continue
        m = re.match(r"^(.+?) (\d+)-(\d+) (.+)$", l)
        if section == "res" and m and not l.startswith(("Highlights", "Watch")):
            results.append({"md": md, "date": day, "home": m.group(1), "away": m.group(4),
                            "hg": int(m.group(2)), "ag": int(m.group(3))})
            continue
        m = re.match(r"^(.+?) vs (.+)$", l)
        if section == "fx" and m and ":" not in l:
            ko = "21:00"
            if i + 1 < len(lines) and re.match(r"^\(\d\d:\d\d\)$", lines[i + 1]):
                ko = lines[i + 1][1:-1]
            fixtures.append({"md": md, "date": day + pd.Timedelta(ko + ":00"),
                             "home": m.group(1), "away": m.group(2)})
    return pd.DataFrame(fixtures), pd.DataFrame(results).drop_duplicates()


def update_uefa() -> tuple[int, int]:
    r = requests.get(ARTICLE, timeout=60, headers={"User-Agent": "analysbucks/0.1"})
    r.raise_for_status()
    fx, res = parse_uefa_article(r.content.decode("utf-8", errors="replace"))
    fx.to_csv(UCL / f"{SEASON}_fixtures.csv", index=False)
    res.to_csv(UCL / f"{SEASON}_results.csv", index=False)
    return len(fx), len(res)


def update_domestic() -> None:
    """Свежие внутренние чемпионаты участников ЛЧ + ближайшие матчи (fixtures.csv)."""
    for d in DOMESTIC:
        (RAW / f"{d}_{SEASON}.csv").write_bytes(data._get(data.BASE.format(season=SEASON, div=d)))
    for n in EXTRA:
        (RAW / f"new_{n}.csv").write_bytes(data._get(f"https://www.football-data.co.uk/new/{n}.csv"))
    (RAW / "fixtures.csv").write_bytes(data._get(data.FIXTURES))


def load_fixtures() -> pd.DataFrame:
    return pd.read_csv(UCL / f"{SEASON}_fixtures.csv", parse_dates=["date"])


def load_results() -> pd.DataFrame:
    return pd.read_csv(UCL / f"{SEASON}_results.csv", parse_dates=["date"])


# ------------------------------------------------------------------ таблица ЛЧ


def ucl_table(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in res.iterrows():
        for t, gf, ga in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            rows.append({"team": t, "gp": 1, "w": int(gf > ga), "d": int(gf == ga), "l": int(gf < ga),
                         "gf": gf, "ga": ga, "pts": 3 if gf > ga else 1 if gf == ga else 0})
    t = pd.DataFrame(rows).groupby("team").sum()
    t["gd"] = t.gf - t.ga
    t = t.sort_values(["pts", "gd", "gf"], ascending=False)
    t["pos"] = range(1, len(t) + 1)
    return t


def ucl_stake(pos: int, pts: int, played: int, total: int = 8) -> str:
    """Что на кону: по очкам после N туров (ориентиры: ~16 очков — топ-8, ~10 — топ-24)."""
    left = total - played
    if pts + 3 * left < 10:
        return "почти без шансов на плей-офф"
    if pos <= 8:
        return "борется за топ-8 (прямой выход в 1/8)"
    if pos <= 24:
        return "в зоне плей-офф, нужны очки"
    return "под угрозой вылета — матч важнее"


# ------------------------------------------------------------------ внутренние чемпионаты


def domestic_matches() -> pd.DataFrame:
    frames = []
    for d in DOMESTIC:
        p = RAW / f"{d}_{SEASON}.csv"
        if not p.exists():
            continue
        x = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace")
        frames.append(pd.DataFrame({
            "div": d, "date": pd.to_datetime(x.Date, dayfirst=True, format="mixed"),
            "home": x.HomeTeam, "away": x.AwayTeam, "hg": x.FTHG, "ag": x.FTAG,
            "hst": x.get("HST"), "ast": x.get("AST")}))
    for n in EXTRA:
        p = RAW / f"new_{n}.csv"
        if not p.exists():
            continue
        x = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace")
        x = x[pd.to_datetime(x.Date, dayfirst=True, format="mixed") >= "2026-01-01"]
        frames.append(pd.DataFrame({
            "div": n, "date": pd.to_datetime(x.Date, dayfirst=True, format="mixed"),
            "home": x.Home, "away": x.Away, "hg": x.HG, "ag": x.AG, "hst": np.nan, "ast": np.nan,
            "season": x.Season.astype(str)}))
    return pd.concat(frames, ignore_index=True).dropna(subset=["hg"]).sort_values("date")


def next_domestic(team: tuple[str, str], after: pd.Timestamp) -> pd.Timestamp | None:
    p = RAW / "fixtures.csv"
    if not p.exists():
        return None
    fx = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace")
    fx = fx[(fx.Div == team[0]) & ((fx.HomeTeam == team[1]) | (fx.AwayTeam == team[1]))]
    d = pd.to_datetime(fx.Date, dayfirst=True, format="mixed")
    d = d[d > after.normalize()]
    return d.min() if len(d) else None


def team_profile(uefa_name: str, kickoff: pd.Timestamp, dom: pd.DataFrame) -> dict:
    key = TEAMS.get(uefa_name)
    out = {"данные": "нет (лига не покрыта бесплатными источниками)"}
    if key is None:
        return out
    div, name = key
    g = dom[(dom["div"] == div) & ((dom.home == name) | (dom.away == name)) & (dom.date < kickoff.normalize())]
    if g.empty:
        return out
    if "season" in g and g.season.notna().any():  # AUT/NOR — только текущий сезон
        g = g[g.season == g.season.dropna().iloc[-1]]
    last = g.tail(5)
    pts, gd, sot = [], [], []
    for _, r in last.iterrows():
        home = r.home == name
        gf, ga = (r.hg, r.ag) if home else (r.ag, r.hg)
        pts.append(3 if gf > ga else 1 if gf == ga else 0)
        gd.append(gf - ga)
        if pd.notna(r.hst):
            sot.append((r.hst - r.ast) if home else (r.ast - r.hst))
    # таблица чемпионата
    lg = dom[(dom["div"] == div) & (dom.date >= g.date.min() - pd.Timedelta(days=1))]
    if "season" in g and g.season.notna().any():
        lg = dom[(dom["div"] == div) & (dom.season == g.season.iloc[-1])]
    tab: dict[str, int] = {}
    for _, r in lg.iterrows():
        tab[r.home] = tab.get(r.home, 0) + (3 if r.hg > r.ag else 1 if r.hg == r.ag else 0)
        tab[r.away] = tab.get(r.away, 0) + (3 if r.ag > r.hg else 1 if r.hg == r.ag else 0)
    order = sorted(tab, key=lambda t: -tab[t])
    last_date = g.date.max()
    league_last = dom[dom["div"] == div].date.max()
    stale = (kickoff.normalize() - league_last).days > 7  # файл лиги давно не обновлялся
    nxt = next_domestic(key, kickoff)
    return {
        "данные": f"{div}, по {last_date:%d.%m}" + (" (данные лиги устарели — отдых неизвестен)" if stale else ""),
        "место в лиге": order.index(name) + 1 if name in order else None,
        "форма (очки за 5)": int(sum(pts)),
        "форма": "".join("В" if p == 3 else "Н" if p == 1 else "П" for p in pts),
        "мячи ±5": int(sum(gd)),
        "удары в створ ±5": int(sum(sot)) if sot else None,
        "отдых, дней": None if stale else int((kickoff.normalize() - last_date.normalize()).days),
        "след. матч в лиге через, дней": int((nxt - kickoff.normalize()).days) if nxt is not None else None,
    }


# ------------------------------------------------------------------ рынок


@dataclass
class MarketRef:
    """Историческая таблица «кэф/вероятность одной стороны -> полный расклад 1X2»,
    чтобы восстановить недостающие исходы, когда известна цена только на одну команду."""
    hist: pd.DataFrame

    @classmethod
    def build(cls) -> MarketRef:
        m = data.load_matches()
        m = m[m.avg_h.notna() & m.avg_d.notna() & m.avg_a.notna()]
        P = fair_power(m[["avg_h", "avg_d", "avg_a"]].values)
        return cls(pd.DataFrame({"oh": m.avg_h.values, "oa": m.avg_a.values,
                                 "ph": P[:, 0], "pd": P[:, 1], "pa": P[:, 2]}).dropna())

    def from_side(self, side: str, odds: float | None = None, prob: float | None = None) -> np.ndarray:
        h = self.hist
        if odds is not None:
            col = "oh" if side == "h" else "oa"
            near = h.iloc[(np.log(h[col]) - np.log(odds)).abs().argsort()[:150]]
        else:
            col = {"h": "ph", "a": "pa"}[side]
            near = h.iloc[(h[col] - prob).abs().argsort()[:150]]
        p = near[["ph", "pd", "pa"]].median().to_numpy(copy=True)
        if prob is not None:  # точная вероятность известна — подгоняем остальные пропорционально
            i = 0 if side == "h" else 2
            rest = [j for j in range(3) if j != i]
            p[rest] = p[rest] / p[rest].sum() * (1 - prob)
            p[i] = prob
        return p / p.sum()


def market_consensus(inputs: pd.DataFrame, ref: MarketRef) -> pd.DataFrame:
    """inputs: home, away, odds_h, odds_d, odds_a, p_h, p_a, source (несколько строк на матч)."""
    out = []
    for (h, a), g in inputs.groupby(["home", "away"], sort=False):
        ests, srcs = [], []
        for _, r in g.iterrows():
            o = [r.get("odds_h"), r.get("odds_d"), r.get("odds_a")]
            if all(pd.notna(x) for x in o):
                ests.append(fair_power(np.array(o, float)))
            elif pd.notna(r.get("p_h")):
                ests.append(ref.from_side("h", prob=r.p_h))
            elif pd.notna(r.get("p_a")):
                ests.append(ref.from_side("a", prob=r.p_a))
            elif pd.notna(r.get("odds_h")):
                ests.append(ref.from_side("h", odds=r.odds_h))
            elif pd.notna(r.get("odds_a")):
                ests.append(ref.from_side("a", odds=r.odds_a))
            else:
                continue
            srcs.append(str(r.source))
        if not ests:
            continue
        p = np.mean(ests, axis=0)
        spread = float(np.max([e[0] for e in ests]) - np.min([e[0] for e in ests])) if len(ests) > 1 else np.nan
        out.append({"home": h, "away": a, "p_h": p[0], "p_d": p[1], "p_a": p[2],
                    "источников": len(ests), "разброс П1": spread, "источники": "; ".join(srcs)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ прогноз тура


def verdict(p_h: float, p_d: float, p_a: float) -> tuple[str, str]:
    """Исход и уровень уверенности (по вероятности фаворита)."""
    fav, pf = ("П1", p_h) if p_h >= p_a else ("П2", p_a)
    if pf >= 0.80:
        return fav, "очень высокая"
    if pf >= 0.65:
        return fav, "высокая"
    if pf >= 0.50:
        return fav, "средняя"
    dc = "1X" if p_h >= p_a else "X2"
    if pf >= 0.40:
        return f"{fav} (безопаснее {dc}: {p_d + pf:.0%})", "низкая"
    return f"равный матч, {dc} {p_d + pf:.0%}", "низкая"


def flags(hp: dict, ap: dict, hs: str, as_: str) -> list[str]:
    f = []
    rh, ra = hp.get("отдых, дней"), ap.get("отдых, дней")
    if rh is not None and ra is not None and abs(rh - ra) >= 2 and min(rh, ra) <= 3:
        f.append(f"отдых {rh} vs {ra} дн.")
    for name, p in (("хозяева", hp), ("гости", ap)):
        n = p.get("след. матч в лиге через, дней")
        if n is not None and n <= 3:
            f.append(f"{name}: матч в лиге через {n} дн. — возможна ротация")
        r = p.get("отдых, дней")
        if r is not None and r > 12:
            f.append(f"{name}: без игр {r} дн. (межсезонье/пауза — риск «ржавчины»)")
        fo = p.get("форма (очки за 5)")
        if fo is not None and fo <= 4:
            f.append(f"{name}: слабая форма в лиге ({p.get('форма')})")
        if fo is not None and fo >= 13:
            f.append(f"{name}: отличная форма в лиге ({p.get('форма')})")
    if "вылета" in hs and "топ-8" in as_:
        f.append("хозяевам матч важнее")
    if "вылета" in as_ and "топ-8" in hs:
        f.append("гостям матч важнее")
    return f


def forecast(md: int | None = None, inputs_path=None) -> dict:
    fx, res = load_fixtures(), load_results()
    md = md or int(fx.md.min())
    fx = fx[fx.md == md]
    tab = ucl_table(res)
    played = int(res.md.max())
    inputs = pd.read_csv(inputs_path or UCL / f"{SEASON}_md{md}_market.csv")
    mk = market_consensus(inputs, MarketRef.build())
    dom = domestic_matches()
    rows = []
    for _, r in fx.iterrows():
        m = mk[(mk.home == r.home) & (mk.away == r.away)]
        hp, ap = team_profile(r.home, r.date, dom), team_profile(r.away, r.date, dom)
        th = tab.loc[r.home] if r.home in tab.index else None
        ta = tab.loc[r.away] if r.away in tab.index else None
        hs = ucl_stake(int(th.pos), int(th.pts), played) if th is not None else ""
        as_ = ucl_stake(int(ta.pos), int(ta.pts), played) if ta is not None else ""
        row = {"date": r.date, "home": r.home, "away": r.away,
               "ucl_h": f"{int(th.pos)}-е, {int(th.pts)} оч." if th is not None else "",
               "ucl_a": f"{int(ta.pos)}-е, {int(ta.pts)} оч." if ta is not None else "",
               "stake_h": hs, "stake_a": as_, "prof_h": hp, "prof_a": ap}
        if len(m):
            m = m.iloc[0]
            v, conf = verdict(m.p_h, m.p_d, m.p_a)
            row.update({"p_h": m.p_h, "p_d": m.p_d, "p_a": m.p_a, "прогноз": v, "уверенность": conf,
                        "источников": m["источников"], "источники": m["источники"]})
        row["флаги"] = flags(hp, ap, hs, as_)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return {"md": md, "matches": out, "table": tab, "value": value_bets(out)}


def value_bets(fc: pd.DataFrame) -> pd.DataFrame:
    """Если есть data/ucl/my_odds.csv (home,away,sel[H|D|A],odds[,bookmaker]) — считаем перевес."""
    p = UCL / "my_odds.csv"
    if not p.exists() or fc.empty:
        return pd.DataFrame()
    my = pd.read_csv(p)
    st = Staker(cfg()["staking"], 1000.0)
    from .service import current_bank
    bank = current_bank()
    rows = []
    for _, o in my.iterrows():
        m = fc[(fc.home == o.home) & (fc.away == o.away)]
        if m.empty or "p_h" not in m:
            continue
        pr = float(m.iloc[0][{"H": "p_h", "D": "p_d", "A": "p_a"}[o.sel]])
        edge = pr * o.odds - 1
        rows.append({"матч": f"{o.home} – {o.away}", "исход": {"H": "П1", "D": "X", "A": "П2"}[o.sel],
                     "кэф": o.odds, "p": round(pr, 3), "перевес": round(edge, 3),
                     "ставка": st.size(pr, o.odds, bank).stake if edge > cfg()["betting"]["min_edge"] else 0.0,
                     "букмекер": o.get("bookmaker", "")})
    return pd.DataFrame(rows)


def report_text(fc: dict, html: bool = False) -> str:
    """Текстовый отчёт по туру (для консоли и Telegram)."""
    b = (lambda x: f"<b>{x}</b>") if html else (lambda x: x)
    m = fc["matches"]
    lines = [b(f"Лига чемпионов, тур {fc['md']} — прогноз"), ""]
    for day, g in m.groupby(m.date.dt.date):
        lines.append(b(pd.Timestamp(day).strftime("%d.%m")))
        for _, x in g.iterrows():
            if pd.isna(x.get("p_h")):
                lines.append(f"  {x.home} – {x.away}: нет данных рынка")
                continue
            lines.append(f"  {x.date:%H:%M} {x.home} – {x.away}: П1 {x.p_h:.0%} · X {x.p_d:.0%} · "
                         f"П2 {x.p_a:.0%} → {b(x['прогноз'])} [{x['уверенность']}]")
            for f in x["флаги"]:
                lines.append(f"      • {f}")
        lines.append("")
    strong = m[(m[["p_h", "p_a"]].max(axis=1) >= 0.75)].copy()
    if len(strong) >= 2:
        strong["p"] = strong[["p_h", "p_a"]].max(axis=1)
        strong = strong.sort_values("p", ascending=False).head(3)
        pr = float(strong.p.prod())
        names = ", ".join(f"{x.home if x.p_h >= x.p_a else x.away} ({x.p:.0%})" for _, x in strong.iterrows())
        lines.append(b("Самые надёжные фавориты: ") + names)
        lines.append(f"  Экспресс из них проходит с вероятностью ~{pr:.0%}; справедливый кэф {1 / pr:.2f}. "
                     "Ставить, только если у букмекера кэф выше справедливого.")
    if not fc["value"].empty:
        lines.append("")
        lines.append(b("Перевес по вашим кэфам (data/ucl/my_odds.csv):"))
        for _, v in fc["value"].iterrows():
            lines.append(f"  {v['матч']}: {v['исход']} @{v['кэф']} — перевес {v['перевес']:+.1%}, ставка {v['ставка']}")
    return "\n".join(lines)
