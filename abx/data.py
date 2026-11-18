"""Загрузка и нормализация данных football-data.co.uk.

Источник бесплатный: результаты, статистика матчей (удары, удары в створ, угловые,
карточки), коэффициенты букмекеров на открытии и закрытии линии.
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import requests

from .config import MANUAL, RAW, cfg, season_code, seasons

log = logging.getLogger(__name__)

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
UA = {"User-Agent": "analysbucks/0.1 (football analytics)"}


def _get(url: str, tries: int = 4) -> bytes:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:  # pragma: no cover - сеть
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"Не удалось скачать {url}: {last}")


def download(force_current: bool = True) -> None:
    """Скачивает все сезоны всех лиг. Прошлые сезоны — один раз, текущий — всегда."""
    cur = season_code()
    for league, div in cfg()["leagues"].items():
        for s in seasons():
            path = RAW / f"{div}_{s}.csv"
            if path.exists() and not (force_current and s == cur):
                continue
            try:
                path.write_bytes(_get(BASE.format(season=s, div=div)))
                log.info("Скачан %s %s", league, s)
            except RuntimeError as e:
                log.warning("%s", e)
    try:
        (RAW / "fixtures.csv").write_bytes(_get(FIXTURES))
    except RuntimeError as e:
        log.warning("%s", e)


def _read_csv(path) -> pd.DataFrame:
    raw = path.read_bytes() if hasattr(path, "read_bytes") else path
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", encoding_errors="replace",
                       on_bad_lines="skip")


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """Первая существующая и непустая колонка из списка (поэлементно)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for n in names:
        if n in df.columns:
            out = out.fillna(pd.to_numeric(df[n], errors="coerce"))
    return out


def _normalize(df: pd.DataFrame, league: str, season: str | None) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    date = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    if "Time" in df.columns:
        t = pd.to_timedelta(df["Time"].fillna("15:00").astype(str) + ":00", errors="coerce")
        date = date + t.fillna(pd.Timedelta(hours=15))
    else:
        date = date + pd.Timedelta(hours=15)
    out = pd.DataFrame({
        "league": league,
        "season": season,
        "date": date,
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
        "hg": _col(df, "FTHG"), "ag": _col(df, "FTAG"),
        "hs": _col(df, "HS"), "as": _col(df, "AS"),
        "hst": _col(df, "HST"), "ast": _col(df, "AST"),
        "hc": _col(df, "HC"), "ac": _col(df, "AC"),
        "hy": _col(df, "HY"), "ay": _col(df, "AY"),
        "hr": _col(df, "HR"), "ar": _col(df, "AR"),
        "referee": df.get("Referee"),
        # Коэффициенты до матча (открытие/предматч)
        "avg_h": _col(df, "AvgH", "BbAvH"), "avg_d": _col(df, "AvgD", "BbAvD"),
        "avg_a": _col(df, "AvgA", "BbAvA"),
        "max_h": _col(df, "MaxH", "BbMxH"), "max_d": _col(df, "MaxD", "BbMxD"),
        "max_a": _col(df, "MaxA", "BbMxA"),
        "ps_h": _col(df, "PSH"), "ps_d": _col(df, "PSD"), "ps_a": _col(df, "PSA"),
        "bfe_h": _col(df, "BFEH"), "bfe_d": _col(df, "BFED"), "bfe_a": _col(df, "BFEA"),
        "b365_h": _col(df, "B365H"), "b365_d": _col(df, "B365D"), "b365_a": _col(df, "B365A"),
        "avg_o25": _col(df, "Avg>2.5", "BbAv>2.5"), "avg_u25": _col(df, "Avg<2.5", "BbAv<2.5"),
        "max_o25": _col(df, "Max>2.5", "BbMx>2.5"), "max_u25": _col(df, "Max<2.5", "BbMx<2.5"),
        "b365_o25": _col(df, "B365>2.5"), "b365_u25": _col(df, "B365<2.5"),
        # Закрытие линии (для CLV — главного показателя качества ставок)
        "cl_h": _col(df, "PSCH", "AvgCH", "B365CH"), "cl_d": _col(df, "PSCD", "AvgCD", "B365CD"),
        "cl_a": _col(df, "PSCA", "AvgCA", "B365CA"),
        "cl_o25": _col(df, "PC>2.5", "AvgC>2.5", "B365C>2.5"),
        "cl_u25": _col(df, "PC<2.5", "AvgC<2.5", "B365C<2.5"),
    })
    # Если в фикстурах нет средних — подставим доступные
    for k in ("h", "d", "a"):
        out[f"avg_{k}"] = out[f"avg_{k}"].fillna(out[f"b365_{k}"]).fillna(out[f"max_{k}"])
        out[f"max_{k}"] = out[f"max_{k}"].fillna(out[f"avg_{k}"])
    for k in ("o25", "u25"):
        out[f"avg_{k}"] = out[f"avg_{k}"].fillna(out[f"b365_{k}"]).fillna(out[f"max_{k}"])
        out[f"max_{k}"] = out[f"max_{k}"].fillna(out[f"avg_{k}"])
    return out.dropna(subset=["date"])


def load_matches() -> pd.DataFrame:
    """Все сыгранные матчи всех лиг, отсортированные по времени."""
    frames = []
    for league, div in cfg()["leagues"].items():
        for s in seasons():
            path = RAW / f"{div}_{s}.csv"
            if not path.exists():
                continue
            frames.append(_normalize(_read_csv(path), league, s))
    if not frames:
        raise RuntimeError("Нет данных. Запустите: python -m abx update")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["hg", "ag"]).sort_values("date").reset_index(drop=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    df["match_id"] = (df.league + "|" + df.date.dt.strftime("%Y-%m-%d") + "|"
                      + df.home + "|" + df.away)
    return df


def _empty_frame() -> pd.DataFrame:
    return _normalize(pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"]), "", None)


def load_fixtures(played: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ближайшие матчи (football-data публикует их с коэффициентами раз в неделю)
    + пользовательские матчи из data/manual/fixtures.csv."""
    divs = {v: k for k, v in cfg()["leagues"].items()}
    frames = []
    path = RAW / "fixtures.csv"
    if path.exists():
        raw = _read_csv(path)
        raw = raw[raw["Div"].isin(divs)]
        frames = [_normalize(g, divs[d], season_code()) for d, g in raw.groupby("Div")]
    fx = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
    manual = MANUAL / "fixtures.csv"
    if manual.exists():  # league,date,home,away[,avg_h,avg_d,avg_a,max_h,...]
        m = pd.read_csv(manual, parse_dates=["date"])
        m["season"] = season_code()
        for col in fx.columns:
            if col not in m:
                m[col] = np.nan
        fx = pd.concat([fx, m[fx.columns]], ignore_index=True)
    if fx.empty:
        return fx
    fx["match_id"] = (fx.league + "|" + fx.date.dt.strftime("%Y-%m-%d") + "|"
                      + fx.home + "|" + fx.away)
    fx = fx.drop_duplicates("match_id")
    if played is not None and len(played):
        fx = fx[~fx.match_id.isin(set(played.match_id))]
    return fx.sort_values("date").reset_index(drop=True)


def xg_proxy_coefs(df: pd.DataFrame) -> tuple[float, float]:
    """Коэффициенты «xG-прокси»: голы ≈ a*удары_в_створ + b*удары_мимо (МНК без свободного члена)."""
    sot = np.concatenate([df.hst.values, df.ast.values])
    shots = np.concatenate([df.hs.values, df["as"].values])
    goals = np.concatenate([df.hg.values, df.ag.values])
    m = ~(np.isnan(sot) | np.isnan(shots) | np.isnan(goals))
    X = np.column_stack([sot[m], np.clip(shots[m] - sot[m], 0, None)])
    coef, *_ = np.linalg.lstsq(X, goals[m], rcond=None)
    return float(coef[0]), max(float(coef[1]), 0.0)


def add_xg_proxy(df: pd.DataFrame) -> pd.DataFrame:
    a, b = xg_proxy_coefs(df)
    df = df.copy()
    df["hxg"] = a * df.hst + b * (df.hs - df.hst).clip(lower=0)
    df["axg"] = a * df.ast + b * (df["as"] - df.ast).clip(lower=0)
    df["hxg"] = df["hxg"].fillna(df.hg)
    df["axg"] = df["axg"].fillna(df.ag)
    return df
