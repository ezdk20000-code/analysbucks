"""Погода на стадионе в момент матча — Open-Meteo (бесплатно, без ключа).

История берётся из архива (archive-api), будущие матчи — из прогноза (до 16 дней).
Запросы кэшируются на диск: один запрос на стадион за сезон.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import requests

from .config import WEATHER_DIR
from .teams import STADIUMS

log = logging.getLogger(__name__)
VARS = "temperature_2m,precipitation,wind_speed_10m"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
COLS = ["w_temp", "w_rain", "w_wind"]


_FAILS = {"n": 0}


def _fetch(url: str, params: dict) -> dict | None:
    if _FAILS["n"] >= 8:  # сервис недоступен — не тормозим весь расчёт
        return None
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=40)
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            r.raise_for_status()
            _FAILS["n"] = 0
            return r.json()
        except requests.RequestException as e:  # pragma: no cover - сеть
            log.warning("Open-Meteo: %s", e)
            time.sleep(2 ** i)
    _FAILS["n"] += 1
    return None


def _hourly_frame(js: dict) -> pd.DataFrame:
    h = js["hourly"]
    return pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "w_temp": h["temperature_2m"], "w_rain": h["precipitation"], "w_wind": h["wind_speed_10m"],
    }).set_index("time")


def _archive(team: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    key = WEATHER_DIR / f"{team.replace(' ', '_').replace(chr(39), '')}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if key.exists():
        return _hourly_frame(json.loads(key.read_text()))
    lat, lon = STADIUMS[team]
    js = _fetch(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": f"{start:%Y-%m-%d}",
                          "end_date": f"{end:%Y-%m-%d}", "hourly": VARS,
                          # время матчей в football-data — британское для всех лиг
                          "timezone": "Europe/London"})
    if not js or "hourly" not in js:
        return None
    key.write_text(json.dumps(js))
    return _hourly_frame(js)


def attach_weather(df: pd.DataFrame, enabled: bool = True, seasons: int = 3) -> pd.DataFrame:
    """Добавляет колонки w_temp, w_rain, w_wind для каждого матча (по стадиону хозяев)."""
    df = df.copy()
    for c in COLS:
        df[c] = np.nan
    if not enabled:
        return df
    now = pd.Timestamp.now().normalize()
    hist = df[df.date < now - pd.Timedelta(days=5)]
    recent = sorted(hist.season.dropna().unique())[-seasons:]  # история погоды — только последние сезоны
    hist = hist[hist.season.isin(recent)]
    for (team, season), g in hist.groupby(["home", "season"]):
        if team not in STADIUMS:
            continue
        start, end = g.date.min().normalize(), g.date.max().normalize()
        wf = _archive(team, start, end)
        if wf is None:
            continue
        ts = g.date.dt.floor("h")
        df.loc[g.index, COLS] = wf.reindex(ts).values
    fut = df[df.date >= now - pd.Timedelta(days=5)]
    for team, g in fut.groupby("home"):
        if team not in STADIUMS:
            continue
        lat, lon = STADIUMS[team]
        js = _fetch(FORECAST, {"latitude": lat, "longitude": lon, "hourly": VARS,
                               "past_days": 5, "forecast_days": 16, "timezone": "Europe/London"})
        if not js or "hourly" not in js:
            continue
        wf = _hourly_frame(js)
        df.loc[g.index, COLS] = wf.reindex(g.date.dt.floor("h")).values
    return df
