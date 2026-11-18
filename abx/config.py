"""Загрузка конфигурации и путей проекта."""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
WEATHER_DIR = DATA / "weather"
MANUAL = DATA / "manual"
CACHE = DATA / "cache"
LEDGER_DB = DATA / "ledger.db"

for _p in (DATA, RAW, WEATHER_DIR, MANUAL, CACHE):
    _p.mkdir(parents=True, exist_ok=True)


def _load_env() -> None:
    """Простейший парсер .env (без внешних зависимостей)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


@lru_cache(maxsize=1)
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def season_code(d: date | None = None) -> str:
    """Код сезона football-data: 2026-11-18 -> '2627'."""
    d = d or date.today()
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def seasons() -> list[str]:
    first = int(str(cfg()["first_season"])[:2])
    cur = int(season_code()[:2])
    return [f"{y:02d}{(y + 1) % 100:02d}" for y in range(first, cur + 1)]
