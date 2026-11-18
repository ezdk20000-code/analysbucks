"""Справочник клубов: координаты стадионов (для погоды и расстояния переездов).

Названия — как в данных football-data.co.uk. Если в лигу пришёл новый клуб,
добавьте его сюда (широта, долгота); без координат просто не будет погоды/переезда.
"""
from __future__ import annotations

import math

STADIUMS: dict[str, tuple[float, float]] = {
    # --- Англия ---
    "Arsenal": (51.5549, -0.1084),
    "Aston Villa": (52.5092, -1.8848),
    "Bournemouth": (50.7352, -1.8383),
    "Brentford": (51.4907, -0.2886),
    "Brighton": (50.8616, -0.0837),
    "Burnley": (53.7890, -2.2302),
    "Chelsea": (51.4817, -0.1910),
    "Coventry": (52.4481, -1.4956),
    "Crystal Palace": (51.3983, -0.0855),
    "Everton": (53.4246, -3.0015),
    "Fulham": (51.4749, -0.2217),
    "Hull": (53.7462, -0.3678),
    "Ipswich": (52.0545, 1.1447),
    "Leeds": (53.7778, -1.5722),
    "Leicester": (52.6204, -1.1422),
    "Liverpool": (53.4308, -2.9608),
    "Luton": (51.8842, -0.4316),
    "Man City": (53.4831, -2.2004),
    "Man United": (53.4631, -2.2913),
    "Middlesbrough": (54.5781, -1.2170),
    "Newcastle": (54.9756, -1.6217),
    "Norwich": (52.6221, 1.3087),
    "Nott'm Forest": (52.9400, -1.1328),
    "Sheffield United": (53.3703, -1.4709),
    "Southampton": (50.9058, -1.3911),
    "Sunderland": (54.9146, -1.3884),
    "Tottenham": (51.6043, -0.0664),
    "Watford": (51.6498, -0.4015),
    "West Brom": (52.5090, -1.9639),
    "West Ham": (51.5386, -0.0166),
    "Wolves": (52.5903, -2.1306),
    # --- Испания ---
    "Alaves": (42.8371, -2.6880),
    "Almeria": (36.8400, -2.4357),
    "Ath Bilbao": (43.2642, -2.9494),
    "Ath Madrid": (40.4362, -3.5995),
    "Barcelona": (41.3809, 2.1228),
    "Betis": (37.3565, -5.9817),
    "Cadiz": (36.5027, -6.2729),
    "Celta": (42.2118, -8.7397),
    "Eibar": (43.1814, -2.4757),
    "Elche": (38.2672, -0.6631),
    "Espanol": (41.3480, 2.0756),
    "Getafe": (40.3257, -3.7148),
    "Girona": (41.9611, 2.8286),
    "Granada": (37.1530, -3.5958),
    "Huesca": (42.1317, -0.4220),
    "La Coruna": (43.3687, -8.4174),
    "Las Palmas": (28.1001, -15.4567),
    "Leganes": (40.3404, -3.7605),
    "Levante": (39.4947, -0.3644),
    "Malaga": (36.7340, -4.4266),
    "Mallorca": (39.5900, 2.6301),
    "Osasuna": (42.7966, -1.6371),
    "Oviedo": (43.3606, -5.8698),
    "Real Madrid": (40.4531, -3.6883),
    "Santander": (43.4762, -3.7932),
    "Sevilla": (37.3840, -5.9705),
    "Sociedad": (43.3014, -1.9737),
    "Valencia": (39.4746, -0.3583),
    "Valladolid": (41.6445, -4.7612),
    "Vallecano": (40.3919, -3.6586),
    "Villarreal": (39.9441, -0.1035),
}


def haversine_km(a: str, b: str) -> float:
    """Расстояние между стадионами (км). NaN, если координат нет."""
    if a not in STADIUMS or b not in STADIUMS:
        return float("nan")
    (lat1, lon1), (lat2, lon2) = STADIUMS[a], STADIUMS[b]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))
