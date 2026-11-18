import numpy as np
import pandas as pd
import pytest

from abx import ledger
from abx.betting import build_expresses, outcome, simulate
from abx.odds import fair_power, fair_prop
from abx.ratings import markets_from_matrix, score_matrix
from abx.staking import Staker, kelly

STAKING = {"strategy": "kelly", "kelly_fraction": 0.25, "percent": 0.01, "flat_unit": 0.01,
           "max_stake_pct": 0.03, "max_express_pct": 0.01, "max_daily_exposure_pct": 0.15,
           "drawdown_brake": 0.2, "stop_loss_pct": 0.5, "min_stake": 1.0}


def test_kelly():
    assert kelly(0.5, 2.0) == 0
    assert kelly(0.6, 2.0) == pytest.approx(0.2)
    assert kelly(0.4, 2.0) == 0  # отрицательный перевес — не ставим


def test_staker_caps_and_brakes():
    s = Staker(STAKING, 1000)
    assert s.size(0.9, 2.0, 1000).stake == 30.0          # лимит 3%
    assert s.size(0.9, 2.0, 1000, legs=3).stake == 10.0  # экспресс — 1%
    s.update_peak(2000)
    assert s.size(0.9, 2.0, 1500).stake == 22.5          # просадка >20% — половина
    assert s.size(0.9, 2.0, 400).stake == 0              # стоп-лосс
    assert sum(s.apply_exposure_cap([100, 100], 1000)) == pytest.approx(150)


def test_fair_power_sums_to_one_and_favours_favourite():
    odds = np.array([[1.30, 5.5, 10.0]])
    p_pow, p_prop = fair_power(odds)[0], fair_prop(odds)[0]
    assert p_pow.sum() == pytest.approx(1)
    assert p_pow[0] > p_prop[0]  # степенной метод отдаёт фавориту больше


def test_score_matrix_markets():
    m = markets_from_matrix(score_matrix(1.5, 1.1, -0.05))
    assert m["p_h"] + m["p_d"] + m["p_a"] == pytest.approx(1)
    assert m["p_o25"] + m["p_u25"] == pytest.approx(1)
    assert m["p_h"] > m["p_a"]


def test_outcome():
    assert outcome("1X2", "H", 2, 1) == 1
    assert outcome("1X2", "D", 2, 1) == 0
    assert outcome("OU25", "O", 2, 1) == 1
    assert outcome("OU25", "U", 1, 1) == 1
    assert outcome("1X2", "H", np.nan, np.nan) is None


def _cands():
    d = pd.Timestamp("2025-01-04 15:00")
    rows = []
    for i, (p, o, w) in enumerate([(0.8, 1.3, 1), (0.75, 1.4, 1), (0.7, 1.5, 0), (0.6, 1.6, 1)]):
        rows.append({"match_id": f"m{i}", "league": "EPL", "season": "2425", "date": d, "home": f"H{i}",
                     "away": f"A{i}", "market": "1X2", "sel": "H", "p": p, "odds": o,
                     "edge": p * o - 1, "win": float(w), "p_close": p})
    return pd.DataFrame(rows)


def test_express_combines_edges():
    ex = build_expresses(_cands(), {"legs": 2, "max_per_week": 2, "min_leg_edge": 0.0,
                                    "leg_min_odds": 1.2, "leg_max_odds": 1.7})
    # у m3 перевес отрицательный (0.6*1.6<1) — он не попадает; из 3 событий — 1 экспресс по 2
    assert len(ex) == 1
    first = ex.iloc[0]
    assert set(first.match_id.split("+")) == {"m1", "m2"}  # два лучших по перевесу
    assert first.odds == pytest.approx(1.4 * 1.5)
    assert first.edge == pytest.approx(first.p * first.odds - 1)
    assert first.win == 0  # m2 проиграл — экспресс проигран


def test_simulate_grows_with_edge():
    rng = np.random.default_rng(0)
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    p, o = 0.6, 1.8
    bets = pd.DataFrame({"date": dates, "p": p, "odds": o,
                         "win": (rng.random(n) < p).astype(float), "clv": 0.02, "season": "x"})
    r = simulate(bets, 1000, STAKING)
    assert r.summary["ставок"] == n
    assert r.summary["итог банка"] > 1000


def test_ledger_flow(tmp_path):
    db = tmp_path / "l.db"
    ledger.bank_event("deposit", 1000, path=db)
    leg = {"match_id": "EPL|2025-01-04|Arsenal|Chelsea", "home": "Arsenal", "away": "Chelsea",
           "market": "1X2", "sel": "H", "odds": 2.0, "p": 0.55, "date": "2025-01-04"}
    bid = ledger.place_bet([leg], 50, path=db)
    assert ledger.balance(path=db)["свободно"] == 950
    results = pd.DataFrame({"match_id": [leg["match_id"]], "home": ["Arsenal"], "away": ["Chelsea"],
                            "date": [pd.Timestamp("2025-01-04 15:00")], "hg": [2], "ag": [0]})
    assert ledger.auto_settle(results, path=db) == [bid]
    b = ledger.balance(path=db)
    assert b["банк"] == 1050 and b["открытых ставок"] == 0
    assert ledger.stats(path=db)["ROI %"] == 100.0
