import pandas as pd

from abx import ucl

PAGE = """<html><body><p>Fixtures</p><p>Matchday 5</p><p>Tuesday 24 November 2026</p>
<p>Bodø/Glimt vs LASK</p><p>(18:45)</p><p>Arsenal vs Borussia Dortmund</p>
<p>Results</p><p>Matchday 4</p><p>Tuesday 3 November 2026</p><p>Shakhtar Donetsk 0-1 Sporting CP</p>
<p>Highlights: Roma 3-4 Real Madrid</p><p>© 1998-2026 UEFA. All rights reserved.</p></body></html>"""


def test_parse_uefa_article():
    fx, res = ucl.parse_uefa_article(PAGE)
    assert list(fx.home) == ["Bodø/Glimt", "Arsenal"]
    assert fx.date.iloc[0] == pd.Timestamp("2026-11-24 18:45")
    assert fx.date.iloc[1] == pd.Timestamp("2026-11-24 21:00")
    assert len(res) == 1 and res.iloc[0].away == "Sporting CP" and res.iloc[0].ag == 1


def test_table_and_verdict():
    res = pd.DataFrame({"md": [1, 1], "date": pd.Timestamp("2026-09-08"), "home": ["A", "C"],
                        "away": ["B", "D"], "hg": [2, 1], "ag": [0, 1]})
    t = ucl.ucl_table(res)
    assert t.loc["A", "pts"] == 3 and t.loc["A", "pos"] == 1 and t.loc["C", "pts"] == 1
    assert ucl.verdict(0.85, 0.1, 0.05) == ("П1", "очень высокая")
    v, conf = ucl.verdict(0.35, 0.30, 0.35)
    assert conf == "низкая" and "1X" in v
