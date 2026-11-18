"""Командная строка: python -m abx <команда>.

  update                      скачать данные, рассчитать завершённые ставки
  predict                     прогнозы и рекомендуемые ставки на ближайшие матчи
  match "Arsenal" "Chelsea"   прогноз любой пары по текущим рейтингам
  table EPL|LaLiga            рейтинги силы команд
  backtest [--mode both|singles|express] [--since 1213] [--legs 3] [--kelly 0.25]
  deposit 1000 | withdraw 200 открыть/пополнить/вывести банк
  bank                        состояние банка и статистика ставок
  bets [--open]               список ставок
  bet --home A --away B --market 1X2 --sel H --odds 1.85 --stake 20 [--real]
  take N [--stake X] [--real] записать рекомендацию №N из последнего predict
  settle ID won|lost|void     ручной расчёт ставки
  notify                      отправить рекомендации в Telegram
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from . import ledger, service
from .betting import SEL_RU
from .config import CACHE

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
LAST = CACHE / "last_recommendations.json"


def _print_dict(d: dict) -> None:
    for k, v in d.items():
        print(f"  {k}: {v}")


def cmd_predict(_a) -> None:
    rec = service.recommendations()
    up = service.upcoming_predictions()
    if up.empty:
        print("Ближайших матчей с линией нет (football-data публикует их за 2–5 дней до тура).\n"
              "Для любой пары: python -m abx match \"Arsenal\" \"Chelsea\"")
        return
    print(f"Банк для расчёта ставок: {rec['bank']:.2f}\n")
    print("Прогнозы (рынок без маржи / модель):")
    for _, r in up.iterrows():
        print(f"  {r.date:%d.%m %H:%M} {r.league:6} {r.home} – {r.away}: "
              f"П1 {r.p_h:.0%} X {r.p_d:.0%} П2 {r.p_a:.0%} | ТБ2.5 {r.p_o25:.0%} "
              f"| модель П1 {r.m_p_h:.0%} X {r.m_p_d:.0%} П2 {r.m_p_a:.0%}")
    items = []
    print("\nValue-ординары:")
    for _, r in rec["singles"].iterrows():
        items.append({"legs": [service.leg_dict(r)], "stake": float(r.stake)})
        print(f"  [{len(items)}] {r.date:%d.%m} {r.home} – {r.away}: {SEL_RU[r.sel]} "
              f"@{r.odds:.2f} (p={r.p:.1%}, перевес {r.edge:+.1%}) → ставка {r.stake:.2f}")
    if rec["singles"].empty:
        print("  нет")
    print("\nЭкспрессы:")
    for _, r in rec["express"].iterrows():
        items.append({"legs": r.legs_data, "stake": float(r.stake)})
        print(f"  [{len(items)}] {r.sel}\n      итоговый кэф {r.odds:.2f}, проход {r.p:.1%}, "
              f"перевес {r.edge:+.1%} → ставка {r.stake:.2f}")
    if rec["express"].empty:
        print("  нет")
    LAST.write_text(json.dumps(items, ensure_ascii=False, default=str))
    print("\nЗаписать ставку: python -m abx take N [--stake X] [--real]")


def cmd_take(a) -> None:
    items = json.loads(LAST.read_text()) if LAST.exists() else []
    if not (1 <= a.n <= len(items)):
        sys.exit("Нет такой рекомендации — сначала запустите predict")
    it = items[a.n - 1]
    bid = ledger.place_bet(it["legs"], a.stake or it["stake"], "real" if a.real else "paper",
                           a.bookmaker or "")
    print(f"Ставка #{bid} записана: {ledger.describe_legs(it['legs'])}, сумма {a.stake or it['stake']}")


def cmd_bet(a) -> None:
    leg = {"home": a.home, "away": a.away, "market": a.market, "sel": a.sel, "odds": a.odds,
           "date": a.date, "league": a.league, "p": a.p}
    bid = ledger.place_bet([leg], a.stake, "real" if a.real else "paper", a.bookmaker or "")
    print(f"Ставка #{bid} записана")


def cmd_backtest(a) -> None:
    ov: dict = {"mode": a.mode, "staking": {}, "express": {}, "betting": {}}
    if a.kelly is not None:
        ov["staking"]["kelly_fraction"] = a.kelly
    if a.legs:
        ov["express"]["legs"] = a.legs
    if a.min_edge is not None:
        ov["betting"]["min_edge"] = a.min_edge
    r = service.backtest(ov, since=a.since)
    print("Итог бэктеста:")
    _print_dict(r["summary"])
    if "by_season" in r:
        print("\nПо сезонам:\n", r["by_season"].to_string())
    if r["mc"]:
        print("\nМонте-Карло (перемешивание порядка ставок):")
        _print_dict(r["mc"])


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="abx", description="AnalysBucks — прогнозы и ставки")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("update")
    sp.add_parser("predict")
    m = sp.add_parser("match"); m.add_argument("home"); m.add_argument("away")
    t = sp.add_parser("table"); t.add_argument("league", choices=["EPL", "LaLiga"])
    b = sp.add_parser("backtest")
    b.add_argument("--mode", default="both", choices=["both", "singles", "express"])
    b.add_argument("--since"); b.add_argument("--legs", type=int)
    b.add_argument("--kelly", type=float); b.add_argument("--min-edge", type=float)
    for k in ("deposit", "withdraw"):
        x = sp.add_parser(k); x.add_argument("amount", type=float); x.add_argument("--note", default="")
    sp.add_parser("bank")
    bl = sp.add_parser("bets"); bl.add_argument("--open", action="store_true")
    bt = sp.add_parser("bet")
    for k in ("home", "away", "market", "sel"):
        bt.add_argument(f"--{k}", required=True)
    bt.add_argument("--odds", type=float, required=True); bt.add_argument("--stake", type=float, required=True)
    bt.add_argument("--date", default=""); bt.add_argument("--league", default="")
    bt.add_argument("--p", type=float); bt.add_argument("--bookmaker"); bt.add_argument("--real", action="store_true")
    tk = sp.add_parser("take"); tk.add_argument("n", type=int); tk.add_argument("--stake", type=float)
    tk.add_argument("--real", action="store_true"); tk.add_argument("--bookmaker")
    st = sp.add_parser("settle"); st.add_argument("id", type=int); st.add_argument("status", choices=["won", "lost", "void"])
    sp.add_parser("notify")
    a = p.parse_args(argv)

    if a.cmd == "update":
        _print_dict(service.update())
    elif a.cmd == "predict":
        cmd_predict(a)
    elif a.cmd == "match":
        f = service.match_forecast(a.home, a.away)
        print(f"{f['home']} – {f['away']} ({f['league']}): xG {f['xg_home']:.2f} : {f['xg_away']:.2f}")
        print(f"  П1 {f['p_h']:.1%}  X {f['p_d']:.1%}  П2 {f['p_a']:.1%}  ТБ2.5 {f['p_o25']:.1%}  ОЗ {f['p_btts']:.1%}")
        print("  Справедливые кэфы:", f["fair_odds"])
        print("  Вероятные счета:", ", ".join(f"{s} ({p:.0%})" for s, p in f["top_scores"]))
        for n in f["notes"]:
            print("  Поправка:", n)
    elif a.cmd == "table":
        print(service.strength_table(a.league).to_string())
    elif a.cmd == "backtest":
        cmd_backtest(a)
    elif a.cmd in ("deposit", "withdraw"):
        ledger.bank_event(a.cmd, a.amount, a.note)
        _print_dict(ledger.balance())
    elif a.cmd == "bank":
        print("Банк:"); _print_dict(ledger.balance())
        print("Статистика:"); _print_dict(ledger.stats())
    elif a.cmd == "bets":
        df = ledger.bets_df("open" if a.open else None)
        if df.empty:
            print("Ставок нет")
        else:
            print(df[["id", "created_at", "mode", "kind", "описание", "odds", "stake", "status", "pnl"]].to_string(index=False))
    elif a.cmd == "bet":
        cmd_bet(a)
    elif a.cmd == "take":
        cmd_take(a)
    elif a.cmd == "settle":
        ledger.settle_bet(a.id, a.status); print("Готово"); _print_dict(ledger.balance())
    elif a.cmd == "notify":
        from bot.bot import send_recommendations
        send_recommendations()


if __name__ == "__main__":
    main()
