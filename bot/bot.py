"""Telegram-бот AnalysBucks (long polling, без внешних библиотек кроме requests).

Запуск:  python -m bot.bot
Нужно в .env:
  TELEGRAM_BOT_TOKEN=123456:ABC...        (токен от @BotFather)
  TELEGRAM_ALLOWED_CHATS=123456789        (ваш chat id; несколько — через запятую)
Если TELEGRAM_ALLOWED_CHATS пуст, бот ответит на /start и покажет ваш chat id.

Команды:
  /predict          ближайшие матчи и рекомендуемые ставки (с номерами)
  /take N [сумма]   записать рекомендацию №N в учёт (виртуально)
  /takereal N [сумма] записать как реальную ставку
  /match Команда1 - Команда2   прогноз любой пары
  /bank             банк и статистика
  /bets             открытые ставки
  /settle ID won|lost|void     ручной расчёт
  /deposit СУММА, /withdraw СУММА
  /update           обновить данные и рассчитать ставки
  /table EPL|LaLiga|SerieA|... рейтинги
  /ucl              прогноз ближайшего тура Лиги чемпионов
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abx import ledger, service  # noqa: E402
from abx.betting import SEL_RU  # noqa: E402
from abx.config import CACHE, cfg  # noqa: E402

log = logging.getLogger("bot")
API = "https://api.telegram.org/bot{token}/{method}"
LAST = CACHE / "bot_last_recommendations.json"


def token() -> str:
    t = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not t:
        sys.exit("Нет TELEGRAM_BOT_TOKEN в .env")
    return t


def allowed() -> set[int]:
    return {int(x) for x in os.environ.get("TELEGRAM_ALLOWED_CHATS", "").replace(" ", "").split(",") if x}


def call(method: str, **params):
    r = requests.post(API.format(token=token(), method=method), json=params, timeout=70)
    r.raise_for_status()
    return r.json()


def send(chat_id: int, text: str) -> None:
    # лимит Telegram — 4096 символов
    for i in range(0, len(text), 3900):
        call("sendMessage", chat_id=chat_id, text=text[i:i + 3900], parse_mode="HTML",
             disable_web_page_preview=True)


# ------------------------------------------------------------------ тексты


def recommendations_text() -> str:
    rec = service.recommendations()
    up = service.upcoming_predictions()
    if up.empty:
        return ("Ближайших матчей с линией пока нет — football-data публикует их за 2–5 дней до тура.\n"
                "Прогноз любой пары: /match Arsenal - Chelsea")
    lines = [f"<b>Банк для расчёта:</b> {rec['bank']:.2f}\n", "<b>Прогнозы</b> (П1 / X / П2):"]
    for _, r in up.iterrows():
        lines.append(f"{r.date:%d.%m %H:%M} {r.home} – {r.away}: "
                     f"{r.p_h:.0%} / {r.p_d:.0%} / {r.p_a:.0%}")
    items = []
    lines.append("\n<b>Value-ординары:</b>")
    for _, r in rec["singles"].iterrows():
        items.append({"legs": [service.leg_dict(r)], "stake": float(r.stake)})
        lines.append(f"[{len(items)}] {r.home} – {r.away}: <b>{SEL_RU[r.sel]} @{r.odds:.2f}</b> "
                     f"(перевес {r.edge:+.1%}) → {r.stake:.2f}")
    if rec["singles"].empty:
        lines.append("нет")
    lines.append("\n<b>Экспрессы:</b>")
    for _, r in rec["express"].iterrows():
        items.append({"legs": r.legs_data, "stake": float(r.stake)})
        lines.append(f"[{len(items)}] {r.sel}\nкэф <b>{r.odds:.2f}</b>, проход {r.p:.0%}, "
                     f"перевес {r.edge:+.1%} → {r.stake:.2f}")
    if rec["express"].empty:
        lines.append("нет")
    LAST.write_text(json.dumps(items, ensure_ascii=False, default=str))
    lines.append("\nЗаписать: /take N [сумма] (виртуально) или /takereal N [сумма]")
    lines.append("<i>Коэффициент — лучший на рынке. Ставьте, только если у вашего букмекера "
                 "кэф не ниже указанного.</i>")
    return "\n".join(lines)


def bank_text() -> str:
    b, s = ledger.balance(), ledger.stats()
    return ("<b>Банк</b>\n" + "\n".join(f"{k}: {v}" for k, v in b.items())
            + "\n\n<b>Статистика</b>\n" + "\n".join(f"{k}: {v}" for k, v in s.items()))


def bets_text() -> str:
    df = ledger.bets_df("open")
    if df.empty:
        return "Открытых ставок нет"
    return "\n".join(f"#{r.id} [{r['mode']}] {r['описание']} | кэф {r.odds:.2f} | {r.stake:.2f}"
                     for _, r in df.iterrows())


def match_text(arg: str) -> str:
    if "-" not in arg:
        return "Формат: /match Arsenal - Chelsea"
    home, away = (x.strip() for x in arg.split("-", 1))
    names = {n.lower(): n for teams in service.team_names().values() for n in teams}
    home, away = names.get(home.lower(), home), names.get(away.lower(), away)
    try:
        f = service.match_forecast(home, away)
    except KeyError as e:
        return f"{e}. Названия как в football-data, например: Man City, Nott'm Forest, Ath Madrid"
    sc = ", ".join(f"{s} ({p:.0%})" for s, p in f["top_scores"])
    fo = f["fair_odds"]
    t = (f"<b>{f['home']} – {f['away']}</b> ({f['league']})\n"
         f"xG {f['xg_home']:.2f} : {f['xg_away']:.2f}\n"
         f"П1 {f['p_h']:.1%} | X {f['p_d']:.1%} | П2 {f['p_a']:.1%}\n"
         f"ТБ 2.5 {f['p_o25']:.1%} | Обе забьют {f['p_btts']:.1%}\n"
         f"Справедливые кэфы: П1 {fo['p_h']} X {fo['p_d']} П2 {fo['p_a']} ТБ {fo['p_o25']} ТМ {fo['p_u25']}\n"
         f"Счета: {sc}")
    if f["notes"]:
        t += "\nПоправки: " + "; ".join(f["notes"])
    return t + "\n<i>Это модель по рейтингам (без рынка). Value ищите в /predict.</i>"


def take(arg: str, real: bool) -> str:
    parts = arg.split()
    if not parts or not parts[0].isdigit():
        return "Формат: /take N [сумма]"
    items = json.loads(LAST.read_text()) if LAST.exists() else []
    n = int(parts[0])
    if not 1 <= n <= len(items):
        return "Нет такой рекомендации — сначала /predict"
    it = items[n - 1]
    stake = float(parts[1].replace(",", ".")) if len(parts) > 1 else it["stake"]
    bid = ledger.place_bet(it["legs"], stake, "real" if real else "paper")
    return f"Ставка #{bid} записана: {ledger.describe_legs(it['legs'])} — {stake:.2f}"


def handle(chat_id: int, text: str) -> str:
    cmd, _, arg = text.strip().partition(" ")
    cmd = cmd.split("@")[0].lower()
    if cmd in ("/start", "/help"):
        return __doc__.split("Команды:")[1].strip() if __doc__ else "Привет"
    if cmd == "/predict":
        return recommendations_text()
    if cmd == "/take":
        return take(arg, False)
    if cmd == "/takereal":
        return take(arg, True)
    if cmd == "/match":
        return match_text(arg)
    if cmd == "/bank":
        return bank_text()
    if cmd == "/bets":
        return bets_text()
    if cmd == "/settle":
        p = arg.split()
        if len(p) != 2 or p[1] not in ("won", "lost", "void"):
            return "Формат: /settle ID won|lost|void"
        ledger.settle_bet(int(p[0]), p[1])
        return "Готово\n" + bank_text()
    if cmd in ("/deposit", "/withdraw"):
        try:
            ledger.bank_event(cmd[1:], float(arg.replace(",", ".")))
        except ValueError:
            return f"Формат: {cmd} 1000"
        return bank_text()
    if cmd == "/update":
        r = service.update()
        return "Обновлено: " + ", ".join(f"{k} {v}" for k, v in r.items())
    if cmd == "/ucl":
        from abx import ucl
        return ucl.report_text(ucl.forecast(), html=True)
    if cmd == "/table":
        names = {k.lower(): k for k in cfg()["leagues"]}
        lg = names.get((arg.strip() or "EPL").lower())
        if lg is None:
            return "Лиги: " + ", ".join(cfg()["leagues"])
        t = service.strength_table(lg)
        return "<pre>" + t[["team", "elo", "xG за (vs средний)", "xG против"]].to_string(index=False) + "</pre>"
    return "Не понял команду. /help"


def send_recommendations() -> None:
    """Для расписания (cron): разослать рекомендации всем разрешённым чатам."""
    text = recommendations_text()
    for chat in allowed():
        send(chat, text)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    offset = None
    ok = allowed()
    log.info("Бот запущен. Разрешённые чаты: %s", ok or "не заданы")
    while True:
        try:
            upd = call("getUpdates", timeout=cfg()["telegram"]["poll_timeout"], offset=offset)
        except requests.RequestException as e:
            log.warning("Сеть: %s", e)
            time.sleep(5)
            continue
        for u in upd.get("result", []):
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("edited_message")
            if not msg or "text" not in msg:
                continue
            chat = msg["chat"]["id"]
            if not ok:
                send(chat, f"Ваш chat id: <code>{chat}</code>\nДобавьте его в TELEGRAM_ALLOWED_CHATS в .env "
                           "и перезапустите бота.")
                continue
            if chat not in ok:
                continue
            try:
                send(chat, handle(chat, msg["text"]))
            except Exception as e:  # noqa: BLE001 — бот не должен падать
                log.exception("Ошибка")
                send(chat, f"Ошибка: {e}")


if __name__ == "__main__":
    run()
