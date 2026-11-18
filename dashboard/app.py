"""Дашборд AnalysBucks.  Запуск:  streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abx import ledger, service  # noqa: E402
from abx.betting import SEL_RU  # noqa: E402
from abx.config import cfg  # noqa: E402

st.set_page_config(page_title="AnalysBucks", page_icon="⚽", layout="wide")


@st.cache_data(ttl=1800, show_spinner="Считаю прогнозы…")
def get_pred():
    return service.predictions()


@st.cache_data(ttl=3600, show_spinner="Гоняю бэктест…")
def get_backtest(mode, min_edge, max_odds, kelly, legs, leg_max, strategy, since):
    return service.backtest({"mode": mode,
                             "betting": {"min_edge": min_edge, "max_odds": max_odds},
                             "staking": {"kelly_fraction": kelly, "strategy": strategy},
                             "express": {"legs": legs, "leg_max_odds": leg_max}},
                            since=since, pred=get_pred())


st.title("⚽ AnalysBucks — прогнозы и ставки")
bal = ledger.balance()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Банк", f"{bal['банк']:.2f}")
c2.metric("Свободно", f"{bal['свободно']:.2f}")
c3.metric("В игре", f"{bal['в игре']:.2f}", f"{bal['открытых ставок']} ставок")
c4.metric("Прибыль", f"{bal['прибыль']:.2f}")

with st.sidebar:
    st.header("Данные")
    if st.button("🔄 Обновить данные и рассчитать ставки"):
        with st.spinner("Скачиваю…"):
            r = service.update()
        get_pred.clear()
        st.success(f"Готово: {r}")
    st.caption("Источники: football-data.co.uk (результаты, удары, коэффициенты), Open-Meteo (погода).")
    st.header("Банк")
    amt = st.number_input("Сумма", min_value=0.0, value=1000.0, step=100.0)
    b1, b2 = st.columns(2)
    if b1.button("Пополнить"):
        ledger.bank_event("deposit", amt)
        st.rerun()
    if b2.button("Вывести"):
        ledger.bank_event("withdraw", amt)
        st.rerun()

tab_pred, tab_match, tab_ledger, tab_bt, tab_tbl = st.tabs(
    ["📅 Прогнозы и ставки", "🔍 Матч", "📒 Учёт ставок", "🧪 Бэктест", "📊 Рейтинги"])

# ------------------------------------------------------------------ прогнозы
with tab_pred:
    pred = get_pred()
    up = service.upcoming_predictions(pred)
    if up.empty:
        st.info("Ближайших матчей с линией нет. football-data публикует их за 2–5 дней до тура. "
                "Можно добавить матчи вручную в data/manual/fixtures.csv.")
    else:
        view = up[["date", "league", "home", "away", "p_h", "p_d", "p_a", "p_o25",
                   "m_p_h", "m_p_d", "m_p_a", "max_h", "max_d", "max_a", "w_temp", "w_rain", "w_wind",
                   "h_rest", "a_rest", "travel_km"]].copy()
        view.columns = ["Дата", "Лига", "Хозяева", "Гости", "П1", "X", "П2", "ТБ2.5",
                        "Модель П1", "Модель X", "Модель П2", "Кэф П1", "Кэф X", "Кэф П2",
                        "°C", "Осадки мм", "Ветер км/ч", "Отдых хоз.", "Отдых гост.", "Переезд км"]
        st.dataframe(view.style.format({c: "{:.0%}" for c in ["П1", "X", "П2", "ТБ2.5", "Модель П1",
                                                              "Модель X", "Модель П2"]}
                                       | {"Кэф П1": "{:.2f}", "Кэф X": "{:.2f}", "Кэф П2": "{:.2f}",
                                          "Дата": "{:%d.%m %H:%M}", "Осадки мм": "{:.1f}",
                                          "Ветер км/ч": "{:.0f}", "Отдых хоз.": "{:.0f}",
                                          "Отдых гост.": "{:.0f}",
                                          "°C": "{:.0f}", "Переезд км": "{:.0f}"}, na_rep="—"),
                     width="stretch", hide_index=True)
        rec = service.recommendations(pred)
        st.subheader(f"Рекомендации (банк {rec['bank']:.2f}, стратегия {cfg()['staking']['strategy']})")
        mode = st.radio("Записать как", ["paper", "real"], horizontal=True,
                        format_func=lambda x: "виртуально" if x == "paper" else "реальная ставка")
        for i, r in rec["singles"].reset_index(drop=True).iterrows():
            cols = st.columns([6, 2, 2])
            cols[0].markdown(f"**{r.home} – {r.away}**: {SEL_RU[r.sel]} @ **{r.odds:.2f}** · "
                             f"p={r.p:.1%} · перевес **{r.edge:+.1%}**")
            stake = cols[1].number_input("Сумма", value=float(r.stake), key=f"s{i}", label_visibility="collapsed")
            if cols[2].button("Записать", key=f"bs{i}"):
                ledger.place_bet([service.leg_dict(r)], stake, mode)
                st.success("Записано")
        for i, r in rec["express"].reset_index(drop=True).iterrows():
            cols = st.columns([6, 2, 2])
            cols[0].markdown(f"**Экспресс**: {r.sel}<br>кэф **{r.odds:.2f}** · проход {r.p:.0%} · "
                             f"перевес **{r.edge:+.1%}**", unsafe_allow_html=True)
            stake = cols[1].number_input("Сумма", value=float(r.stake), key=f"e{i}", label_visibility="collapsed")
            if cols[2].button("Записать", key=f"be{i}"):
                ledger.place_bet(r.legs_data, stake, mode)
                st.success("Записано")
        if rec["singles"].empty and rec["express"].empty:
            st.write("Value-ставок на ближайшие матчи нет — это нормально, ставим только с перевесом.")
        st.caption("Кэф — лучший на рынке (max среди букмекеров). Если у вашего букмекера ниже — "
                   "перевес может исчезнуть.")

# ------------------------------------------------------------------ матч
with tab_match:
    names = service.team_names()
    lg = st.selectbox("Лига", list(names))
    a, b = st.columns(2)
    home = a.selectbox("Хозяева", names[lg], index=0)
    away = b.selectbox("Гости", names[lg], index=1)
    if home != away:
        f = service.match_forecast(home, away, lg)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("П1", f"{f['p_h']:.1%}", f"кэф {f['fair_odds']['p_h']}")
        m2.metric("X", f"{f['p_d']:.1%}", f"кэф {f['fair_odds']['p_d']}")
        m3.metric("П2", f"{f['p_a']:.1%}", f"кэф {f['fair_odds']['p_a']}")
        m4.metric("ТБ 2.5", f"{f['p_o25']:.1%}", f"кэф {f['fair_odds']['p_o25']}")
        st.write(f"Ожидаемые голы: **{f['xg_home']:.2f} : {f['xg_away']:.2f}** · "
                 f"Elo {f['elo_home']:.0f} vs {f['elo_away']:.0f} · Обе забьют {f['p_btts']:.0%}")
        st.write("Вероятные счета: " + ", ".join(f"{s} ({p:.0%})" for s, p in f["top_scores"]))
        for n in f["notes"]:
            st.warning(f"Ручная поправка: {n}")

# ------------------------------------------------------------------ учёт
with tab_ledger:
    s = ledger.stats()
    st.write(" · ".join(f"**{k}**: {v}" for k, v in s.items()))
    hist = ledger.bank_history()
    if len(hist):
        st.plotly_chart(px.line(hist, x="ts", y="bank", title="Банк"), width="stretch")
    df = ledger.bets_df()
    if not df.empty:
        st.dataframe(df[["id", "created_at", "mode", "kind", "описание", "odds", "stake", "edge",
                         "status", "pnl", "clv"]], width="stretch", hide_index=True)
        op = df[df.status == "open"]
        if len(op):
            st.subheader("Рассчитать вручную")
            c = st.columns([3, 2, 2])
            bid = c[0].selectbox("Ставка", op.id, format_func=lambda i: f"#{i} {op.set_index('id').at[i, 'описание']}")
            stt = c[1].selectbox("Итог", ["won", "lost", "void"],
                                 format_func={"won": "выигрыш", "lost": "проигрыш", "void": "возврат"}.get)
            if c[2].button("Рассчитать"):
                ledger.settle_bet(int(bid), stt)
                st.rerun()
    with st.expander("➕ Добавить ставку вручную"):
        names = service.team_names()
        c = st.columns(3)
        lg2 = c[0].selectbox("Лига ", list(names))
        h2 = c[1].selectbox("Хозяева ", names[lg2])
        a2 = c[2].selectbox("Гости ", names[lg2], index=1)
        c = st.columns(5)
        mk = c[0].selectbox("Рынок", ["1X2", "OU25"])
        sel = c[1].selectbox("Исход", ["H", "D", "A"] if mk == "1X2" else ["O", "U"], format_func=SEL_RU.get)
        od = c[2].number_input("Кэф", min_value=1.01, value=1.9)
        sk = c[3].number_input("Сумма ", min_value=0.0, value=10.0)
        dt = c[4].date_input("Дата матча")
        bk = st.text_input("Букмекер")
        real = st.checkbox("Реальная ставка")
        if st.button("Записать ставку"):
            ledger.place_bet([{"home": h2, "away": a2, "league": lg2, "market": mk, "sel": sel,
                               "odds": od, "date": str(dt)}], sk, "real" if real else "paper", bk)
            st.rerun()

# ------------------------------------------------------------------ бэктест
with tab_bt:
    st.write("Симуляция стратегии на истории с реальными коэффициентами и управлением банком.")
    c = st.columns(4)
    mode = c[0].selectbox("Что ставим", ["both", "singles", "express"],
                          format_func={"both": "ординары + экспрессы", "singles": "только ординары",
                                       "express": "только экспрессы"}.get)
    strategy = c[1].selectbox("Мани-менеджмент", ["kelly", "percent", "flat"],
                              format_func={"kelly": "Келли (доля)", "percent": "% от банка",
                                           "flat": "флэт"}.get)
    kelly = c[2].slider("Доля Келли", 0.05, 1.0, float(cfg()["staking"]["kelly_fraction"]), 0.05)
    since = c[3].selectbox("С сезона", ["1213", "1516", "1819", "2122"], index=0)
    c = st.columns(4)
    min_edge = c[0].slider("Мин. перевес", 0.0, 0.10, float(cfg()["betting"]["min_edge"]), 0.005)
    max_odds = c[1].slider("Макс. кэф ординара", 1.3, 6.0, float(cfg()["betting"]["max_odds"]), 0.1)
    legs = c[2].slider("Событий в экспрессе", 2, 6, int(cfg()["express"]["legs"]))
    leg_max = c[3].slider("Макс. кэф события в экспрессе", 1.2, 2.5, float(cfg()["express"]["leg_max_odds"]), 0.05)
    r = get_backtest(mode, min_edge, max_odds, kelly, legs, leg_max, strategy, since)
    sm = r["summary"]
    if sm.get("ставок", 0) == 0:
        st.warning("При этих настройках ставок нет")
    else:
        k = st.columns(5)
        k[0].metric("Рост банка", f"×{sm['рост банка, раз']}")
        k[1].metric("ROI", f"{sm['ROI (доходность на оборот) %']}%")
        k[2].metric("Ставок", sm["ставок"])
        k[3].metric("Макс. просадка", f"{sm['макс. просадка %']}%")
        k[4].metric("CLV", f"{sm['средний CLV %']}%")
        st.plotly_chart(px.line(r["curve"], x="date", y="bank", title="Банк в бэктесте", log_y=True),
                        width="stretch")
        st.dataframe(r["by_season"], width="stretch")
        if r["mc"]:
            st.subheader("Монте-Карло: насколько результат зависит от удачи")
            st.write(r["mc"])
        st.caption("Честно: параметры по умолчанию подобраны на этой же истории, поэтому реальный "
                   "результат, скорее всего, будет хуже. Сначала — виртуальные ставки и контроль CLV.")

# ------------------------------------------------------------------ рейтинги
with tab_tbl:
    lg3 = st.radio("Лига", ["EPL", "LaLiga"], horizontal=True)
    st.dataframe(service.strength_table(lg3), width="stretch", hide_index=True)
    st.subheader("Честность рынка (калибровка вероятностей без маржи)")
    st.dataframe(service.market_check(), width="stretch")
