"""Интерактивный "олл-тайм" дашборд поверх агрегатов дроп-реквизитов.

Запуск:
    streamlit run dashboard.py -- --demo
    streamlit run dashboard.py -- --db platshield_data.sqlite
    streamlit run dashboard.py -- --db platshield_data.sqlite,platshield_gsheets_test.sqlite,platshield_archive_local.sqlite

Данные и агрегаты — из visualize.py (общий модуль с export_static(), чтобы
интерактивные и статические графики были согласованы). Здесь — только UI:
KPI-карточки, сайдбар-фильтры и Plotly-графики.

152-ФЗ / жёсткое требование: сюда попадает только DataFrame из
visualize.load_dataframe(), в котором физически нет колонок card/phone
(см. докстринг visualize.py) — соответственно ни один график, тултип или
подпись на этой странице не может показать полный PAN или телефон.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.express import treemap as px_treemap
from plotly.subplots import make_subplots

import visualize as v

# CLI-аргументы дашборда (после "--" в `streamlit run dashboard.py -- ...`)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--db", default="platshield_data.sqlite,platshield_gsheets_test.sqlite,platshield_archive_local.sqlite"
    )
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


CLI_ARGS = _parse_args()

st.set_page_config(page_title="P2P-дропы: олл-тайм аналитика", layout="wide", page_icon="📊")

# Тема (references/palette.md skill'а dataviz — валидированный дефолт,
# применяется как есть). Светлая/тёмная — через prefers-color-scheme,
# независимо от темы самого Streamlit.

_CSS = f"""
<style>
.viz-root {{
  --surface-1: {v.CHROME["surface"]};
  --text-primary: {v.CHROME["text_primary"]};
  --text-secondary: {v.CHROME["text_secondary"]};
  --muted: {v.CHROME["muted"]};
  --border: rgba(11,11,11,0.10);
}}
@media (prefers-color-scheme: dark) {{
  .viz-root {{
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --border: rgba(255,255,255,0.10);
  }}
}}
.kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.2rem; }}
.kpi-card {{
  flex: 1 1 150px;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}}
.kpi-label {{ font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 4px; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 600; color: var(--text-primary); font-variant-numeric: tabular-nums; }}
</style>
<div class="viz-root"></div>
"""
st.markdown(_CSS, unsafe_allow_html=True)

FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=v.CHROME["text_primary"])
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=v.CHROME["surface"],
    font=FONT,
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor=v.CHROME["surface"], font=FONT),
)


# Загрузка данных (кэш — чтобы фильтры не дёргали SQLite/генератор заново)


@st.cache_data(show_spinner="Загрузка данных…")
def _load(demo: bool, db_path: str, n: int, seed: int) -> pd.DataFrame:
    return v.load_dataframe(db_path=db_path, demo=demo, demo_n=n, demo_seed=seed)


def render_kpi_cards(kpis: dict) -> None:
    last_update = kpis["last_update"].strftime("%Y-%m-%d %H:%M") if kpis["last_update"] is not None else "—"
    cards = [
        ("Всего записей", f"{kpis['total_records']:,}".replace(",", " ")),
        ("Уникальных БИН", f"{kpis['unique_bins']:,}".replace(",", " ")),
        ("Банков", f"{kpis['unique_banks']:,}".replace(",", " ")),
        ("Операторов СБП", f"{kpis['unique_operators']:,}".replace(",", " ")),
        ("Доля с валидной по Луну картой", f"{kpis['luhn_valid_share'] * 100:.1f}%"),
        ("Последнее обновление", last_update),
    ]
    html = ['<div class="viz-root"><div class="kpi-row">']
    for label, value in cards:
        html.append(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>'
        )
    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# Графики (Plotly) — каждый принимает уже отфильтрованный df


def chart_cumulative_growth(df: pd.DataFrame) -> go.Figure:
    data = v.cumulative_growth(df)
    fig = go.Figure(
        go.Scatter(
            x=data["date"], y=data["cumulative"], mode="lines", fill="tozeroy",
            line=dict(color=v.PAYMENT_SYSTEM_COLORS["Visa"], width=2),
            fillcolor="rgba(42,120,214,0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>Всего: %{y}<extra></extra>",
            name="Всего записей",
        )
    )
    fig.update_layout(title="Кумулятивный рост списка (весь период)", showlegend=False, **PLOT_LAYOUT)
    fig.update_yaxes(title="Записей, накопительно", gridcolor=v.CHROME["gridline"])
    fig.update_xaxes(gridcolor=v.CHROME["gridline"])
    return fig


def chart_new_per_period(df: pd.DataFrame, freq: str) -> go.Figure:
    data = v.new_per_period(df, freq=freq)
    label = "неделю" if freq == "W" else "месяц"
    fig = go.Figure(
        go.Bar(
            x=data["period"], y=data["count"], marker_color=v.PAYMENT_SYSTEM_COLORS["Visa"],
            hovertemplate="%{x|%Y-%m-%d}<br>Новых: %{y}<extra></extra>",
        )
    )
    fig.update_layout(title=f"Новых записей за {label}", showlegend=False, **PLOT_LAYOUT)
    fig.update_yaxes(title="Новых записей", gridcolor=v.CHROME["gridline"])
    fig.update_xaxes(gridcolor=v.CHROME["gridline"])
    return fig


def chart_calendar_heatmap(df: pd.DataFrame) -> go.Figure:
    grid = v.calendar_heatmap_grid(df)
    weekday_labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    years = sorted(grid["year"].unique()) if not grid.empty else []
    if not years:
        fig = go.Figure()
        fig.update_layout(title="Календарный хитмап добавлений — нет данных", **PLOT_LAYOUT)
        return fig

    fig = make_subplots(rows=len(years), cols=1, subplot_titles=[str(y) for y in years], vertical_spacing=0.12)
    all_weeks = list(range(1, 54))
    for i, year in enumerate(years, start=1):
        year_grid = grid[grid["year"] == year]
        pivot = (
            year_grid.pivot(index="weekday", columns="iso_week", values="count")
            .reindex(index=range(1, 8), columns=all_weeks)
        )
        fig.add_trace(
            go.Heatmap(
                z=pivot.values, x=all_weeks, y=weekday_labels, colorscale="Blues", zmin=0,
                showscale=(i == 1), colorbar=dict(title="Записей/день") if i == 1 else None,
                hovertemplate="Неделя %{x}, %{y}<br>Записей: %{z}<extra></extra>",
                xgap=2, ygap=2,
            ),
            row=i, col=1,
        )
        fig.update_xaxes(title="Неделя года (ISO)" if i == len(years) else None, row=i, col=1)
    fig.update_layout(title="Календарный хитмап добавлений (весь период)", height=220 * len(years) + 60, **PLOT_LAYOUT)
    return fig


def chart_bank_system_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = v.bank_system_matrix(df)
    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(title="Банк × платёжная система — нет данных", **PLOT_LAYOUT)
        return fig
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values, x=list(pivot.columns), y=list(pivot.index), colorscale="Blues",
            text=pivot.values, texttemplate="%{text}", textfont=dict(color=v.CHROME["text_primary"]),
            hovertemplate="%{y} × %{x}<br>Записей: %{z}<extra></extra>",
            colorbar=dict(title="Записей"), xgap=2, ygap=2,
        )
    )
    fig.update_layout(title="Банк × платёжная система", **PLOT_LAYOUT)
    return fig


def chart_treemap(df: pd.DataFrame) -> go.Figure:
    data = v.treemap_data(df)
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title="Банки/системы — нет данных", **PLOT_LAYOUT)
        return fig
    color_map = dict(v.PAYMENT_SYSTEM_COLORS)
    color_map["Прочие"] = v.MUTED_GRAY
    fig = px_treemap(
        data, path=["payment_system", "bank"], values="count",
        color="payment_system", color_discrete_map=color_map,
    )
    fig.update_traces(
        hovertemplate="%{label}<br>Записей: %{value}<br>Доля от родителя: %{percentParent:.1%}<extra></extra>",
        marker=dict(line=dict(color=v.CHROME["surface"], width=2)),
    )
    fig.update_layout(title="Доля реквизитов по банкам/системам", **PLOT_LAYOUT)
    return fig


def chart_operator_share_over_time(df: pd.DataFrame) -> go.Figure:
    data = v.operator_share_over_time(df)
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title="Доля операторов СБП — нет данных", **PLOT_LAYOUT)
        return fig
    pivot = data.pivot(index="period", columns="operator", values="share").fillna(0)
    order = [op for op in ["МТС", "МегаФон", "Билайн", "Tele2", v.OPERATOR_UNKNOWN] if op in pivot.columns]
    fig = go.Figure()
    for operator in order:
        fig.add_trace(
            go.Scatter(
                x=pivot.index, y=pivot[operator], name=operator, stackgroup="one", mode="lines",
                line=dict(width=0.5, color=v.OPERATOR_COLORS.get(operator, v.MUTED_GRAY)),
                fillcolor=v.OPERATOR_COLORS.get(operator, v.MUTED_GRAY),
                hovertemplate=f"{operator}: " + "%{y:.0%}<extra></extra>",
            )
        )
    fig.update_layout(title="Доля операторов СБП во времени", yaxis_tickformat=".0%", **PLOT_LAYOUT)
    fig.update_yaxes(title="Доля", gridcolor=v.CHROME["gridline"])
    return fig


def chart_rolling_trend(df: pd.DataFrame) -> go.Figure:
    data = v.rolling_trend(df, freq="D", window=7)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"], y=data["count"], name="Новых записей/день", mode="lines",
            line=dict(color=v.CHROME["muted"], width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>Новых: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"], y=data["rolling_mean"], name="Скользящее среднее (7 дней с данными)", mode="lines",
            line=dict(color=v.PAYMENT_SYSTEM_COLORS["Visa"], width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Среднее: %{y:.1f}<extra></extra>",
        )
    )
    # "7 дней с данными" — не 7 календарных дней подряд: пустые периоды
    # между "островами" сбора исключены из ряда (см. v.new_per_period), так
    # что окно скользящего среднего не размазывается через межмесячные
    # разрывы; сам разрыв — NaN-строка (см. v.rolling_trend), поэтому линия
    # физически не соединяет разные "острова" сбора.
    fig.update_layout(
        title="Динамика пополнения со скользящим средним<br><sup>7 дней с данными; разрывы между периодами сбора не соединяются линией</sup>",
        **PLOT_LAYOUT,
    )
    fig.update_yaxes(title="Новых записей", gridcolor=v.CHROME["gridline"])
    fig.update_xaxes(gridcolor=v.CHROME["gridline"])
    return fig


# Фильтры


def apply_filters(
    df: pd.DataFrame, date_range: tuple, systems: list[str], banks: list[str], operators: list[str]
) -> pd.DataFrame:
    filtered = df
    if date_range and len(date_range) == 2 and all(date_range):
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[filtered["event_date"].between(start, end)]
    if systems:
        filtered = filtered[filtered["payment_system"].isin(systems)]
    if banks:
        filtered = filtered[filtered["bank"].isin(banks)]
    if operators:
        filtered = filtered[filtered["operator"].isin(operators)]
    return filtered


# Страница


def main() -> None:
    st.title("P2P-дропы: олл-тайм аналитика")
    st.caption(
        "Только агрегированные и маскированные значения. Полные номера карт и телефонов "
        "в дашборд не загружаются (см. visualize.load_dataframe)."
    )

    with st.sidebar:
        st.header("Источник данных")
        source_demo = st.toggle("Синтетика (--demo)", value=CLI_ARGS.demo)
        db_path = CLI_ARGS.db
        if not source_demo:
            db_path = st.text_input("Путь к SQLite (через запятую — несколько БД)", value=CLI_ARGS.db)

        df_all = _load(demo=source_demo, db_path=db_path, n=CLI_ARGS.n, seed=CLI_ARGS.seed)

        st.header("Фильтры")
        if df_all["event_date"].notna().any():
            min_date = df_all["event_date"].min().date()
            max_date = df_all["event_date"].max().date()
            date_range = st.date_input("Период", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            date_range = None

        systems = st.multiselect(
            "Платёжная система", sorted(df_all["payment_system"].unique()), default=None,
            placeholder="Все",
        )
        banks = st.multiselect(
            "Банк", sorted(df_all["bank"].unique()), default=None, placeholder="Все",
        )
        operators = st.multiselect(
            "Оператор СБП", sorted(df_all["operator"].unique()), default=None, placeholder="Все",
        )

    if df_all.empty:
        st.warning("Нет данных для отображения.")
        return

    kpis = v.compute_kpis(df_all)  # all-time — не зависит от фильтров
    render_kpi_cards(kpis)

    filtered = apply_filters(df_all, date_range, systems, banks, operators)
    st.caption(f"Отфильтровано записей: {len(filtered)} из {len(df_all)}")

    st.plotly_chart(chart_cumulative_growth(filtered), width="stretch")

    period_choice = st.radio("Группировка новых записей", ["Неделя", "Месяц"], horizontal=True)
    freq = "W" if period_choice == "Неделя" else "ME"
    st.plotly_chart(chart_new_per_period(filtered, freq=freq), width="stretch")

    st.plotly_chart(chart_calendar_heatmap(filtered), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_bank_system_heatmap(filtered), width="stretch")
    with col2:
        st.plotly_chart(chart_treemap(filtered), width="stretch")

    st.plotly_chart(chart_operator_share_over_time(filtered), width="stretch")
    st.plotly_chart(chart_rolling_trend(filtered), width="stretch")


if __name__ == "__main__":
    main()
