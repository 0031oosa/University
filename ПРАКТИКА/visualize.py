"""Данные и агрегаты для "олл-тайм" дашборда (dashboard.py) и статического
экспорта графиков в отчёт (export_static()).

Общая точка входа для интерактивного Streamlit-дашборда и офлайн-экспорта
в PNG: обе стороны используют одни и те же функции агрегации, поэтому
графики в дашборде и в отчёте всегда согласованы.

152-ФЗ / жёсткое требование: DataFrame, возвращаемый load_dataframe(),
НИКОГДА не содержит полного номера карты или телефона — только
БИН (первые 6 цифр, уже отделены в scraper.py), платёжную систему и
3-значный префикс телефона (для приблизительного определения оператора).
Это гарантируется на уровне SQL-запроса (см. _SQL_QUERY): полные колонки
card/phone из таблицы drops сюда не выбираются вообще, то есть их не
получится случайно "протащить" дальше в графики.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analyze import _OPERATOR_PREFIXES
from scraper import generate_demo_records

logger = logging.getLogger("visualize")

REPORT_ASSETS_DIR = Path(__file__).parent / "report_assets"

OPERATOR_UNKNOWN = "Прочие/неизвестно"
BANK_UNKNOWN = "Неизвестно"

# Палитра (реф. references/palette.md skill'а dataviz — валидирована,
# используется как есть, без изменений порядка слотов).

CATEGORICAL = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}

MUTED_GRAY = "#898781"  # для "Неизвестно"/"Прочие" — сознательно вне категориальных 8 слотов

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

CHROME = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
}

# Платёжные системы — фиксированный порядок категориальных слотов 1-4,
# "Неизвестно" вынесено в приглушённый серый (это не идентичность,
# а бакет "не определено", см. color-formula.md § collision rule).
PAYMENT_SYSTEM_COLORS = {
    "Visa": CATEGORICAL["blue"],
    "МИР": CATEGORICAL["aqua"],
    "Mastercard": CATEGORICAL["yellow"],
    "AmEx": CATEGORICAL["green"],
    "Неизвестно": MUTED_GRAY,
}

# Операторы СБП — слоты 1, 5, 6, 7 (не пересекаются по факту использования
# с платёжными системами ни на одном графике, но порядок всё равно
# фиксирован и не меняется между перерисовками).
OPERATOR_COLORS = {
    "МТС": CATEGORICAL["blue"],
    "МегаФон": CATEGORICAL["violet"],
    "Билайн": CATEGORICAL["red"],
    "Tele2": CATEGORICAL["magenta"],
    OPERATOR_UNKNOWN: MUTED_GRAY,
}

BANK_SLOT_ORDER = ["blue", "aqua", "yellow", "green", "violet", "red", "magenta", "orange"]

# Только для --demo: иллюстративное сопоставление БИН -> "банк", чтобы
# показать дашборд с осмысленной группировкой без реальной БИН-базы.
# На реальных данных (без внешней БИН-базы) bank == "БИН {bin}" — см.
# resolve_bank(). Это НЕ настоящие банки, только для демонстрации UI.
_DEMO_BIN_BANK_HINTS = {
    "220220": "МИР Банк-1 (демо)",
    "220100": "МИР Банк-2 (демо)",
    "400000": "Visa Банк-3 (демо)",
    "400111": "Visa Банк-4 (демо)",
    "510000": "Mastercard Банк-5 (демо)",
    "555500": "Mastercard Банк-6 (демо)",
    "340000": "AmEx Банк-7 (демо)",
}


def resolve_bank(card_bin: str | float | None, demo: bool) -> str:
    """Атрибуция банка по БИН. Внешняя БИН-база не подключена (см.
    scraper.BinLookup) — на реальных данных банк-бакет равен самому БИН.
    В --demo используется небольшая иллюстративная таблица соответствий,
    чтобы дашборд показывал содержательную группировку "по банкам".

    Реальная находка (2026-07-15): для строк-телефонов (нет card_bin)
    pandas отдаёт из SQL не None, а float('nan') — а `not float('nan')`
    в Python равно False (NaN истинно в булевом контексте), поэтому
    старая проверка `if not card_bin` эту NaN пропускала, и 261 893
    строки (все телефонные) попадали в фиктивный бакет "БИН nan" вместо
    "Неизвестно". pd.isna() ловит и None, и NaN корректно."""
    if pd.isna(card_bin) or not card_bin:
        return BANK_UNKNOWN
    if demo:
        return _DEMO_BIN_BANK_HINTS.get(card_bin, f"БИН {card_bin} (демо)")
    return f"БИН {card_bin}"


def resolve_operator(phone_prefix: str | float | None) -> str:
    # Тот же случай NaN-truthiness, что и в resolve_bank: для строк-карт
    # phone_prefix из SQL приходит как float('nan'), не None. Здесь это
    # не давало наблюдаемой ошибки (nan не совпадает ни с одним строковым
    # ключом словаря, .get() и так возвращал OPERATOR_UNKNOWN по
    # умолчанию) — но проверка исправлена для единообразия с resolve_bank.
    if pd.isna(phone_prefix) or not phone_prefix:
        return OPERATOR_UNKNOWN
    return _OPERATOR_PREFIXES.get(phone_prefix, OPERATOR_UNKNOWN)


# Загрузка данных — только агрегируемые поля, ни card, ни phone целиком.

_SQL_QUERY = """
    SELECT
        card_bin,
        payment_system,
        substr(phone, 3, 3) AS phone_prefix,
        date_iso AS added_at,
        first_seen_at,
        (card IS NOT NULL) AS has_card
    FROM sheet_drops
"""


def _split_db_paths(db_path: str) -> list[str]:
    """"a.sqlite,b.sqlite" -> ["a.sqlite", "b.sqlite"] — единый способ задать
    несколько БД одной строкой (CLI-флаг и текстовое поле в дашборде
    остаются простыми строками, без смены типа аргумента)."""
    return [p.strip() for p in db_path.split(",") if p.strip()]


def load_dataframe(
    db_path: str | None = None,
    demo: bool = False,
    demo_n: int = 400,
    demo_seed: int = 42,
    demo_months_span: int = 18,
) -> pd.DataFrame:
    """Возвращает DataFrame с производными, безопасными для показа полями:
    card_bin, payment_system, bank, operator, event_date, first_seen_date,
    has_card. Полные PAN/телефон в него никогда не попадают.

    db_path может перечислять несколько SQLite-файлов через запятую
    ("a.sqlite,b.sqlite") — используется, чтобы свести воедино данные из
    разных источников (живой сбор с platshield.com, валидационный лист
    Google Sheets, локальный архив) в одну аналитику."""
    _DEMO_COLUMNS = ["card_bin", "payment_system", "phone_prefix", "added_at", "first_seen_at", "has_card"]
    if demo:
        records = generate_demo_records(n=demo_n, seed=demo_seed, months_span=demo_months_span)
        raw = pd.DataFrame(
            [
                {
                    "card_bin": r.card_bin,
                    "payment_system": r.payment_system,
                    "phone_prefix": (r.phone[2:5] if r.phone else None),
                    "added_at": r.added_at,
                    "first_seen_at": r.added_at,  # в демо нет отдельного момента сбора
                    "has_card": r.card is not None,
                }
                for r in records
            ],
            columns=_DEMO_COLUMNS,  # гарантирует нужные колонки и для n=0 (пустой список)
        )
    else:
        frames = []
        for path in _split_db_paths(db_path):
            conn = sqlite3.connect(path)
            try:
                frames.append(pd.read_sql_query(_SQL_QUERY, conn))
            finally:
                conn.close()
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_DEMO_COLUMNS[:-1] + ["has_card"])
        raw["has_card"] = raw["has_card"].astype(bool)

    df = raw.copy()
    df["payment_system"] = df["payment_system"].fillna("Неизвестно")
    df["bank"] = df["card_bin"].apply(lambda b: resolve_bank(b, demo=demo))
    df["operator"] = df["phone_prefix"].apply(resolve_operator)

    # added_at — дата, заявленная источником (может отсутствовать или быть
    # в произвольном формате на реальных данных); first_seen_at — момент
    # локального наблюдения (ISO, всегда есть). Для оси времени берём
    # added_at, при его отсутствии/непарсимости — first_seen_at.
    event_date = pd.to_datetime(df["added_at"], errors="coerce")
    fallback_date = pd.to_datetime(df["first_seen_at"], errors="coerce").dt.tz_localize(None)
    df["event_date"] = event_date.fillna(fallback_date)
    df["first_seen_date"] = pd.to_datetime(df["first_seen_at"], errors="coerce").dt.tz_localize(None)

    # Санитарная проверка: заявленная источником дата не может быть позже
    # момента, когда мы сами эту строку реально увидели/собрали
    # (first_seen_date) — иначе это опечатка в исходнике, а не факт.
    # Реальный пример (2026-07-14, локальный архив additional_data/):
    # строки с date_raw="26.05.2038" и "05.06.2040" внутри вкладок за
    # 2025 год — растягивали ось времени графиков на 15 лет вперёд.
    # Такие event_date обнуляются (строка не выпадает из total_records/
    # KPI, только из графиков, построенных по датам).
    impossible = df["event_date"] > df["first_seen_date"]
    if impossible.any():
        logger.warning(
            "Обнаружено %d строк с датой источника позже момента сбора (вероятная опечатка "
            "в исходнике) — event_date для них обнулён, из графиков по времени исключены",
            int(impossible.sum()),
        )
        df.loc[impossible, "event_date"] = pd.NaT

    return df


# KPI (all-time, на нефильтрованном df)


def compute_kpis(df: pd.DataFrame) -> dict:
    banks = df.loc[df["bank"] != BANK_UNKNOWN, "bank"]
    operators = df.loc[df["operator"] != OPERATOR_UNKNOWN, "operator"]
    last_update = df["first_seen_date"].max()
    return {
        "total_records": int(len(df)),
        "unique_bins": int(df["card_bin"].dropna().nunique()),
        "unique_banks": int(banks.nunique()),
        "unique_operators": int(operators.nunique()),
        "luhn_valid_share": float(df["has_card"].mean()) if len(df) else 0.0,
        "last_update": last_update if pd.notna(last_update) else None,
    }


# Агрегаты для графиков (принимают уже отфильтрованный df)


def cumulative_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Кумулятивный рост числа записей по дням (event_date)."""
    daily = df.dropna(subset=["event_date"]).groupby(df["event_date"].dt.normalize()).size()
    daily = daily.sort_index()
    out = daily.cumsum().rename("cumulative").reset_index().rename(columns={"event_date": "date"})
    return out


def new_per_period(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Число новых записей за период (freq="W" — неделя, "ME" — месяц).

    Периоды без единой записи не включаются в результат. Сбор шёл
    отдельными "островами" месяцев (см. ТЗ и провенанс источников), а не
    непрерывным мониторингом день-в-день — pd.Grouper по умолчанию
    заполняет весь календарный диапазон между первой и последней датой
    нулями, из-за чего график между островами превращался в длинную
    плоскую пустую полосу (реальный случай, 2026-07-14: ~47% дней в
    rolling_trend были нулевыми из-за многомесячных разрывов между
    источниками). Отсутствие периода в выдаче означает "не наблюдали",
    а не "наблюдали ноль записей" — разница существенна и график не
    должен эти два случая смешивать."""
    dated = df.dropna(subset=["event_date"])
    if dated.empty:
        return pd.DataFrame(columns=["period", "count"])
    grouped = dated.groupby(pd.Grouper(key="event_date", freq=freq)).size()
    grouped = grouped[grouped > 0]
    return grouped.rename("count").reset_index().rename(columns={"event_date": "period"})


def calendar_heatmap_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Сетка (год, неделя ISO, день недели, число записей) — весь период."""
    dated = df.dropna(subset=["event_date"]).copy()
    if dated.empty:
        return pd.DataFrame(columns=["year", "iso_week", "weekday", "count"])
    iso = dated["event_date"].dt.isocalendar()
    dated["year"] = iso["year"]
    dated["iso_week"] = iso["week"]
    dated["weekday"] = iso["day"]  # 1=пн .. 7=вс
    grid = dated.groupby(["year", "iso_week", "weekday"]).size().rename("count").reset_index()
    return grid


def bank_system_matrix(df: pd.DataFrame, top_n_banks: int = 8) -> pd.DataFrame:
    """Сводная таблица bank × payment_system (счётчики), топ-N банков +
    остальные схлопнуты в "Прочие"."""
    if df.empty:
        return pd.DataFrame()
    top_banks = df["bank"].value_counts().head(top_n_banks).index.tolist()
    bucketed = df["bank"].where(df["bank"].isin(top_banks), other="Прочие")
    pivot = pd.crosstab(bucketed, df["payment_system"])
    return pivot


def treemap_data(df: pd.DataFrame, top_n_banks: int = 10) -> pd.DataFrame:
    """(payment_system, bank, count) для treemap; банки вне топ-N по
    платёжной системе схлопываются в "Прочие"."""
    if df.empty:
        return pd.DataFrame(columns=["payment_system", "bank", "count"])
    counts = df.groupby(["payment_system", "bank"]).size().rename("count").reset_index()

    buckets: list[pd.DataFrame] = []
    for system in counts["payment_system"].unique():
        group = counts[counts["payment_system"] == system].sort_values("count", ascending=False)
        if len(group) <= top_n_banks:
            buckets.append(group)
            continue
        head, tail = group.iloc[:top_n_banks], group.iloc[top_n_banks:]
        buckets.append(head)
        buckets.append(pd.DataFrame([{"payment_system": system, "bank": "Прочие", "count": tail["count"].sum()}]))

    return pd.concat(buckets, ignore_index=True) if buckets else counts


def operator_share_over_time(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Доля каждого оператора СБП по периодам (для 100%-stacked area)."""
    dated = df.dropna(subset=["event_date"])
    if dated.empty:
        return pd.DataFrame(columns=["period", "operator", "share"])
    counts = dated.groupby([pd.Grouper(key="event_date", freq=freq), "operator"]).size().rename("count").reset_index()
    totals = counts.groupby("event_date")["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts.rename(columns={"event_date": "period"})


def rolling_trend(df: pd.DataFrame, freq: str = "D", window: int = 7) -> pd.DataFrame:
    """Динамика пополнения (freq) + скользящее среднее по window периодам.

    new_per_period уже не заполняет нулями периоды без данных — но
    line-plot по умолчанию всё равно соединил бы соседние точки прямой,
    рисуя ложный линейный тренд через многомесячный разрыв между
    "островами" сбора (реальный случай, 2026-07-14: видно было бы плавное
    "снижение" с октября 2025 по март 2026, хотя данных там просто нет).
    Поэтому в местах разрыва (интервал между соседними точками больше
    одного шага freq) вставляется строка NaN — matplotlib/Plotly рвут
    линию на NaN вместо того, чтобы её продолжать."""
    daily = new_per_period(df, freq=freq).rename(columns={"period": "date"})
    if daily.empty:
        return daily.assign(rolling_mean=[])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["rolling_mean"] = daily["count"].rolling(window=window, min_periods=1).mean()

    step = pd.tseries.frequencies.to_offset(freq)
    prev_date = daily["date"].shift(1)
    gaps = daily["date"] > (prev_date + step)
    if gaps.any():
        midpoints = prev_date[gaps] + (daily.loc[gaps, "date"] - prev_date[gaps]) / 2
        break_rows = pd.DataFrame({"date": midpoints.to_numpy(), "count": np.nan, "rolling_mean": np.nan})
        daily = pd.concat([daily, break_rows], ignore_index=True).sort_values("date").reset_index(drop=True)
    return daily


# Статический экспорт (без Streamlit) — matplotlib, PNG в report_assets/


def _style_axes(ax) -> None:
    ax.set_facecolor(CHROME["surface"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHROME["baseline"])
    ax.spines["bottom"].set_color(CHROME["baseline"])
    ax.tick_params(colors=CHROME["text_secondary"])
    ax.grid(axis="y", color=CHROME["gridline"], linewidth=0.8)
    ax.set_axisbelow(True)


def export_cumulative_growth(df: pd.DataFrame, out_dir: Path) -> Path:
    data = cumulative_growth(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(data["date"], data["cumulative"], color=PAYMENT_SYSTEM_COLORS["Visa"], alpha=0.25)
    ax.plot(data["date"], data["cumulative"], color=PAYMENT_SYSTEM_COLORS["Visa"], linewidth=2)
    ax.set_title("Кумулятивный рост списка дропов (весь период)")
    ax.set_ylabel("Всего записей (накопительно)")
    _style_axes(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / "fig5_cumulative_growth.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_calendar_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    grid = calendar_heatmap_grid(df)
    years = sorted(grid["year"].unique()) if not grid.empty else []
    fig, axes = plt.subplots(max(len(years), 1), 1, figsize=(10, 2.2 * max(len(years), 1)), squeeze=False)
    weekday_labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    for i, year in enumerate(years or [None]):
        ax = axes[i][0]
        if year is None:
            ax.set_visible(False)
            continue
        year_grid = grid[grid["year"] == year]
        # reindex на полный диапазон недель ISO (1..53) — иначе недели без
        # единой записи молча выпадают из pivot и ось X сжимается, а её
        # подписи (позиционные индексы) перестают соответствовать реальным
        # номерам недель.
        all_weeks = range(1, 54)
        pivot = (
            year_grid.pivot(index="weekday", columns="iso_week", values="count")
            .reindex(index=range(1, 8), columns=all_weeks)
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0)
        ax.set_yticks(range(7))
        ax.set_yticklabels(weekday_labels, fontsize=8)
        tick_positions = list(range(0, len(all_weeks), 4))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([all_weeks[i] for i in tick_positions], fontsize=8)
        ax.set_title(f"{year}", loc="left", fontsize=10, color=CHROME["text_secondary"])
        ax.set_xlabel("Неделя года (ISO)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Записей/день")
    fig.suptitle("Календарный хитмап добавлений (весь период)")
    fig.tight_layout()
    path = out_dir / "fig6_calendar_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_bank_system_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    pivot = bank_system_matrix(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    if pivot.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for yi in range(pivot.shape[0]):
            for xi in range(pivot.shape[1]):
                val = pivot.values[yi, xi]
                if val:
                    ax.text(xi, yi, str(val), ha="center", va="center", fontsize=8, color=CHROME["text_primary"])
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Записей")
    ax.set_title("Банк × платёжная система")
    fig.tight_layout()
    path = out_dir / "fig7_bank_system_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_bank_system_shares(df: pd.DataFrame, out_dir: Path) -> Path:
    """Горизонтальный stacked-bar долей банков внутри каждой платёжной
    системы. Использует ту же группировку (top-N банков ГЛОБАЛЬНО +
    "Прочие"), что и fig7 (bank_system_matrix) — для согласованности между
    двумя графиками одного и того же среза "банк × платёжная система".

    Раньше строился через treemap_data() (top-N ПО КАЖДОЙ платёжной
    системе отдельно) — реальная находка (2026-07-15): это давало до ~40
    строк в легенде (до 10 БИН на каждую из 4 систем), что было незаметно,
    пока сама легенда обрезалась за пределами PNG (см. правку
    bbox_inches="tight" в fig.savefig); после починки обрезки стало видно,
    что легенда физически нечитаема."""
    pivot = bank_system_matrix(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    if pivot.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
    else:
        systems = pivot.columns.tolist()
        banks = pivot.index.tolist()
        bottoms = np.zeros(len(systems))
        for i, bank in enumerate(banks):
            color = MUTED_GRAY if bank == "Прочие" else list(CATEGORICAL.values())[i % len(CATEGORICAL)]
            values = pivot.loc[bank, systems].to_numpy()
            ax.barh(systems, values, left=bottoms, label=bank, color=color)
            bottoms += values
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_title("Доля реквизитов по банкам внутри платёжной системы")
    ax.set_xlabel("Записей")
    _style_axes(ax)
    fig.tight_layout()
    path = out_dir / "fig8_bank_shares.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_operator_share_over_time(df: pd.DataFrame, out_dir: Path) -> Path:
    data = operator_share_over_time(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    if data.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
    else:
        pivot = data.pivot(index="period", columns="operator", values="share").fillna(0)
        colors = [OPERATOR_COLORS.get(op, MUTED_GRAY) for op in pivot.columns]
        ax.stackplot(pivot.index, pivot.values.T, labels=pivot.columns, colors=colors)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
        fig.autofmt_xdate()
    ax.set_title("Доля операторов СБП во времени")
    _style_axes(ax)
    fig.tight_layout()
    path = out_dir / "fig9_operator_share_over_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_rolling_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    data = rolling_trend(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not data.empty:
        ax.plot(data["date"], data["count"], color=CHROME["muted"], linewidth=1, label="Новых записей/день")
        ax.plot(
            data["date"], data["rolling_mean"], color=PAYMENT_SYSTEM_COLORS["Visa"], linewidth=2,
            label="Скользящее среднее (7 дней с данными)",
        )
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        fig.autofmt_xdate()
    # "7 дней с данными", а не "7 календарных дней подряд": пустые периоды
    # между "островами" сбора (см. new_per_period) исключены из ряда, окно
    # скользящего среднего считается по оставшимся точкам — иначе разрыв
    # в несколько месяцев между источниками смешивался бы в среднее с
    # семью реальными днями. Разрыв в ряду (см. rolling_trend) не рисуется
    # соединяющей линией. Название — в две строки, чтобы не обрезалось.
    ax.set_title("Динамика пополнения со скользящим средним\n(7 дней с данными; разрывы между периодами сбора не соединяются линией)", fontsize=10)
    ax.set_ylabel("Новых записей")
    _style_axes(ax)
    fig.tight_layout()
    path = out_dir / "fig10_rolling_trend.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_static(df: pd.DataFrame, out_dir: Path | str = REPORT_ASSETS_DIR) -> list[Path]:
    """Рендерит статические PNG-версии дашборд-графиков (без Streamlit) —
    для вставки в отчёт по ГОСТ. Использует те же функции агрегации, что
    и интерактивный dashboard.py. На вход подаётся df из load_dataframe();
    полные PAN/телефоны в df физически отсутствуют (см. докстринг модуля)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        export_cumulative_growth(df, out_dir),
        export_calendar_heatmap(df, out_dir),
        export_bank_system_heatmap(df, out_dir),
        export_bank_system_shares(df, out_dir),
        export_operator_share_over_time(df, out_dir),
        export_rolling_trend(df, out_dir),
    ]


# CLI


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Синтетические данные вместо SQLite")
    parser.add_argument(
        "--db",
        default="platshield_data.sqlite,platshield_gsheets_test.sqlite,platshield_archive_local.sqlite",
        help="Путь(и) к SQLite-хранилищу через запятую (несколько БД сводятся в одну аналитику)",
    )
    parser.add_argument("--n", type=int, default=400, help="Число синтетических записей для --demo")
    parser.add_argument("--seed", type=int, default=42, help="Seed для --demo")
    parser.add_argument("--out-dir", default=str(REPORT_ASSETS_DIR), help="Каталог для PNG")
    args = parser.parse_args()

    df = load_dataframe(db_path=args.db, demo=args.demo, demo_n=args.n, demo_seed=args.seed)
    kpis = compute_kpis(df)
    logger.info(
        "KPI: записей=%d, БИН=%d, банков=%d, операторов=%d, доля Луна=%.1f%%, обновлено=%s",
        kpis["total_records"], kpis["unique_bins"], kpis["unique_banks"], kpis["unique_operators"],
        kpis["luhn_valid_share"] * 100, kpis["last_update"],
    )
    paths = export_static(df, args.out_dir)
    for p in paths:
        logger.info("Сохранён график: %s", p)


if __name__ == "__main__":
    main()
