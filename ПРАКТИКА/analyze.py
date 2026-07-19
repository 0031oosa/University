"""Обезличенная агрегатная аналитика по собранным дроп-реквизитам.

Строит графики распределений (платёжные системы, БИН-группы, операторы
СБП по префиксу телефона, динамика во времени) и сохраняет их в
report_assets/. На вход графиков и в консольный вывод НИКОГДА не
попадают полные номера карт или телефонов — только агрегаты и счётчики
(см. п.3.4, п.7 критериев приёмки ТЗ).

Режим --demo генерирует воспроизводимый (фиксированный seed) синтетический
набор Luhn-валидных фейковых номеров через scraper.generate_demo_records
и не обращается к реальному хранилищу.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scraper import NormalizedRecord, generate_demo_records

logger = logging.getLogger("analyze")

REPORT_ASSETS_DIR = Path(__file__).parent / "report_assets"

# Приблизительная (демонстрационная) таблица соответствия префикса
# номера оператору СБП. Не претендует на точность атрибуции —
# для боевого использования подключается официальная база ABC-XYZ
# (реестр НСС/Роскомнадзора).
_OPERATOR_PREFIXES: dict[str, str] = {
    "900": "Tele2", "901": "Tele2", "902": "Tele2", "904": "Tele2",
    "910": "МТС", "915": "МТС", "916": "МТС", "917": "МТС", "919": "МТС", "926": "МТС", "928": "МТС",
    "920": "МегаФон", "921": "МегаФон", "922": "МегаФон", "923": "МегаФон", "925": "МегаФон", "931": "МегаФон",
    "903": "Билайн", "905": "Билайн", "906": "Билайн", "909": "Билайн", "929": "Билайн", "965": "Билайн", "999": "Билайн",
}


def detect_operator(phone_e164: str) -> str:
    """Определяет оператора СБП по префиксу номера (см. оговорку выше)."""
    digits = "".join(c for c in phone_e164 if c.isdigit())
    prefix = digits[1:4] if len(digits) == 11 else ""
    return _OPERATOR_PREFIXES.get(prefix, "Прочие/неизвестно")


# Загрузка данных


def load_records_from_db(db_path: str) -> list[NormalizedRecord]:
    """db_path может перечислять несколько SQLite-файлов через запятую
    ("a.sqlite,b.sqlite") — так сводятся воедино данные из разных
    источников (живой сбор с platshield.com, валидационный лист Google
    Sheets, локальный архив) в одну аналитику. Схема — sheet_drops
    (sheet_collector.py); source_note заполняется sheet_slug для
    трассируемости происхождения записи."""
    records: list[NormalizedRecord] = []
    dropped_future_dates = 0
    for path in (p.strip() for p in db_path.split(",") if p.strip()):
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                "SELECT card, card_bin, payment_system, phone, date_iso, sheet_slug, first_seen_at FROM sheet_drops"
            ).fetchall()
        finally:
            conn.close()
        for card, card_bin, payment_system, phone, date_iso, sheet_slug, first_seen_at in rows:
            # Дата источника не может быть позже момента, когда мы сами эту
            # строку реально собрали (first_seen_at) — иначе это опечатка в
            # исходнике (реальный пример: "26.05.2038"/"05.06.2040" внутри
            # вкладок за 2025 год, см. visualize.load_dataframe). Сравнение
            # строк работает благодаря ISO 8601 формату обеих дат.
            added_at = date_iso
            if date_iso and first_seen_at and date_iso > first_seen_at[:10]:
                added_at = None
                dropped_future_dates += 1
            records.append(
                NormalizedRecord(
                    card=card, card_bin=card_bin, payment_system=payment_system, phone=phone,
                    added_at=added_at, source_note=sheet_slug,
                )
            )
    if dropped_future_dates:
        logger.warning(
            "Обнаружено %d строк с датой источника позже момента сбора (вероятная опечатка "
            "в исходнике) — added_at для них не используется в помесячной динамике",
            dropped_future_dates,
        )
    return records


# Агрегаты (обезличенные — только счётчики по категориям)


def aggregate_payment_systems(records: list[NormalizedRecord]) -> Counter:
    return Counter(r.payment_system or "Неизвестно" for r in records if r.card)


def aggregate_top_bins(records: list[NormalizedRecord], top_n: int = 10) -> list[tuple[str, int]]:
    counts = Counter(r.card_bin for r in records if r.card_bin)
    return counts.most_common(top_n)


def aggregate_operators(records: list[NormalizedRecord]) -> Counter:
    return Counter(detect_operator(r.phone) for r in records if r.phone)


def aggregate_monthly_dynamics(records: list[NormalizedRecord]) -> list[tuple[str, int]]:
    counts = Counter(r.added_at[:7] for r in records if r.added_at)
    return sorted(counts.items())


# Графики


def plot_payment_systems(records: list[NormalizedRecord], out_dir: Path) -> Path:
    data = aggregate_payment_systems(records)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, values = zip(*sorted(data.items(), key=lambda kv: -kv[1])) if data else ([], [])
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title("Распределение по платёжным системам")
    ax.set_ylabel("Количество записей")
    fig.tight_layout()
    out_path = out_dir / "fig1_payment_systems.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_top_bins(records: list[NormalizedRecord], out_dir: Path) -> Path:
    data = aggregate_top_bins(records)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [b for b, _ in data]
    values = [c for _, c in data]
    ax.barh(labels[::-1], values[::-1], color="#55A868")
    ax.set_title("Топ БИН-групп по частоте")
    ax.set_xlabel("Количество записей")
    fig.tight_layout()
    out_path = out_dir / "fig2_top_bins.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_operators(records: list[NormalizedRecord], out_dir: Path) -> Path:
    data = aggregate_operators(records)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, values = zip(*sorted(data.items(), key=lambda kv: -kv[1])) if data else ([], [])
    ax.pie(values, labels=labels, autopct="%1.0f%%", colors=plt.cm.Set2.colors)
    ax.set_title("Распределение по операторам СБП (по префиксу номера)")
    fig.tight_layout()
    out_path = out_dir / "fig3_operators.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_monthly_dynamics(records: list[NormalizedRecord], out_dir: Path) -> Path:
    data = aggregate_monthly_dynamics(records)
    fig, ax = plt.subplots(figsize=(7, 4))
    months = [m for m, _ in data]
    values = [c for _, c in data]
    ax.plot(months, values, marker="o", color="#C44E52")
    ax.set_title("Динамика пополнения списка дропов по месяцам")
    ax.set_ylabel("Новых записей")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_path = out_dir / "fig4_monthly_dynamics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_all_plots(records: list[NormalizedRecord], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_payment_systems(records, out_dir),
        plot_top_bins(records, out_dir),
        plot_operators(records, out_dir),
        plot_monthly_dynamics(records, out_dir),
    ]


# CLI


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Синтетический прогон на воспроизводимых данных")
    parser.add_argument(
        "--db",
        default="platshield_data.sqlite,platshield_gsheets_test.sqlite,platshield_archive_local.sqlite",
        help="Путь(и) к SQLite-хранилищу через запятую (без --demo)",
    )
    parser.add_argument("--out-dir", default=str(REPORT_ASSETS_DIR), help="Каталог для графиков")
    parser.add_argument("--seed", type=int, default=42, help="Seed для --demo")
    parser.add_argument("--n", type=int, default=300, help="Число синтетических записей для --demo")
    args = parser.parse_args()

    if args.demo:
        records = generate_demo_records(n=args.n, seed=args.seed)
        logger.info("Сгенерировано %d синтетических записей (seed=%d)", len(records), args.seed)
    else:
        records = load_records_from_db(args.db)
        logger.info("Загружено %d записей из %s", len(records), args.db)

    paths = build_all_plots(records, Path(args.out_dir))
    for p in paths:
        logger.info("Сохранён график: %s", p)


if __name__ == "__main__":
    main()
