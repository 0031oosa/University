"""Полный НЕмаскированный экспорт всех трёх источников (sheet_drops) в один
CSV — по прямому запросу пользователя, для внутреннего анализа/ML, а НЕ
для отчёта.

Внимание: в отличие от export_masked_csv (sheet_collector.SheetStorage) —
единственного маршрута экспорта, безопасного для отчёта/публикации — этот
скрипт намеренно выгружает card/phone/full_name в открытом виде. Файл
содержит реальные ПДн (152-ФЗ) и не должен покидать локальную рабочую
среду: не коммитить в git, не прикладывать к отчёту, не публиковать.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("export_unmasked")

_COLUMNS = [
    "sheet_slug", "tab_name", "date_iso", "value_kind", "card", "card_bin", "payment_system",
    "phone", "casino_name", "mirror_url", "payment_channel_url", "full_name", "site_bank_name",
    "collected_at",
]


def export_unmasked_csv(db_paths: list[str], out_path: str | Path) -> int:
    out_path = Path(out_path)
    total = 0
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(_COLUMNS)
        for db_path in db_paths:
            conn = sqlite3.connect(db_path)
            try:
                rows = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM sheet_drops").fetchall()
            finally:
                conn.close()
            writer.writerows(rows)
            total += len(rows)
            logger.info("%s: %d строк", db_path, len(rows))
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", action="append", default=None,
        help="Путь к SQLite-хранилищу (флаг можно повторять); по умолчанию — все три реальных источника",
    )
    parser.add_argument("--out", default="unmasked_full_dataset.csv", help="Путь для немаскированного CSV")
    args = parser.parse_args()

    db_paths = args.db or [
        "platshield_data.sqlite", "platshield_gsheets_test.sqlite", "platshield_archive_local.sqlite",
    ]
    total = export_unmasked_csv(db_paths, args.out)
    logger.info("Готово: %d строк (немаскировано) -> %s", total, args.out)
    logger.warning("Файл содержит реальные ПДн — держите локально, не для отчёта/публикации")


if __name__ == "__main__":
    main()
