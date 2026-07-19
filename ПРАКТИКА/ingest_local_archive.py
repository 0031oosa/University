"""Импорт CSV-файлов архива p2p-drops, которые пользователь выгрузил сам
(со своего доступа) и положил в additional_data/ — в отличие от
sheet_collector.py, этот модуль не делает НИ ОДНОГО сетевого запроса: он
читает уже полученные файлы с диска.

Почему так, а не прямым скачиванием по ссылкам: происхождение и права
доступа к исходным Google Sheets из архива не удалось независимо
подтвердить (часть ссылок из списка, предоставленного пользователем,
требует логина). Вместо того чтобы самому обращаться к ссылкам с неясным
статусом, пользователь выгрузил нужные листы через собственный доступ и
передал файлы напрямую — это снимает вопрос об авторизации моей стороной
и делает объект работы обычным локальным файлом, как и любой другой вход
для анализа.

Формат файлов идентичен вкладкам живого API platshield.com (см.
sheet_collector.normalize_sheet_row): A=Дата, B=Номер карты/СБП,
C=Название Казино, D=Зеркало (URL), E=Платёжная система (URL), F=ФИО,
G=Название банка, H=Доп. инфо (не используется). Поэтому нормализация,
маскирование и проверка на утечки переиспользуются как есть — без
дублирования логики.

Каждый файл рассматривается как отдельный "лист" с единственной вкладкой
(tab_index=0): sheet_slug = "local_archive:<имя файла без расширения>".
Это даёт: (а) уникальный identity_hash на файл+номер_строки, поэтому
повторный запуск идемпотентен; (б) трассируемость — в export_masked_csv
видно, из какого именно файла взята строка.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from sheet_collector import SheetStorage, normalize_sheet_row

logger = logging.getLogger("ingest_local_archive")

DEFAULT_DIR = Path("additional_data")
DEFAULT_DB = "platshield_archive_local.sqlite"


def ingest_csv_file(path: Path, storage: SheetStorage) -> dict:
    """Читает один CSV-файл и вставляет его строки в storage. Возвращает
    {"seen", "inserted"}. Пустые строки (например, финальный перевод строки
    файла) пропускаются, чтобы не засорять statistics и БД пустыми записями."""
    sheet_slug = f"local_archive:{path.stem}"
    sheet_title = f"Локальный архив (файл предоставлен пользователем): {path.name}"
    seen = inserted = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # заголовок
        for row_number, row in enumerate(reader, start=1):
            if not any(cell.strip() for cell in row):
                continue
            seen += 1
            record = normalize_sheet_row(row, sheet_slug, sheet_title, path.stem, 0, row_number)
            if storage.insert(record):
                inserted += 1
    return {"seen": seen, "inserted": inserted}


def ingest_dir(dir_path: Path, storage: SheetStorage) -> dict:
    """Импортирует все *.csv в каталоге. Возвращает {"<имя файла>": {"seen", "inserted"}}."""
    stats: dict[str, dict] = {}
    for path in sorted(dir_path.glob("*.csv")):
        file_stats = ingest_csv_file(path, storage)
        stats[path.name] = file_stats
        logger.info(
            "%s: получено %d, новых вставлено %d", path.name, file_stats["seen"], file_stats["inserted"]
        )
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="Каталог с CSV-файлами архива")
    parser.add_argument("--db", default=DEFAULT_DB, help="Путь к SQLite-хранилищу")
    parser.add_argument("--export-csv", default=None, help="Путь для маскированного CSV-экспорта после импорта")
    args = parser.parse_args()

    storage = SheetStorage(args.db)
    try:
        stats = ingest_dir(Path(args.dir), storage)
        if args.export_csv:
            storage.export_masked_csv(args.export_csv)
    finally:
        storage.close()

    total_seen = sum(s["seen"] for s in stats.values())
    total_inserted = sum(s["inserted"] for s in stats.values())
    logger.info(
        "Импорт завершён. Файлов: %d, строк получено: %d, новых вставлено: %d, в БД: %s",
        len(stats), total_seen, total_inserted, args.db,
    )


if __name__ == "__main__":
    main()
