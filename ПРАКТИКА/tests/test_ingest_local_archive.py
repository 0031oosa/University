"""Тесты ingest_local_archive.py на фиктивных CSV-файлах во временном
каталоге — без обращения к реальному additional_data/ и без единого
сетевого запроса. Проверяем: чтение CSV с BOM, пропуск пустых строк,
идемпотентность повторного импорта и корректную привязку sheet_slug/
tab_name к конкретному файлу."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest_local_archive import ingest_csv_file, ingest_dir
from sheet_collector import SheetStorage

REAL_CARD = "4111111111111111"  # Luhn-валидный тестовый номер
REAL_PHONE_RAW = "79101234567"
REAL_NAME = "Иванов Иван"

CSV_HEADER = "Дата,Номер карты,Название Казино,Зеркало (URL),Платежная система (URL),ФИО,Название банка,Доп. инфо\n"


class IngestCsvFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(__file__).parent / "_tmp_test_ingest_local_archive"
        self.tmp_dir.mkdir(exist_ok=True)
        self.db_path = self.tmp_dir / "_tmp.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = SheetStorage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        for p in self.tmp_dir.glob("*"):
            p.unlink()
        self.tmp_dir.rmdir()

    def _write_csv(self, name: str, body: str) -> Path:
        path = self.tmp_dir / name
        path.write_text("﻿" + CSV_HEADER + body, encoding="utf-8")
        return path

    def test_rows_are_ingested_with_file_derived_slug_and_tab_name(self) -> None:
        path = self._write_csv(
            "p2p_drops (65)МАЙ 2025.xlsx - Список карт номиналов. Май25.csv",
            f"01.05.2025,{REAL_CARD},ЧЕМПИОН,https://example.com/ru,https://pay.example/inv/1,{REAL_NAME},Сбербанк,\n",
        )
        stats = ingest_csv_file(path, self.storage)
        self.assertEqual(stats, {"seen": 1, "inserted": 1})

        row = self.storage._conn.execute(
            "SELECT sheet_slug, tab_name, value_kind, card, full_name FROM sheet_drops"
        ).fetchone()
        sheet_slug, tab_name, value_kind, card, full_name = row
        self.assertEqual(sheet_slug, f"local_archive:{path.stem}")
        self.assertEqual(tab_name, path.stem)
        self.assertEqual(value_kind, "card")
        self.assertEqual(card, REAL_CARD)
        self.assertEqual(full_name, REAL_NAME)

    def test_blank_trailing_lines_are_skipped(self) -> None:
        path = self._write_csv(
            "sample.csv",
            f"01.05.2025,{REAL_PHONE_RAW},Casino,,,,Альфа,\n\n",
        )
        stats = ingest_csv_file(path, self.storage)
        self.assertEqual(stats, {"seen": 1, "inserted": 1})

    def test_rerunning_same_file_is_idempotent(self) -> None:
        path = self._write_csv(
            "sample.csv",
            f"01.05.2025,{REAL_PHONE_RAW},Casino,,,,Альфа,\n"
            f"02.05.2025,{REAL_CARD},Casino,,,,Сбербанк,\n",
        )
        first = ingest_csv_file(path, self.storage)
        second = ingest_csv_file(path, self.storage)
        self.assertEqual(first, {"seen": 2, "inserted": 2})
        self.assertEqual(second, {"seen": 2, "inserted": 0})
        self.assertEqual(self.storage.count(), 2)

    def test_different_files_do_not_collide_on_identity(self) -> None:
        # Одинаковый номер строки (1) в двух разных файлах не должен
        # схлопываться дедупом — sheet_slug у них разный.
        path_a = self._write_csv("a.csv", f"01.05.2025,{REAL_CARD},Casino,,,,Сбербанк,\n")
        path_b = self._write_csv("b.csv", f"01.05.2025,{REAL_CARD},Casino,,,,Сбербанк,\n")
        ingest_csv_file(path_a, self.storage)
        ingest_csv_file(path_b, self.storage)
        self.assertEqual(self.storage.count(), 2)


class IngestDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(__file__).parent / "_tmp_test_ingest_local_archive_dir"
        self.tmp_dir.mkdir(exist_ok=True)
        self.db_path = self.tmp_dir / "_tmp.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = SheetStorage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        for p in self.tmp_dir.glob("*"):
            p.unlink()
        self.tmp_dir.rmdir()

    def test_all_csv_files_in_dir_are_ingested(self) -> None:
        (self.tmp_dir / "one.csv").write_text(
            "﻿" + CSV_HEADER + f"01.05.2025,{REAL_CARD},Casino,,,,Сбербанк,\n", encoding="utf-8"
        )
        (self.tmp_dir / "two.csv").write_text(
            "﻿" + CSV_HEADER + f"01.05.2025,{REAL_PHONE_RAW},Casino,,,,Альфа,\n", encoding="utf-8"
        )
        stats = ingest_dir(self.tmp_dir, self.storage)
        self.assertEqual(set(stats.keys()), {"one.csv", "two.csv"})
        self.assertEqual(self.storage.count(), 2)


if __name__ == "__main__":
    unittest.main()
