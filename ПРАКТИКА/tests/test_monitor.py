"""Тесты monitor.py: миграция схемы, дифф по ключу идентичности, тело
письма (без сырых ПДн) и dry-run прогон на синтетике."""

import re
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor import build_alert_email, diff_new_records, store_new_records
from scraper import NormalizedRecord, Storage, compute_identity_hash, generate_demo_records

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rec(card: str, phone: str, added_at: str = "2026-01-01") -> NormalizedRecord:
    return NormalizedRecord(
        card=card, card_bin=card[:6], payment_system="Visa" if card.startswith("4") else "МИР",
        phone=phone, added_at=added_at, source_note="test",
    )


class SchemaMigrationTests(unittest.TestCase):
    """Проверяем, что открытие Storage на БД со старой схемой (без
    identity_hash/first_seen_at) не теряет существующие строки и
    корректно их обратно заполняет."""

    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_migration.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()

        # Создаём БД по "старой" схеме, как до появления monitor.py.
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE drops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card TEXT,
                card_bin TEXT,
                payment_system TEXT,
                phone TEXT,
                added_at TEXT,
                source_note TEXT,
                collected_at TEXT NOT NULL,
                UNIQUE(card, phone)
            );
            """
        )
        conn.execute(
            "INSERT INTO drops (card, card_bin, payment_system, phone, added_at, source_note, collected_at) "
            "VALUES ('4111111111111111', '411111', 'Visa', '+79101234567', '2025-01-01', 'legacy', '2025-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_migration_preserves_existing_row_and_backfills(self) -> None:
        storage = Storage(self.db_path)
        try:
            row = storage._conn.execute(
                "SELECT card, phone, identity_hash, first_seen_at, collected_at FROM drops"
            ).fetchone()
        finally:
            storage.close()

        card, phone, identity_hash, first_seen_at, collected_at = row
        self.assertEqual(card, "4111111111111111")
        self.assertEqual(phone, "+79101234567")
        # Старая строка не потеряна и обратно заполнена.
        self.assertEqual(identity_hash, compute_identity_hash(card, phone))
        self.assertEqual(first_seen_at, collected_at)

    def test_migration_is_idempotent(self) -> None:
        Storage(self.db_path).close()
        storage2 = Storage(self.db_path)  # повторное открытие после миграции
        try:
            self.assertEqual(storage2.count(), 1)
        finally:
            storage2.close()


class DiffNewRecordsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_diff.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = Storage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_first_run_all_new(self) -> None:
        batch = [_rec("4111111111111111", "+79101234567"), _rec("5555555555554444", "+79151234567")]
        new = diff_new_records(batch, self.storage.existing_identity_hashes())
        self.assertEqual(len(new), 2)

    def test_repeat_run_zero_new(self) -> None:
        batch = [_rec("4111111111111111", "+79101234567")]
        new = diff_new_records(batch, self.storage.existing_identity_hashes())
        store_new_records(self.storage, new)

        # Повторный запуск с тем же снимком -> идемпотентность.
        new_again = diff_new_records(batch, self.storage.existing_identity_hashes())
        self.assertEqual(len(new_again), 0)

    def test_partial_overlap_counts_as_new(self) -> None:
        """Совпадение только по одному полю пары (card, phone) — другая
        идентичность, должна считаться новой записью."""
        batch1 = [_rec("4111111111111111", "+79101234567")]
        store_new_records(self.storage, diff_new_records(batch1, self.storage.existing_identity_hashes()))

        same_card_diff_phone = _rec("4111111111111111", "+79269998877")
        same_phone_diff_card = _rec("5555555555554444", "+79101234567")
        batch2 = [same_card_diff_phone, same_phone_diff_card]
        new2 = diff_new_records(batch2, self.storage.existing_identity_hashes())
        self.assertEqual(len(new2), 2)

    def test_duplicate_within_same_batch_counted_once(self) -> None:
        batch = [_rec("4111111111111111", "+79101234567"), _rec("4111111111111111", "+79101234567")]
        new = diff_new_records(batch, self.storage.existing_identity_hashes())
        self.assertEqual(len(new), 1)

    def test_full_cycle_idempotent_via_generate_demo_records(self) -> None:
        snapshot = generate_demo_records(n=30, seed=7)
        new1 = diff_new_records(snapshot, self.storage.existing_identity_hashes())
        inserted1 = store_new_records(self.storage, new1)
        self.assertEqual(inserted1, len(new1))
        self.assertGreater(inserted1, 0)

        new2 = diff_new_records(snapshot, self.storage.existing_identity_hashes())
        self.assertEqual(len(new2), 0)


class AlertEmailContentTests(unittest.TestCase):
    def test_subject_format(self) -> None:
        records = [_rec("4111111111111111", "+79101234567")]
        subject, _ = build_alert_email(records, "2026-07-13")
        self.assertEqual(subject, "[P2P-alert] Новых: 1 (2026-07-13)")

    def test_zero_new_subject(self) -> None:
        subject, _ = build_alert_email([], "2026-07-13")
        self.assertEqual(subject, "[P2P-alert] Новых: 0 (2026-07-13)")

    def test_body_contains_no_raw_pan_or_phone(self) -> None:
        records = generate_demo_records(n=25, seed=3)
        subject, body = build_alert_email(records, "2026-07-13")
        full_text = subject + "\n" + body

        for r in records:
            self.assertNotIn(r.card, full_text)
            self.assertNotIn(r.phone, full_text)

        # Нет "голых" длинных цифровых последовательностей (потенциальный PAN/телефон).
        self.assertNotRegex(full_text, r"(?<!\*)\d{10,}(?!\*)")

    def test_body_contains_masked_preview_markers(self) -> None:
        records = generate_demo_records(n=5, seed=3)
        _, body = build_alert_email(records, "2026-07-13")
        self.assertIn("Маскированное превью", body)
        self.assertTrue(re.search(r"\d{6}\*+\d{4}", body))  # маска карты
        self.assertTrue(re.search(r"\+7\(\d{3}\)\*", body))  # маска телефона


class DryRunCliTests(unittest.TestCase):
    """Полный прогон через CLI на синтетике: --demo --dry-run не должен
    трогать сеть/почту и не должен печатать сырые ПДн."""

    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_cli.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def _run(self, extra_args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "monitor.py"), "--demo", "--n", "20", "--seed", "5",
             "--db", str(self.db_path), "--dry-run", *extra_args],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )

    def test_first_run_prints_alert_without_raw_pdn(self) -> None:
        result = self._run([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[P2P-alert] Новых:", result.stdout)
        self.assertNotRegex(result.stdout, r"(?<!\*)\d{10,}(?!\*)")

        demo_records = generate_demo_records(n=20, seed=5)
        for r in demo_records:
            self.assertNotIn(r.card, result.stdout)
            self.assertNotIn(r.phone, result.stdout)

    def test_second_run_zero_new_no_output(self) -> None:
        self._run([])  # первый запуск наполняет БД
        result = self._run([])  # повторный запуск на том же снимке
        self.assertEqual(result.returncode, 0, result.stderr)
        # Без --always-send при 0 новых письмо не печатается.
        self.assertNotIn("Subject:", result.stdout)

    def test_always_send_prints_even_with_zero_new(self) -> None:
        self._run([])
        result = self._run(["--always-send"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[P2P-alert] Новых: 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
