"""Тесты нормализации, маскирования и парсинга (без сети и без реальных ПДн)."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper import (
    DEFAULT_USER_AGENT,
    NormalizedRecord,
    Storage,
    build_demo_html,
    detect_payment_system,
    extract_bin,
    luhn_valid,
    mask_card,
    mask_phone,
    normalize_phone,
    normalize_record,
    parse_table,
)


class UserAgentTests(unittest.TestCase):
    def test_default_user_agent_is_valid_http_header_value(self) -> None:
        # HTTP-заголовки кодируются в latin-1; нелатинские символы (например,
        # кириллица) роняют requests.get() с UnicodeEncodeError ещё до сети —
        # реальный баг, который юнит-тесты на фиктивном fetcher'е не ловили.
        try:
            DEFAULT_USER_AGENT.encode("latin-1")
        except UnicodeEncodeError as exc:
            self.fail(f"DEFAULT_USER_AGENT не кодируется в latin-1 (невалидный HTTP-заголовок): {exc}")


class LuhnTests(unittest.TestCase):
    def test_valid_known_number(self) -> None:
        # Стандартный тестовый номер Visa, проходящий алгоритм Луна.
        self.assertTrue(luhn_valid("4111111111111111"))

    def test_invalid_number(self) -> None:
        self.assertFalse(luhn_valid("4111111111111112"))

    def test_too_short_is_invalid(self) -> None:
        self.assertFalse(luhn_valid("12345"))

    def test_ignores_non_digits(self) -> None:
        self.assertTrue(luhn_valid("4111 1111 1111 1111"))


class BinDetectionTests(unittest.TestCase):
    def test_extract_bin(self) -> None:
        self.assertEqual(extract_bin("4111 1111 1111 1111"), "411111")

    def test_visa(self) -> None:
        self.assertEqual(detect_payment_system("4111111111111111"), "Visa")

    def test_mastercard(self) -> None:
        self.assertEqual(detect_payment_system("5555555555554444"), "Mastercard")

    def test_mir(self) -> None:
        self.assertEqual(detect_payment_system("2200150000000000"), "МИР")

    def test_amex(self) -> None:
        self.assertEqual(detect_payment_system("340000000000009"), "AmEx")

    def test_unknown_range(self) -> None:
        self.assertEqual(detect_payment_system("999999999999999"), "Неизвестно")


class PhoneNormalizationTests(unittest.TestCase):
    def test_plus7_format(self) -> None:
        self.assertEqual(normalize_phone("+7 (910) 123-45-67"), "+79101234567")

    def test_8_format(self) -> None:
        self.assertEqual(normalize_phone("8 910 123 45 67"), "+79101234567")

    def test_bare_10_digits(self) -> None:
        self.assertEqual(normalize_phone("9101234567"), "+79101234567")

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(normalize_phone("не телефон"))

    def test_wrong_length_returns_none(self) -> None:
        self.assertIsNone(normalize_phone("123"))


class MaskingTests(unittest.TestCase):
    def test_mask_card_format(self) -> None:
        self.assertEqual(mask_card("2202201234561234"), "220220******1234")

    def test_mask_card_no_full_digits_leak(self) -> None:
        card = "4111111111111111"
        masked = mask_card(card)
        self.assertNotEqual(masked, card)
        # В середине не должно остаться распознаваемого блока из >4 подряд исходных цифр.
        self.assertNotIn(card[6:-4], masked)

    def test_mask_phone_format(self) -> None:
        self.assertEqual(mask_phone("+79101234567"), "+7(910)***-**-67")

    def test_mask_phone_hides_middle_digits(self) -> None:
        masked = mask_phone("+79101234567")
        # Средние 5 цифр номера (23445..) не должны присутствовать открытым текстом.
        self.assertNotRegex(masked, r"\d{5,}")


class NormalizeRecordTests(unittest.TestCase):
    def test_valid_card_and_phone(self) -> None:
        rec = normalize_record({"card": "4111 1111 1111 1111", "phone": "+7 910 123 45 67"})
        self.assertEqual(rec.card, "4111111111111111")
        self.assertEqual(rec.phone, "+79101234567")
        self.assertEqual(rec.payment_system, "Visa")

    def test_invalid_card_becomes_none(self) -> None:
        rec = normalize_record({"card": "1234567890123456", "phone": "+79101234567"})
        self.assertIsNone(rec.card)
        self.assertIsNone(rec.payment_system)
        self.assertEqual(rec.phone, "+79101234567")


class ParseTableTests(unittest.TestCase):
    def test_parses_demo_html(self) -> None:
        html = build_demo_html(n_rows=5, seed=1)
        records = parse_table(html)
        self.assertEqual(len(records), 5)
        self.assertIn("card", records[0])
        self.assertIn("phone", records[0])

    def test_positional_fallback_without_th(self) -> None:
        html = "<table><tbody><tr><td>4111111111111111</td><td>+79101234567</td></tr></tbody></table>"
        records = parse_table(html)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["card"], "4111111111111111")
        self.assertEqual(records[0]["phone"], "+79101234567")


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_storage.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = Storage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        if self.db_path.exists():
            self.db_path.unlink()
        csv_path = self.db_path.with_suffix(".csv")
        if csv_path.exists():
            csv_path.unlink()

    def test_dedup_by_card_and_phone(self) -> None:
        rec = NormalizedRecord(
            card="4111111111111111", card_bin="411111", payment_system="Visa",
            phone="+79101234567", added_at="2026-01-01", source_note="t",
        )
        self.assertTrue(self.storage.insert(rec))
        self.assertFalse(self.storage.insert(rec))
        self.assertEqual(self.storage.count(), 1)

    def test_export_csv_contains_no_raw_pan_or_phone(self) -> None:
        rec = NormalizedRecord(
            card="4111111111111111", card_bin="411111", payment_system="Visa",
            phone="+79101234567", added_at="2026-01-01", source_note="t",
        )
        self.storage.insert(rec)
        csv_path = self.db_path.with_suffix(".csv")
        self.storage.export_masked_csv(csv_path)
        content = csv_path.read_text(encoding="utf-8-sig")
        self.assertNotIn("4111111111111111", content)
        self.assertNotIn("+79101234567", content)
        self.assertNotRegex(content, r"\b\d{13,19}\b")


if __name__ == "__main__":
    unittest.main()
