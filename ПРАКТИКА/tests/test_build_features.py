"""Тесты build_features.py: агрегация по уникальной сущности (card/phone),
подсчёт повторяемости и связности через общие площадки, отсутствие сырых
card/phone в маскированном экспорте признаков."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_features import build_entity_features, export_masked_features_csv, load_raw
from sheet_collector import SheetStorage, normalize_sheet_row

REAL_CARD_A = "4111111111111111"  # Luhn-валидный тестовый номер
REAL_CARD_B = "5500000000000004"  # Luhn-валидный тестовый номер (Mastercard-диапазон)
REAL_PHONE = "+79101234567"


def _row(date, value, casino, mirror, pay, bank=""):
    return normalize_sheet_row(
        [date, value, casino, mirror, pay, "", bank, ""],
        sheet_slug="s1", sheet_title="Т", tab_name="Карты", tab_index=0, row_number=1,
    )


class BuildEntityFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_build_features.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = SheetStorage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def _insert(self, *records) -> None:
        for i, r in enumerate(records, start=1):
            r.row_number = i
            self.storage.insert(r)

    def test_occurrences_counts_repeats_across_rows(self) -> None:
        self._insert(
            _row("01.05.2025", REAL_CARD_A, "CasinoA", "https://a.example", "https://pay-a.example"),
            _row("02.05.2025", REAL_CARD_A, "CasinoB", "https://b.example", "https://pay-b.example"),
            _row("03.05.2025", REAL_CARD_A, "CasinoA", "https://a.example", "https://pay-a.example"),
        )
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        self.assertEqual(len(features), 1)
        row = features.iloc[0]
        self.assertEqual(row["occurrences"], 3)
        self.assertEqual(row["distinct_casinos"], 2)
        self.assertEqual(row["distinct_mirror_domains"], 2)
        self.assertEqual(row["distinct_payment_domains"], 2)

    def test_different_entities_not_merged(self) -> None:
        self._insert(
            _row("01.05.2025", REAL_CARD_A, "CasinoA", "https://a.example", "https://pay-a.example"),
            _row("01.05.2025", REAL_CARD_B, "CasinoA", "https://a.example", "https://pay-a.example"),
            _row("01.05.2025", REAL_PHONE, "CasinoA", "https://a.example", "https://pay-a.example"),
        )
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        self.assertEqual(len(features), 3)
        self.assertEqual(set(features["value_kind"]), {"card", "phone"})

    def test_active_span_days_computed_from_first_last_seen(self) -> None:
        self._insert(
            _row("01.05.2025", REAL_CARD_A, "CasinoA", "", ""),
            _row("10.05.2025", REAL_CARD_A, "CasinoA", "", ""),
        )
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        self.assertEqual(features.iloc[0]["active_span_days"], 9)

    def test_shared_casino_degree_reflects_cluster_size(self) -> None:
        # CasinoA обслуживает 3 разные сущности -> у каждой co_entities=2.
        # CasinoB обслуживает только REAL_CARD_A -> не увеличивает признак.
        self._insert(
            _row("01.05.2025", REAL_CARD_A, "CasinoA", "", ""),
            _row("01.05.2025", REAL_CARD_B, "CasinoA", "", ""),
            _row("01.05.2025", REAL_PHONE, "CasinoA", "", ""),
            _row("02.05.2025", REAL_CARD_A, "CasinoB", "", ""),
        )
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        by_entity = features.set_index("entity_value")
        self.assertEqual(by_entity.loc[REAL_CARD_A, "max_shared_casino_degree"], 2)
        self.assertEqual(by_entity.loc[REAL_CARD_B, "max_shared_casino_degree"], 2)

    def test_isolated_entity_has_zero_shared_degree(self) -> None:
        self._insert(_row("01.05.2025", REAL_CARD_A, "CasinoOnlyMe", "", ""))
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        self.assertEqual(features.iloc[0]["max_shared_casino_degree"], 0)

    def test_phone_operator_resolved_from_prefix(self) -> None:
        self._insert(_row("01.05.2025", REAL_PHONE, "Casino", "", ""))
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        self.assertEqual(features.iloc[0]["operator"], "МТС")

    def test_empty_db_returns_empty_dataframe_without_error(self) -> None:
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        self.assertEqual(len(features), 0)
        self.assertIn("entity_hash", features.columns)


class ExportMaskedFeaturesCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_build_features_export.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = SheetStorage(self.db_path)
        self.csv_path = Path(__file__).parent / "_tmp_test_entity_features.csv"

    def tearDown(self) -> None:
        self.storage.close()
        if self.db_path.exists():
            self.db_path.unlink()
        if self.csv_path.exists():
            self.csv_path.unlink()

    def test_export_contains_no_raw_card_or_phone(self) -> None:
        record = normalize_sheet_row(
            [f"01.05.2025", REAL_CARD_A, "Casino", "https://mirror.example", "https://pay.example"] + ["", "", ""],
            sheet_slug="s1", sheet_title="Т", tab_name="Карты", tab_index=0, row_number=1,
        )
        self.storage.insert(record)
        record2 = normalize_sheet_row(
            [f"01.05.2025", REAL_PHONE, "Casino", "https://mirror.example", "https://pay.example"] + ["", "", ""],
            sheet_slug="s1", sheet_title="Т", tab_name="Телефоны", tab_index=1, row_number=1,
        )
        self.storage.insert(record2)

        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        export_masked_features_csv(features, self.csv_path)

        content = self.csv_path.read_text(encoding="utf-8-sig")
        self.assertNotIn(REAL_CARD_A, content)
        self.assertNotIn(REAL_PHONE, content)
        self.assertNotIn("entity_value", content)
        self.assertIn("entity_hash", content)

    def test_entity_hash_is_stable_and_distinct(self) -> None:
        self.storage.insert(
            normalize_sheet_row(
                ["01.05.2025", REAL_CARD_A, "Casino", "", ""] + ["", "", ""],
                sheet_slug="s1", sheet_title="Т", tab_name="Карты", tab_index=0, row_number=1,
            )
        )
        self.storage.insert(
            normalize_sheet_row(
                ["01.05.2025", REAL_CARD_B, "Casino", "", ""] + ["", "", ""],
                sheet_slug="s1", sheet_title="Т", tab_name="Карты", tab_index=0, row_number=2,
            )
        )
        raw = load_raw(str(self.db_path))
        features = build_entity_features(raw)
        hashes = features["entity_hash"].tolist()
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertTrue(all(len(h) == 64 for h in hashes))


if __name__ == "__main__":
    unittest.main()
