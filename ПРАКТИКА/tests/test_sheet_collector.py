"""Тесты sheet_collector.py на фиктивном fetcher'е — без единого реального
сетевого запроса. Проверяем: обнаружение листов, постраничный обход JSON
API, нормализацию строк (card/phone/ФИО/дата), хранение с дедупом по
позиции ячейки и отсутствие сырых PAN/телефонов/ФИО в маскированном CSV."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sheet_collector import (
    SheetStorage,
    collect_all,
    compute_sheet_identity_hash,
    discover_sheet_slugs,
    fetch_sheet_meta,
    iter_tab_rows,
    mask_full_name,
    normalize_sheet_row,
    parse_date_ddmmyyyy,
    sanitize_bank_field,
    sanitize_url_field,
)

SHEETS_HTML = """
<html><body>
<div class="sheet-selector">
  <a class="sheet-selector-btn active" href="/sheets?sheet=p2p-drops-iyul-2026">p2p drops (Июль 2026)</a>
  <a class="sheet-selector-btn" href="/sheets?sheet=default">p2p drops (Март 2026)</a>
</div>
</body></html>
"""

REAL_CARD = "4111111111111111"  # Luhn-валидный тестовый номер Visa (см. test_normalize.py)
REAL_PHONE_RAW = "+7 (910) 123-45-67"
REAL_NAME = "Иванов Иван"


class FakeFetcher:
    """Возвращает заранее заданный текст по точному URL; считает вызовы."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        if url not in self.responses:
            raise KeyError(f"Незапланированный URL в тесте: {url}")
        return self.responses[url]


def _meta_json(tabs: list[dict]) -> str:
    return json.dumps({"sheet": {"slug": "s", "title": "Заголовок"}, "tabs": tabs})


def _rows_json(rows: list[list], page: int, limit: int, total: int) -> str:
    has_next = page * limit < total
    return json.dumps(
        {
            "rows": [{"number": (page - 1) * limit + i + 1, "data": r} for i, r in enumerate(rows)],
            "pagination": {"page": page, "limit": limit, "total": total, "has_next": has_next},
        }
    )


class DiscoverSheetSlugsTests(unittest.TestCase):
    def test_parses_slugs_and_titles(self) -> None:
        fetcher = FakeFetcher({"https://platshield.com/sheets/": SHEETS_HTML})
        sheets = discover_sheet_slugs(fetcher)
        self.assertEqual(len(sheets), 2)
        self.assertEqual(sheets[0]["slug"], "p2p-drops-iyul-2026")
        self.assertIn("Июль", sheets[0]["title"])
        self.assertEqual(sheets[1]["slug"], "default")


class IterTabRowsTests(unittest.TestCase):
    def test_paginates_until_has_next_false(self) -> None:
        url_p1 = "https://platshield.com/api/public-sheet.php?action=rows&sheet=s1&tab=0&page=1&limit=2"
        url_p2 = "https://platshield.com/api/public-sheet.php?action=rows&sheet=s1&tab=0&page=2&limit=2"
        fetcher = FakeFetcher(
            {
                url_p1: _rows_json([["a"], ["b"]], page=1, limit=2, total=3),
                url_p2: _rows_json([["c"]], page=2, limit=2, total=3),
            }
        )
        rows = list(iter_tab_rows(fetcher, "s1", 0, limit=2))
        self.assertEqual(len(rows), 3)
        self.assertEqual(fetcher.calls, [url_p1, url_p2])

    def test_single_page_stops_immediately(self) -> None:
        url = "https://platshield.com/api/public-sheet.php?action=rows&sheet=s1&tab=0&page=1&limit=500"
        fetcher = FakeFetcher({url: _rows_json([["only"]], page=1, limit=500, total=1)})
        rows = list(iter_tab_rows(fetcher, "s1", 0))
        self.assertEqual(len(rows), 1)
        self.assertEqual(fetcher.calls, [url])


class FetchSheetMetaTests(unittest.TestCase):
    def test_parses_meta_json(self) -> None:
        url = "https://platshield.com/api/public-sheet.php?action=meta&sheet=s1"
        fetcher = FakeFetcher({url: _meta_json([{"name": "Номера СБП", "row_count": 5}])})
        meta = fetch_sheet_meta(fetcher, "s1")
        self.assertEqual(meta["tabs"][0]["row_count"], 5)


class DateParsingTests(unittest.TestCase):
    def test_valid_ddmmyyyy(self) -> None:
        self.assertEqual(parse_date_ddmmyyyy("01.07.2026"), "2026-07-01")

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(parse_date_ddmmyyyy(""))
        self.assertIsNone(parse_date_ddmmyyyy(None))

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(parse_date_ddmmyyyy("не дата"))


class MaskFullNameTests(unittest.TestCase):
    def test_two_part_name(self) -> None:
        self.assertEqual(mask_full_name("Иванов Иван"), "И*** И.")

    def test_single_part_name(self) -> None:
        self.assertEqual(mask_full_name("Иванов"), "И***")

    def test_empty_or_none(self) -> None:
        self.assertEqual(mask_full_name(""), "")
        self.assertEqual(mask_full_name(None), "")

    def test_never_contains_full_original_name(self) -> None:
        masked = mask_full_name(REAL_NAME)
        self.assertNotIn(REAL_NAME, masked)
        self.assertNotIn("Иванов", masked)


class SanitizeBankFieldTests(unittest.TestCase):
    """Регрессия на реальную находку 2026-07-13: на боевом сборе (182 113
    строк) поле "Название банка" для значительной части строк вкладки
    "Номера СБП" фактически содержало ФИО, а не банк/оператора, и утекло
    в CSV-экспорт немаскированным до этого фикса."""

    def test_single_word_bank_passes_through(self) -> None:
        self.assertEqual(sanitize_bank_field("Сбербанк"), "Сбербанк")
        self.assertEqual(sanitize_bank_field("Билайн"), "Билайн")

    def test_single_word_first_name_redacted_via_known_names_crosscheck(self) -> None:
        # Регрессия: "Дарья"/"Людмила"/"Мохамед" — реальные однословные ФИО
        # из боевого сбора, которые правило "2+ слова" пропускает. Ловится
        # только сверкой с множеством уже известных ФИО в этом же датасете.
        known = frozenset({"Дарья", "Людмила", "Мохамед"})
        for name in known:
            result = sanitize_bank_field(name, known_full_names=known)
            self.assertNotIn(name, result)
        # Без множества known_full_names (по умолчанию) такое имя не ловится —
        # поэтому export_masked_csv обязан передавать его явно.
        self.assertEqual(sanitize_bank_field("Дарья"), "Дарья")

    def test_two_word_value_is_redacted(self) -> None:
        result = sanitize_bank_field(REAL_NAME)
        self.assertNotIn(REAL_NAME, result)
        self.assertNotIn("Иванов", result)

    def test_multiword_bank_name_also_redacted_conservatively(self) -> None:
        # Известный компромисс: легитимные многословные банки ("Банк Русский
        # Стандарт") тоже маскируются — цена за гарантированное отсутствие ФИО.
        result = sanitize_bank_field("Банк Русский Стандарт")
        self.assertNotIn("Русский Стандарт", result)

    def test_raw_phone_leaked_into_bank_field_is_redacted(self) -> None:
        # Реальный пример из локального архива (additional_data/, 2026-07-14):
        # site_bank_name = "79325030445" — однословное значение, не ловится
        # ни правилом "2+ слова", ни сверкой с known_full_names (это не ФИО).
        result = sanitize_bank_field("79325030445")
        self.assertNotIn("79325030445", result)

    def test_raw_card_leaked_into_bank_field_is_redacted(self) -> None:
        # Тот же реальный прогон: site_bank_name = "2200154507153897".
        result = sanitize_bank_field("2200154507153897")
        self.assertNotIn("2200154507153897", result)

    def test_formatted_card_digits_leaked_into_bank_field_is_redacted(self) -> None:
        # Цифры считаются даже если значение отформатировано пробелами/дефисами.
        result = sanitize_bank_field("2200 1545 0715 3897")
        self.assertNotIn("2200", result)

    def test_empty_or_none(self) -> None:
        self.assertEqual(sanitize_bank_field(""), "")
        self.assertEqual(sanitize_bank_field(None), "")


class SanitizeUrlFieldTests(unittest.TestCase):
    """Регрессия на вторую реальную находку (2026-07-13): в "Зеркало (URL)"
    и "Платёжная система (URL)" встречались (а) прямое ФИО вместо URL и
    (б) настоящие платёжные URL с ФИО/телефоном получателя прямо в
    query-параметрах."""

    def test_plain_url_truncated_to_domain(self) -> None:
        result = sanitize_url_field("https://pay.hh-processing.com/9d0b1144fe3b875a2b1bda9873a2d6634f80459f")
        self.assertEqual(result, "https://pay.hh-processing.com")

    def test_query_params_with_embedded_pii_are_stripped(self) -> None:
        # Реальный пример: ?n=<телефон>&fio=<ФИО>&bank=<банк>
        leaking_url = (
            "https://casipaysa.com/zooma/?n=79246311352&a=21212&bill=21212"
            "&bank=%D0%9E%D0%A2%D0%9F%20%D0%91%D0%B0%D0%BD%D0%BA"
            "&fio=%D0%93%D0%B8%D0%BC%D0%B3%D0%B8%D0%BD%D0%B0%20%D0%9A%D1%81%D0%B5%D0%BD%D0%B8%D1%8F"
            "&type=sbp#"
        )
        result = sanitize_url_field(leaking_url)
        self.assertEqual(result, "https://casipaysa.com")
        self.assertNotIn("79246311352", result)
        self.assertNotIn("fio=", result)

    def test_url_without_explicit_scheme_still_truncated(self) -> None:
        # Реальный пример: источник иногда не пишет "https://" вовсе.
        result = sanitize_url_field("promofast-go.com/?s=137&ref=sk_w222066c328782l17772p2169_&click_id=11cb47")
        self.assertEqual(result, "https://promofast-go.com")
        self.assertNotIn("click_id", result)

    def test_url_with_garbage_prefix_before_scheme_still_truncated(self) -> None:
        # Реальный пример: "vhttps://..." (опечатка/мусорный префикс в источнике).
        result = sanitize_url_field("vhttps://4kush-win9.com/?tracking_link=http%3A%2F%2Fpl4y-zn-ksh.com&web_id=ru0247")
        self.assertEqual(result, "https://4kush-win9.com")
        self.assertNotIn("tracking_link", result)

    def test_raw_phone_masquerading_as_url_is_redacted(self) -> None:
        # sanitize_url_field делегирует не-URL значения в sanitize_bank_field —
        # проверяем, что защита от "убежавшего" номера телефона/карты
        # действует и через этот путь тоже (тот же реальный прогон, что и
        # в SanitizeBankFieldTests).
        result = sanitize_url_field("79325030445")
        self.assertNotIn("79325030445", result)

    def test_plain_name_masquerading_as_url_is_redacted(self) -> None:
        # Реальный пример: значение колонки — не URL вообще, а прямое ФИО.
        result = sanitize_url_field("Виктория Вячеславовна Г.")
        self.assertNotIn("Виктория", result)
        self.assertNotIn("Вячеславовна", result)

    def test_single_word_name_in_url_field_caught_via_known_names(self) -> None:
        known = frozenset({"Ханапия"})
        result = sanitize_url_field("Ханапия", known_full_names=known)
        self.assertNotIn("Ханапия", result)

    def test_empty_or_none(self) -> None:
        self.assertEqual(sanitize_url_field(""), "")
        self.assertEqual(sanitize_url_field(None), "")


class NormalizeSheetRowTests(unittest.TestCase):
    def test_card_row_classified_correctly(self) -> None:
        record = normalize_sheet_row(
            ["01.07.2026", REAL_CARD, "Casino", "https://mirror.example/", "https://pay.example/inv/1",
             REAL_NAME, "Сбербанк", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера карт", tab_index=1, row_number=1,
        )
        self.assertEqual(record.value_kind, "card")
        self.assertEqual(record.card, REAL_CARD)
        self.assertEqual(record.card_bin, REAL_CARD[:6])
        self.assertIsNone(record.phone)
        self.assertEqual(record.date_iso, "2026-07-01")
        self.assertEqual(record.casino_name, "Casino")
        self.assertEqual(record.full_name, REAL_NAME)
        self.assertEqual(record.site_bank_name, "Сбербанк")

    def test_phone_row_classified_correctly(self) -> None:
        record = normalize_sheet_row(
            ["01.07.2026", REAL_PHONE_RAW, "Casino", "", "", "", "Билайн", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера СБП", tab_index=0, row_number=2,
        )
        self.assertEqual(record.value_kind, "phone")
        self.assertEqual(record.phone, "+79101234567")
        self.assertIsNone(record.card)

    def test_unrecognizable_value_is_unknown(self) -> None:
        record = normalize_sheet_row(
            ["01.07.2026", "не число вообще", "Casino", "", "", "", "", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Шопы", tab_index=2, row_number=3,
        )
        self.assertEqual(record.value_kind, "unknown")
        self.assertIsNone(record.card)
        self.assertIsNone(record.phone)

    def test_short_row_padded_without_error(self) -> None:
        record = normalize_sheet_row(
            ["01.07.2026", REAL_CARD],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера карт", tab_index=1, row_number=4,
        )
        self.assertEqual(record.value_kind, "card")
        self.assertIsNone(record.casino_name)


class SheetStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_sheet_storage.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = SheetStorage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        if self.db_path.exists():
            self.db_path.unlink()
        csv_path = self.db_path.with_suffix(".csv")
        if csv_path.exists():
            csv_path.unlink()

    def _record(self, row_number: int = 1, **overrides) -> "object":
        record = normalize_sheet_row(
            ["01.07.2026", REAL_CARD, "Casino", "https://mirror.example/", "https://pay.example/inv/1",
             REAL_NAME, "Сбербанк", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера карт", tab_index=1, row_number=row_number,
        )
        for k, v in overrides.items():
            setattr(record, k, v)
        return record

    def test_dedup_by_sheet_tab_row_position(self) -> None:
        r1 = self._record(row_number=1)
        self.assertTrue(self.storage.insert(r1))
        self.assertFalse(self.storage.insert(r1))  # тот же slug/tab/row -> дубликат
        self.assertEqual(self.storage.count(), 1)

    def test_same_card_different_row_is_not_deduped(self) -> None:
        """Один и тот же card/phone в разных строках — разные реальные
        события (разные казино/даты), не должны схлопываться."""
        r1 = self._record(row_number=1)
        r2 = self._record(row_number=2)  # тот же card, другая позиция
        self.storage.insert(r1)
        self.storage.insert(r2)
        self.assertEqual(self.storage.count(), 2)

    def test_identity_hash_depends_on_position_not_value(self) -> None:
        h1 = compute_sheet_identity_hash("s1", 1, 1)
        h2 = compute_sheet_identity_hash("s1", 1, 2)
        self.assertNotEqual(h1, h2)

    def test_export_csv_contains_no_raw_pan_phone_or_name(self) -> None:
        card_record = self._record(row_number=1)
        phone_record = normalize_sheet_row(
            ["01.07.2026", REAL_PHONE_RAW, "Casino", "", "", REAL_NAME, "Билайн", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера СБП", tab_index=0, row_number=2,
        )
        # Регрессия на реальную находку: ФИО-колонка пустая, а сама ФИО
        # оказалась в колонке "Название банка" (см. SanitizeBankFieldTests).
        leaked_name_record = normalize_sheet_row(
            ["01.07.2026", "9261112233", "Casino", "", "", "", REAL_NAME, ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера СБП", tab_index=0, row_number=3,
        )
        self.storage.insert(card_record)
        self.storage.insert(phone_record)
        self.storage.insert(leaked_name_record)

        csv_path = self.db_path.with_suffix(".csv")
        self.storage.export_masked_csv(csv_path)
        content = csv_path.read_text(encoding="utf-8-sig")

        self.assertNotIn(REAL_CARD, content)
        self.assertNotIn("+79101234567", content)
        self.assertNotIn(REAL_NAME, content)
        self.assertNotIn("Иванов", content)
        # Замаскированные представления присутствуют:
        self.assertIn("411111", content)  # БИН — не секрет
        self.assertIn("И*** И.", content)

    def test_export_redacts_single_word_name_leaked_into_bank_field(self) -> None:
        """Сквозной тест на второй слой защиты: однословное ФИО из одной
        строки не должно попасть незамаскированным в bank-поле другой
        строки, даже если оно формально проходит правило "2+ слова"."""
        first_name_owner = normalize_sheet_row(
            ["01.07.2026", REAL_CARD, "Casino", "", "", "Дарья", "Сбербанк", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера карт", tab_index=1, row_number=1,
        )
        leaked_into_bank_field = normalize_sheet_row(
            ["02.07.2026", "9261112233", "Casino", "", "", "", "Дарья", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера СБП", tab_index=0, row_number=2,
        )
        self.storage.insert(first_name_owner)
        self.storage.insert(leaked_into_bank_field)

        csv_path = self.db_path.with_suffix(".csv")
        self.storage.export_masked_csv(csv_path)
        content = csv_path.read_text(encoding="utf-8-sig")

        self.assertNotIn("Дарья", content)

    def test_export_strips_pii_from_url_fields(self) -> None:
        """Сквозной тест: query-параметры в mirror/payment URL (где может
        быть закодировано ФИО/телефон) не должны попасть в экспорт, и
        значение колонки "URL", являющееся на самом деле прямым ФИО, тоже
        должно быть замаскировано."""
        leaking_url = "https://casipaysa.com/zooma/?n=79246311352&fio=%D0%98%D0%B2%D0%B0%D0%BD%D0%BE%D0%B2"
        record = normalize_sheet_row(
            ["01.07.2026", REAL_CARD, "Casino", leaking_url, "Виктория Вячеславовна Г.", "", "Сбербанк", ""],
            sheet_slug="s1", sheet_title="Июль", tab_name="Номера карт", tab_index=1, row_number=1,
        )
        self.storage.insert(record)

        csv_path = self.db_path.with_suffix(".csv")
        self.storage.export_masked_csv(csv_path)
        content = csv_path.read_text(encoding="utf-8-sig")

        self.assertNotIn("79246311352", content)
        self.assertNotIn("fio=", content)
        self.assertNotIn("Виктория", content)
        self.assertIn("https://casipaysa.com", content)  # домен — легитимный индикатор, сохраняем


class CollectAllTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_collect_all.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_collects_across_sheets_and_tabs(self) -> None:
        meta_url = "https://platshield.com/api/public-sheet.php?action=meta&sheet=s1"
        rows_url = "https://platshield.com/api/public-sheet.php?action=rows&sheet=s1&tab=0&page=1&limit=500"

        fetcher = FakeFetcher(
            {
                meta_url: _meta_json([{"name": "Номера СБП", "row_count": 2}]),
                rows_url: _rows_json(
                    [["01.07.2026", REAL_PHONE_RAW, "C", "", "", "", "Билайн", ""],
                     ["02.07.2026", "9151234567", "C2", "", "", "", "МТС", ""]],
                    page=1, limit=500, total=2,
                ),
            }
        )
        storage = SheetStorage(self.db_path)
        try:
            stats = collect_all(fetcher, storage, sheets=[{"slug": "s1", "title": "Июль"}])
        finally:
            storage.close()

        self.assertEqual(stats["s1:Номера СБП"]["inserted"], 2)
        self.assertEqual(stats["s1:Номера СБП"]["expected"], 2)

    def test_idempotent_rerun_inserts_zero(self) -> None:
        meta_url = "https://platshield.com/api/public-sheet.php?action=meta&sheet=s1"
        rows_url = "https://platshield.com/api/public-sheet.php?action=rows&sheet=s1&tab=0&page=1&limit=500"
        responses = {
            meta_url: _meta_json([{"name": "Номера СБП", "row_count": 1}]),
            rows_url: _rows_json([["01.07.2026", REAL_PHONE_RAW, "C", "", "", "", "Билайн", ""]],
                                  page=1, limit=500, total=1),
        }
        storage = SheetStorage(self.db_path)
        try:
            collect_all(FakeFetcher(responses), storage, sheets=[{"slug": "s1", "title": "Июль"}])
            stats2 = collect_all(FakeFetcher(responses), storage, sheets=[{"slug": "s1", "title": "Июль"}])
        finally:
            storage.close()
        self.assertEqual(stats2["s1:Номера СБП"]["inserted"], 0)


if __name__ == "__main__":
    unittest.main()
