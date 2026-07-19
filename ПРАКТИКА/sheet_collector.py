"""Сбор реальных строк "листов" P2P-дропов с platshield.com через
публичный JSON API сайта (`/api/public-sheet.php`), а не парсингом
статического HTML.

Почему не scraper.py/parse_table(): раздел `/sheets/` не содержит ни
одного `<table>` в статическом HTML — таблица рендерится на клиенте
через `/scripts/spreadsheet-view.js`, который сам делает
`fetch('/api/public-sheet.php?action=rows&...')`. Мы делаем тот же самый
запрос, которым пользуется обычный браузер при открытии публичной
страницы — без авторизации, капчи и обхода технических ограничений.

robots.txt: `PoliteFetcher` (переиспользуется из scraper.py) проверяет
каждый URL через `urllib.robotparser` перед запросом — как страницу
`/sheets/`, так и `/api/public-sheet.php`, и уважает результат.

Схема данных значительно шире учебной demo-модели scraper.py: помимо
карты/телефона источник отдаёт ФИО, название казино, зеркало (URL),
платёжный канал (URL), название банка. Поэтому хранится в ОТДЕЛЬНОЙ
таблице `sheet_drops` (см. SheetStorage) — не смешивается со схемой
`drops` из scraper.py.

152-ФЗ: ФИО и полные номера карт/телефонов остаются только в локальном
SQLite. Во всё, что покидает модуль (export_masked_csv, логи), карта и
телефон идут в стандартной маске, ФИО — в виде инициалов (mask_full_name).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode, urlparse

import sqlite3
from bs4 import BeautifulSoup

from scraper import (
    DEFAULT_USER_AGENT,
    PoliteFetcher,
    detect_payment_system,
    extract_bin,
    luhn_valid,
    mask_card,
    mask_phone,
    normalize_phone,
)

logger = logging.getLogger("sheet_collector")

API_BASE = "https://platshield.com/api/public-sheet.php"
SHEETS_PAGE = "https://platshield.com/sheets/"
# Сервер режет page size сильнее запрошенного (эмпирически: limit=1000
# в ответе стал limit=500) — используем подтверждённый максимум, чтобы
# минимизировать число запросов при сохранении вежливого троттлинга.
ROW_PAGE_LIMIT = 500


# Обнаружение листов и постраничный сбор строк через JSON API


def discover_sheet_slugs(fetcher: PoliteFetcher) -> list[dict]:
    """Возвращает [{'slug', 'title'}] по кнопкам выбора листа на /sheets/."""
    html = fetcher.fetch(SHEETS_PAGE)
    soup = BeautifulSoup(html, "html.parser")
    sheets = []
    for a in soup.select("a.sheet-selector-btn"):
        href = a.get("href", "")
        slug = href.split("sheet=")[-1] if "sheet=" in href else None
        if slug:
            sheets.append({"slug": slug, "title": a.get_text(strip=True)})
    return sheets


def _api_url(params: dict) -> str:
    return f"{API_BASE}?{urlencode(params)}"


def fetch_sheet_meta(fetcher: PoliteFetcher, slug: str) -> dict:
    text = fetcher.fetch(_api_url({"action": "meta", "sheet": slug}))
    return json.loads(text)


def iter_tab_rows(fetcher: PoliteFetcher, slug: str, tab_index: int, limit: int = ROW_PAGE_LIMIT) -> Iterator[dict]:
    """Постранично отдаёт сырые строки (payload['rows'][i], с полем
    'data') одной вкладки листа, пока pagination.has_next не станет false."""
    page = 1
    while True:
        url = _api_url({"action": "rows", "sheet": slug, "tab": tab_index, "page": page, "limit": limit})
        text = fetcher.fetch(url)
        payload = json.loads(text)
        for row in payload.get("rows", []):
            yield row
        if not payload.get("pagination", {}).get("has_next"):
            return
        page += 1


# Нормализация одной строки листа


def parse_date_ddmmyyyy(raw: str | None) -> str | None:
    """'01.07.2026' -> '2026-07-01'. Источник использует dd.mm.yyyy, в
    отличие от внутреннего ISO-формата проекта. None/невалидная строка -> None."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def mask_full_name(name: str | None) -> str:
    """"Иванов Иван" -> "И*** И.". Пусто/None -> ""."""
    if not name:
        return ""
    parts = [p for p in name.split() if p]
    if not parts:
        return ""
    masked = parts[0][0] + "***"
    if len(parts) > 1:
        masked += " " + parts[1][0] + "."
    return masked


def sanitize_bank_field(value: str | None, known_full_names: frozenset[str] = frozenset()) -> str:
    """Название банка/оператора из источника оказалось ненадёжным полем:
    эмпирическая проверка на реальном сборе (2026-07-13, 182 113 строк)
    показала, что для значительной части строк вкладки "Номера СБП" в
    этом столбце фактически лежит ФИО, а не название банка/оператора —
    видимо, из-за того, как по-разному заполняется исходная таблица для
    карточных и телефонных записей (у карточных ФИО обычно в своей
    колонке, у телефонных иногда "утекает" сюда).

    Два независимых правила защиты (проверено на реальных данных — оба
    нужны, одного недостаточно):

    1. Любое значение из двух и более слов маскируем как потенциальное
       ФИО (ценой потери части легитимных многословных названий банков
       вроде "Банк Русский Стандарт" — эта потеря приемлема, утечка
       ФИО — нет).
    2. Однословные значения дополнительно сверяются с множеством уже
       известных ФИО из этого же датасета (known_full_names) — так
       ловятся однословные имена без фамилии (например, только "Дарья"),
       которые правило (1) пропускает. Однословные названия банков вроде
       "Сбербанк"/"МТС"/"Билайн" проходят как есть, если только то же
       самое слово не встретилось где-то в колонке ФИО (тогда, по той же
       консервативной логике, тоже маскируется).
    3. Отдельно от вопроса ФИО: значение, в котором 10+ цифр (после
       удаления пробелов/дефисов/скобок), — это не название банка, а
       "убежавший" в чужую колонку номер карты или телефона (найдено на
       реальных данных: site_bank_name = "79325030445",
       site_bank_name = "2200154507153897"). Такие значения маскируются
       раньше и независимо от проверки на ФИО."""
    if not value:
        return ""
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) >= 10:
        return "(скрыто: похоже на карту/телефон, не банк)"
    words = value.split()
    if len(words) >= 2 or value in known_full_names:
        return "(скрыто: похоже на ФИО, не банк)"
    return value


def sanitize_url_field(value: str | None, known_full_names: frozenset[str] = frozenset()) -> str:
    """"Зеркало (URL)" и "Платёжная система (URL)" оказались ЕЩЁ более
    ненадёжными полями, чем "Название банка": эмпирическая проверка на
    реальном сборе нашла строки, где это поле — не URL вообще, а прямое
    ФИО (данные внесены не в ту колонку исходной таблицы), и строки, где
    это настоящий платёжный URL, но с ФИО и телефоном получателя прямо
    в query-параметрах (например, `?...&n=79261112233&fio=Иванов+Иван&...`).

    Полный URL (с query-строкой) поэтому никогда не экспортируется как
    есть. Если значение — валидный http(s)-URL, оставляем только
    scheme+домен (это и есть содержательный индикатор — "через какой
    шлюз/зеркало шёл платёж" — без query-параметров, где может быть
    спрятано что угодно). Если значение URL-ом не является вообще —
    применяем ту же защиту от ФИО, что и для банковского поля.

    Источник не всегда пишет схему аккуратно: встречаются URL без
    "https://" вовсе ("promofast-go.com/?s=...") и с мусорным префиксом
    перед схемой ("vhttps://..."). Оба случая обрабатываем терпимо, но
    только когда в значении нет пробелов (у настоящего URL их не бывает,
    у ФИО — почти всегда есть) — это не даёт эвристике по ошибке принять
    многословное ФИО за домен."""
    if not value:
        return ""
    parsed = urlparse(value)
    if not (parsed.scheme in ("http", "https") and parsed.netloc) and " " not in value:
        match = re.search(r"https?://\S+", value)
        if match:
            parsed = urlparse(match.group(0))
        elif re.search(r"\.[a-zA-Z]{2,}(/|\?|$)", value):
            parsed = urlparse("https://" + value)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return sanitize_bank_field(value, known_full_names)


@dataclass
class SheetRecord:
    sheet_slug: str
    sheet_title: str
    tab_name: str
    tab_index: int
    row_number: int
    date_raw: str | None
    date_iso: str | None
    value_kind: str  # "card" | "phone" | "unknown"
    card: str | None
    card_bin: str | None
    payment_system: str | None
    phone: str | None
    casino_name: str | None
    mirror_url: str | None
    payment_channel_url: str | None
    full_name: str | None
    site_bank_name: str | None


def normalize_sheet_row(
    raw_cells: list,
    sheet_slug: str,
    sheet_title: str,
    tab_name: str,
    tab_index: int,
    row_number: int,
) -> SheetRecord:
    """Колонки подтверждены вручную по реальному ответу API (2026-07-13):
    A=Дата, B=Номер СБП/карты, C=Название Казино, D=Зеркало (URL),
    E=Платёжная система (URL, платёжный канал/шлюз), F=ФИО,
    G=Название банка, H=(не используется источником)."""
    cells = [("" if c is None else str(c)) for c in raw_cells] + [""] * 8
    date_raw, value_raw, casino, mirror, paysys_url, fio, bank, _unused = cells[:8]

    value_digits = "".join(c for c in value_raw if c.isdigit())
    card = card_bin = payment_system = phone = None
    value_kind = "unknown"
    if value_digits and luhn_valid(value_digits):
        card = value_digits
        card_bin = extract_bin(card)
        payment_system = detect_payment_system(card)
        value_kind = "card"
    else:
        normalized_phone = normalize_phone(value_raw)
        if normalized_phone:
            phone = normalized_phone
            value_kind = "phone"

    return SheetRecord(
        sheet_slug=sheet_slug,
        sheet_title=sheet_title,
        tab_name=tab_name,
        tab_index=tab_index,
        row_number=row_number,
        date_raw=date_raw or None,
        date_iso=parse_date_ddmmyyyy(date_raw),
        value_kind=value_kind,
        card=card,
        card_bin=card_bin,
        payment_system=payment_system,
        phone=phone,
        casino_name=casino or None,
        mirror_url=mirror or None,
        payment_channel_url=paysys_url or None,
        full_name=fio or None,
        site_bank_name=bank or None,
    )


# Хранение — отдельная таблица sheet_drops

_SHEET_SCHEMA = """
CREATE TABLE IF NOT EXISTS sheet_drops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sheet_slug TEXT NOT NULL,
    sheet_title TEXT,
    tab_name TEXT,
    tab_index INTEGER,
    row_number INTEGER,
    date_raw TEXT,
    date_iso TEXT,
    value_kind TEXT,
    card TEXT,
    card_bin TEXT,
    payment_system TEXT,
    phone TEXT,
    casino_name TEXT,
    mirror_url TEXT,
    payment_channel_url TEXT,
    full_name TEXT,
    site_bank_name TEXT,
    collected_at TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    identity_hash TEXT NOT NULL,
    UNIQUE(identity_hash)
);
"""


def compute_sheet_identity_hash(sheet_slug: str, tab_index: int, row_number: int) -> str:
    """Идентичность — конкретная позиция ячейки на сайте (лист/вкладка/
    номер строки), а НЕ хэш от значения карты/телефона. Это осознанно:
    одинаковый card/phone может законно повторяться в разных строках
    (разные казино/даты) — для задачи детекции мулов повторное появление
    одного номера в разных местах само по себе значимый признак, и
    схлопывать такие строки в дедупликации means терять сигнал. Дедуп
    здесь защищает только от повторной вставки той же самой ячейки при
    повторном запуске сборщика (идемпотентность), не от "повторов" данных."""
    canonical = f"{sheet_slug}|{tab_index}|{row_number}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SheetStorage:
    """Локальное SQLite-хранилище строк реальных листов. Содержит полные
    (немаскированные) card/phone/full_name — как и scraper.Storage, это
    осознанное решение для внутренней аналитики; наружу (export_masked_csv)
    выходят только маскированные значения."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(_SHEET_SCHEMA)
        self._conn.commit()

    def insert(self, record: SheetRecord) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        identity_hash = compute_sheet_identity_hash(record.sheet_slug, record.tab_index, record.row_number)
        try:
            self._conn.execute(
                "INSERT INTO sheet_drops (sheet_slug, sheet_title, tab_name, tab_index, row_number, "
                "date_raw, date_iso, value_kind, card, card_bin, payment_system, phone, casino_name, "
                "mirror_url, payment_channel_url, full_name, site_bank_name, collected_at, first_seen_at, "
                "identity_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.sheet_slug, record.sheet_title, record.tab_name, record.tab_index, record.row_number,
                    record.date_raw, record.date_iso, record.value_kind, record.card, record.card_bin,
                    record.payment_system, record.phone, record.casino_name, record.mirror_url,
                    record.payment_channel_url, record.full_name, record.site_bank_name, now, now, identity_hash,
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM sheet_drops").fetchone()[0]

    def export_masked_csv(self, out_path: str | Path) -> None:
        """Экспорт в CSV — card/phone в стандартной маске, ФИО в виде
        инициалов. Единственный разрешённый способ выгрузки за пределы
        SQLite (см. докстринг модуля)."""
        known_full_names = frozenset(
            r[0] for r in self._conn.execute(
                "SELECT DISTINCT full_name FROM sheet_drops WHERE full_name IS NOT NULL AND full_name != ''"
            )
        )
        rows = self._conn.execute(
            "SELECT sheet_slug, tab_name, date_iso, value_kind, card, card_bin, payment_system, phone, "
            "casino_name, mirror_url, payment_channel_url, full_name, site_bank_name, collected_at "
            "FROM sheet_drops"
        ).fetchall()
        out_path = Path(out_path)
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sheet_slug", "tab_name", "date", "value_kind", "card_masked", "card_bin", "payment_system",
                    "phone_masked", "casino_name", "mirror_url", "payment_channel_url", "full_name_masked",
                    "site_bank_name", "collected_at",
                ]
            )
            for (
                sheet_slug, tab_name, date_iso, value_kind, card, card_bin, payment_system, phone,
                casino_name, mirror_url, payment_channel_url, full_name, site_bank_name, collected_at,
            ) in rows:
                writer.writerow(
                    [
                        sheet_slug, tab_name, date_iso or "", value_kind,
                        mask_card(card) if card else "",
                        card_bin or "",
                        payment_system or "",
                        mask_phone(phone) if phone else "",
                        sanitize_url_field(casino_name, known_full_names),
                        sanitize_url_field(mirror_url, known_full_names),
                        sanitize_url_field(payment_channel_url, known_full_names),
                        mask_full_name(full_name),
                        sanitize_url_field(site_bank_name, known_full_names),
                        collected_at,
                    ]
                )
        logger.info(
            "Экспортировано %d строк (карты/телефоны/ФИО маскированы, URL обрезаны до домена) в %s",
            len(rows), out_path,
        )

    def close(self) -> None:
        self._conn.close()


# Обход всех листов/вкладок


def collect_all(fetcher: PoliteFetcher, storage: SheetStorage, sheets: list[dict] | None = None) -> dict:
    """Обходит переданные (или обнаруженные автоматически) листы и все их
    вкладки, вставляя новые строки. Возвращает статистику по каждому
    листу/вкладке: {"<slug>:<tab_name>": {"expected", "seen", "inserted"}}."""
    if sheets is None:
        sheets = discover_sheet_slugs(fetcher)

    stats: dict[str, dict] = {}
    for sheet in sheets:
        slug = sheet["slug"]
        meta = fetch_sheet_meta(fetcher, slug)
        sheet_title = meta.get("sheet", {}).get("title", sheet.get("title", slug))
        tabs = meta.get("tabs", [])
        for tab_index, tab in enumerate(tabs):
            tab_name = tab.get("name", f"tab{tab_index}")
            row_count_expected = tab.get("row_count", 0)
            inserted = 0
            seen = 0
            for row in iter_tab_rows(fetcher, slug, tab_index):
                seen += 1
                row_number = row.get("number", seen)
                record = normalize_sheet_row(
                    row.get("data", []), slug, sheet_title, tab_name, tab_index, row_number
                )
                if storage.insert(record):
                    inserted += 1
            key = f"{slug}:{tab_name}"
            stats[key] = {"expected": row_count_expected, "seen": seen, "inserted": inserted}
            logger.info(
                "Лист %s / вкладка %s: ожидалось %d, получено %d, новых вставлено %d",
                slug, tab_name, row_count_expected, seen, inserted,
            )
    return stats


# CLI


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="platshield_data.sqlite", help="Путь к SQLite-хранилищу")
    parser.add_argument("--export-csv", default=None, help="Путь для маскированного CSV-экспорта после сбора")
    parser.add_argument(
        "--sheet", action="append",
        help="Ограничить сбор конкретным slug листа (флаг можно повторять); по умолчанию — все обнаруженные",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--meta-only", action="store_true",
        help="Только показать список листов и число строк по вкладкам, без реального сбора",
    )
    args = parser.parse_args()

    fetcher = PoliteFetcher(user_agent=args.user_agent)

    sheets = discover_sheet_slugs(fetcher)
    logger.info("Обнаружено листов: %d", len(sheets))
    if args.sheet:
        sheets = [s for s in sheets if s["slug"] in args.sheet]

    if args.meta_only:
        for s in sheets:
            meta = fetch_sheet_meta(fetcher, s["slug"])
            for tab in meta.get("tabs", []):
                logger.info("  %s / %s: %d строк", s["slug"], tab.get("name"), tab.get("row_count", 0))
        return

    storage = SheetStorage(args.db)
    try:
        stats = collect_all(fetcher, storage, sheets)
        if args.export_csv:
            storage.export_masked_csv(args.export_csv)
    finally:
        storage.close()

    total_inserted = sum(s["inserted"] for s in stats.values())
    total_seen = sum(s["seen"] for s in stats.values())
    logger.info("Сбор завершён. Получено строк: %d, новых вставлено: %d, в БД: %s", total_seen, total_inserted, args.db)


if __name__ == "__main__":
    main()
