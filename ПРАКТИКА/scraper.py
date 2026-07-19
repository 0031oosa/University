"""Сбор, нормализация, обезличенное хранение и маскирование открытых
дроп-реквизитов (карты / телефоны СБП) с публичных страниц.

ВАЖНО (152-ФЗ, этика использования):
    Полные номера карт (PAN) и телефонов хранятся ТОЛЬКО локально в SQLite
    и никогда не попадают в логи, консольный вывод или CSV-экспорт — туда
    идут исключительно маскированные значения (см. mask_card / mask_phone).
    Модуль не выполняет обход авторизации/капчи и уважает robots.txt.

Реальный сетевой сбор с конкретного ресурса должен запускаться только
после согласования постановки задачи с научным руководителем и после
того, как пользователь подгонит SELECTORS / COLUMN_ALIASES под фактическую
разметку страницы. До этого момента используйте флаг --demo — он
прогоняет весь конвейер (fetch пропускается, HTML берётся из встроенного
образца) на синтетических, заведомо валидных по Луну номерах.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import re
import sqlite3
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("scraper")

# Конфигурация разметки — подгоняется пользователем под фактический сайт.
# Ничего не захардкожено в логике парсинга: селекторы и алиасы колонок
# вынесены сюда, чтобы смена вёрстки не требовала правки кода.

SELECTORS = {
    # CSS-селектор таблицы (или контейнера строк) с записями дропов.
    "table": "table.drops-table, table",
    "row": "tbody tr",
    "header_row": "thead tr",
    # Ссылка "следующая страница" для обхода всей истории (--all-pages,
    # режим next-link). Подогнать под фактическую вёрстку пагинации сайта.
    "next_page": "a.pagination-next, a[rel='next']",
}

# Алиасы заголовков колонок (в нижнем регистре) -> каноническое имя поля.
COLUMN_ALIASES: dict[str, list[str]] = {
    "card": ["карта", "номер карты", "card", "pan"],
    "phone": ["телефон", "номер телефона", "phone", "сбп", "sbp"],
    "added_at": ["дата", "добавлено", "date", "added"],
    "source_note": ["комментарий", "note", "примечание"],
}

# Позиционный fallback, если у таблицы нет <th> (по порядку колонок).
POSITIONAL_FALLBACK_ORDER = ["card", "phone", "added_at", "source_note"]

DEFAULT_USER_AGENT = "AntifraudResearchBot/1.0 (academic research project; contact: bogdanzanes@gmail.com)"
MIN_REQUEST_INTERVAL_SEC = 2.0
MAX_RETRIES = 4
BACKOFF_BASE_SEC = 2.0


# Вежливый загрузчик страниц


class RobotsDisallowedError(RuntimeError):
    """Страница запрещена к обходу правилами robots.txt."""


class PoliteFetcher:
    """HTTP-клиент, уважающий robots.txt, троттлинг и делающий backoff.

    Параметры
    ---------
    user_agent:
        Честный User-Agent с указанием назначения и контакта — требование
        п.2 ТЗ. Используется как для реальных запросов, так и при проверке
        правил robots.txt (разные боты могут иметь разные разрешения).
    min_interval:
        Минимальная пауза между последовательными запросами, секунды.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval: float = MIN_REQUEST_INTERVAL_SEC,
        max_retries: int = MAX_RETRIES,
        session: requests.Session | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request_ts = 0.0

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if origin not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(origin, "/robots.txt"))
            try:
                rp.read()
            except OSError:
                logger.warning("Не удалось загрузить robots.txt для %s — считаем запрет по умолчанию", origin)
                rp.disallow_all = True  # type: ignore[attr-defined]
            self._robots_cache[origin] = rp
        return self._robots_cache[origin]

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def fetch(self, url: str) -> str:
        """Загружает страницу с учётом robots.txt, троттлинга и backoff."""
        rp = self._robots_for(url)
        if not rp.can_fetch(self.user_agent, url):
            raise RobotsDisallowedError(f"robots.txt запрещает обход {url} для UA={self.user_agent!r}")

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            self._last_request_ts = time.monotonic()
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                resp.encoding = resp.encoding or "utf-8"
                return resp.text
            except (requests.RequestException,) as exc:
                last_exc = exc
                backoff = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                logger.warning(
                    "Попытка %d/%d для %s не удалась (%s), backoff %.1fс",
                    attempt, self.max_retries, url, exc, backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(f"Не удалось загрузить {url} после {self.max_retries} попыток") from last_exc


# Обход всей истории (пагинация/архив) — используется флагом --all-pages.
# Каждая страница по-прежнему идёт через PoliteFetcher.fetch(), то есть с
# тем же троттлингом, backoff и проверкой robots.txt, что и одиночный
# запрос; RobotsDisallowedError прерывает обход немедленно.


def collect_all_pages_next_link(fetcher: PoliteFetcher, start_url: str, max_pages: int = 500) -> Iterator[str]:
    """Обходит страницы, последовательно переходя по ссылке
    SELECTORS['next_page'], пока она не пропадёт, не встретится повторный
    URL (защита от цикла) или не будет достигнут max_pages."""
    url = start_url
    visited: set[str] = set()
    for _ in range(max_pages):
        if url in visited:
            logger.warning("Обнаружен цикл пагинации на %s — обход остановлен", url)
            return
        visited.add(url)

        try:
            html = fetcher.fetch(url)
        except RobotsDisallowedError:
            raise
        except RuntimeError as exc:
            logger.warning("Обход остановлен на %s из-за ошибки сети: %s", url, exc)
            return
        yield html

        soup = BeautifulSoup(html, "html.parser")
        next_link = soup.select_one(SELECTORS["next_page"])
        href = next_link.get("href") if next_link else None
        if not href:
            logger.info("Ссылка на следующую страницу не найдена — конец истории (%s)", url)
            return
        url = urljoin(url, href)


def collect_all_pages_query_param(
    fetcher: PoliteFetcher,
    base_url: str,
    page_param: str = "page",
    start_page: int = 1,
    max_pages: int = 500,
) -> Iterator[str]:
    """Альтернативный режим обхода: перебирает `base_url?page=N` (или
    `&page=N`, если у base_url уже есть query-строка), пока страница не
    окажется без единой строки таблицы дропов — это трактуется как конец
    истории."""
    for page_num in range(start_page, start_page + max_pages):
        sep = "&" if "?" in base_url else "?"
        url = f"{base_url}{sep}{page_param}={page_num}"
        try:
            html = fetcher.fetch(url)
        except RobotsDisallowedError:
            raise
        except RuntimeError as exc:
            logger.warning("Обход остановлен на странице %d (%s) из-за ошибки сети: %s", page_num, url, exc)
            return

        if not parse_table(html):
            logger.info("Страница %d пуста — конец истории (%s)", page_num, url)
            return
        yield html


# Парсинг HTML-таблицы


def _match_column(header_text: str) -> str | None:
    header_text = header_text.strip().lower()
    for field, aliases in COLUMN_ALIASES.items():
        if any(alias in header_text for alias in aliases):
            return field
    return None


def parse_table(html: str) -> list[dict[str, str]]:
    """Разбирает HTML-таблицу дропов в список сырых записей.

    Устойчиво к отсутствию <th>: в этом случае колонки сопоставляются
    по позиции согласно POSITIONAL_FALLBACK_ORDER.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one(SELECTORS["table"])
    if table is None:
        return []

    header_cells = table.select(SELECTORS["header_row"] + " th")
    field_order: list[str | None]
    if header_cells:
        field_order = [_match_column(cell.get_text()) for cell in header_cells]
    else:
        field_order = list(POSITIONAL_FALLBACK_ORDER)

    records: list[dict[str, str]] = []
    for row in table.select(SELECTORS["row"]):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        record: dict[str, str] = {}
        for idx, cell in enumerate(cells):
            field = field_order[idx] if idx < len(field_order) else None
            if field is None:
                continue
            record[field] = cell.get_text(strip=True)
        if record.get("card") or record.get("phone"):
            records.append(record)
    return records


# Нормализация


def luhn_valid(card_number: str) -> bool:
    """Проверяет номер карты по алгоритму Луна."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 12:
        return False
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


# Грубые диапазоны БИН для определения платёжной системы. Для точной
# атрибуции банка-эмитента предусмотрено подключение внешней БИН-базы
# через параметр bin_lookup у normalize_card().
_PAYMENT_SYSTEM_RANGES: list[tuple[str, tuple[int, int]]] = [
    ("МИР", (220000, 220599)),
    ("Visa", (400000, 499999)),
    ("Mastercard", (510000, 559999)),
    ("Mastercard", (222100, 272099)),
    ("AmEx", (340000, 349999)),
    ("AmEx", (370000, 379999)),
]


def extract_bin(card_number: str) -> str:
    """Возвращает БИН (первые 6 цифр) номера карты."""
    digits = "".join(c for c in card_number if c.isdigit())
    return digits[:6]


def detect_payment_system(card_number: str) -> str:
    """Грубо определяет платёжную систему по диапазону БИН."""
    bin_str = extract_bin(card_number)
    if len(bin_str) < 6:
        return "Неизвестно"
    bin_num = int(bin_str)
    for system, (lo, hi) in _PAYMENT_SYSTEM_RANGES:
        if lo <= bin_num <= hi:
            return system
    return "Неизвестно"


class BinLookup:
    """Заглушка-интерфейс для подключения внешней БИН-базы (банк-эмитент).

    Реальная атрибуция банка по БИН требует внешнего датасета
    (например, коммерческой или community BIN-базы), который не входит
    в состав репозитория из-за лицензионных ограничений. Метод lookup
    достаточно переопределить/подменить, передав объект с таким же
    интерфейсом в normalize_card(bin_lookup=...).
    """

    def lookup(self, bin_code: str) -> str | None:
        return None


_PHONE_RE = re.compile(r"\D")


def normalize_phone(raw_phone: str) -> str | None:
    """Приводит телефон к формату +7XXXXXXXXXX. Возвращает None, если
    строка не похожа на российский мобильный номер."""
    digits = _PHONE_RE.sub("", raw_phone)
    if len(digits) == 11 and digits[0] in "78":
        digits = "7" + digits[1:]
    elif len(digits) == 10:
        digits = "7" + digits
    else:
        return None
    return "+" + digits


@dataclass
class NormalizedRecord:
    card: str | None
    card_bin: str | None
    payment_system: str | None
    phone: str | None
    added_at: str | None
    source_note: str | None


def normalize_record(raw: dict[str, str], bin_lookup: BinLookup | None = None) -> NormalizedRecord:
    """Нормализует одну сырую запись: валидирует карту по Луну, приводит
    телефон к каноническому виду. Невалидные значения отбрасываются
    (поле становится None), запись при этом не теряется целиком —
    решение о том, хранить ли частично валидные записи, принимает вызывающий код.
    """
    card_raw = raw.get("card", "")
    card_digits = "".join(c for c in card_raw if c.isdigit())
    card = card_digits if card_digits and luhn_valid(card_digits) else None

    phone = normalize_phone(raw.get("phone", "")) if raw.get("phone") else None

    card_bin = extract_bin(card) if card else None
    payment_system = detect_payment_system(card) if card else None
    if card_bin and bin_lookup is not None:
        bank = bin_lookup.lookup(card_bin)
        if bank:
            payment_system = f"{payment_system} ({bank})"

    return NormalizedRecord(
        card=card,
        card_bin=card_bin,
        payment_system=payment_system,
        phone=phone,
        added_at=raw.get("added_at") or None,
        source_note=raw.get("source_note") or None,
    )


def compute_identity_hash(card: str | None, phone: str | None) -> str:
    """SHA-256 от нормализованной пары (card, phone) — ключ идентичности
    записи для дифф-мониторинга (см. monitor.py).

    Это служебный внутренний идентификатор для сравнения "видели ли мы уже
    эту пару", а не средство обезличивания: он детерминированно выводится
    из PAN/телефона и хранится в той же локальной таблице SQLite, что и
    сами полные реквизиты, — то есть не расширяет поверхность утечки
    (таблица и так содержит plaintext card/phone). За пределы локальной
    БД (письма, CSV, логи) это поле не выводится.
    """
    canonical = f"{card or ''}|{phone or ''}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Маскирование (единственный формат, допустимый во внешних выводах)


def mask_card(card_number: str) -> str:
    """220220 1234 5678 9012 -> 220220******9012"""
    digits = "".join(c for c in card_number if c.isdigit())
    if len(digits) < 10:
        return "*" * len(digits)
    return digits[:6] + "*" * (len(digits) - 10) + digits[-4:]


def mask_phone(phone_e164: str) -> str:
    """+79101234567 -> +7(910)***-**-67"""
    digits = "".join(c for c in phone_e164 if c.isdigit())
    if len(digits) != 11:
        return "*" * len(digits)
    code, tail = digits[1:4], digits[9:11]
    return f"+7({code})***-**-{tail}"


# Хранение

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card TEXT,
    card_bin TEXT,
    payment_system TEXT,
    phone TEXT,
    added_at TEXT,
    source_note TEXT,
    collected_at TEXT NOT NULL,
    identity_hash TEXT,
    first_seen_at TEXT,
    UNIQUE(card, phone)
);
"""


class Storage:
    """Локальное SQLite-хранилище нормализованных записей.

    Содержит полные (немаскированные) реквизиты — это осознанное решение
    п.3.3 ТЗ: без них невозможна дедупликация и внутренняя аналитика.
    За пределы этого файла (CSV-экспорт, отчёт, графики) полные значения
    не должны покидать процесс — используйте export_masked_csv().
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Миграция без потери данных: добавляет identity_hash/first_seen_at
        в базы, созданные до появления monitor.py, и обратно заполняет их
        для уже существующих строк (identity_hash — по хэшу пары card/phone,
        first_seen_at — по имеющемуся collected_at, так как более раннего
        момента наблюдения для старых строк не существует)."""
        existing_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(drops)")}
        if "identity_hash" not in existing_cols:
            self._conn.execute("ALTER TABLE drops ADD COLUMN identity_hash TEXT")
        if "first_seen_at" not in existing_cols:
            self._conn.execute("ALTER TABLE drops ADD COLUMN first_seen_at TEXT")
        self._conn.commit()

        rows_to_backfill = self._conn.execute(
            "SELECT id, card, phone, collected_at FROM drops WHERE identity_hash IS NULL OR first_seen_at IS NULL"
        ).fetchall()
        for row_id, card, phone, collected_at in rows_to_backfill:
            self._conn.execute(
                "UPDATE drops SET identity_hash = ?, first_seen_at = COALESCE(first_seen_at, ?) WHERE id = ?",
                (compute_identity_hash(card, phone), collected_at, row_id),
            )
        if rows_to_backfill:
            self._conn.commit()

        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_drops_identity_hash ON drops(identity_hash)")
        self._conn.commit()

    def insert(self, record: NormalizedRecord) -> bool:
        """Вставляет запись, дедуплицируя по (card, phone) и по
        identity_hash. Возвращает True, если запись была новой; в этом
        случае first_seen_at фиксируется как текущий момент."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO drops (card, card_bin, payment_system, phone, added_at, "
                "source_note, collected_at, identity_hash, first_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.card,
                    record.card_bin,
                    record.payment_system,
                    record.phone,
                    record.added_at,
                    record.source_note,
                    now,
                    compute_identity_hash(record.card, record.phone),
                    now,
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def existing_identity_hashes(self) -> set[str]:
        """Множество identity_hash уже сохранённых записей — используется
        monitor.py для выделения только новых записей."""
        return {row[0] for row in self._conn.execute("SELECT identity_hash FROM drops")}

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM drops").fetchone()[0]

    def export_masked_csv(self, out_path: str | Path) -> None:
        """Экспортирует записи в CSV, где карта и телефон ТОЛЬКО в
        маскированном виде — единственный разрешённый способ выгрузки
        за пределы SQLite (см. п.3.3, п.7 критериев приёмки)."""
        rows = self._conn.execute(
            "SELECT card, card_bin, payment_system, phone, added_at, source_note, collected_at FROM drops"
        ).fetchall()
        out_path = Path(out_path)
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["card_masked", "card_bin", "payment_system", "phone_masked", "added_at", "source_note", "collected_at"]
            )
            for card, card_bin, payment_system, phone, added_at, source_note, collected_at in rows:
                writer.writerow(
                    [
                        mask_card(card) if card else "",
                        card_bin or "",
                        payment_system or "",
                        mask_phone(phone) if phone else "",
                        added_at or "",
                        source_note or "",
                        collected_at,
                    ]
                )
        logger.info("Экспортировано %d записей (маскировано) в %s", len(rows), out_path)

    def close(self) -> None:
        self._conn.close()


# Демо-режим: синтетический HTML, без единого сетевого запроса


def _luhn_checksum_digit(partial_digits: list[int]) -> int:
    """Подбирает последнюю цифру так, чтобы номер прошёл проверку Луна."""
    digits = partial_digits + [0]
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return (10 - checksum % 10) % 10


def _make_demo_card(rng: random.Random, bin_prefix: str) -> str:
    body = [int(d) for d in bin_prefix] + [rng.randint(0, 9) for _ in range(9)]
    check = _luhn_checksum_digit(body)
    return "".join(map(str, body)) + str(check)


def _make_demo_phone(rng: random.Random) -> str:
    operator_prefixes = ["900", "910", "915", "925", "926", "929", "965", "999"]
    prefix = rng.choice(operator_prefixes)
    number = "".join(str(rng.randint(0, 9)) for _ in range(7))
    return f"+7{prefix}{number}"


def generate_demo_records(n: int = 300, seed: int = 42, months_span: int = 12) -> list[NormalizedRecord]:
    """Генерирует список уже нормализованных синтетических записей
    (Luhn-валидные карты, корректные телефоны, даты за последние
    months_span месяцев) — используется analyze.py --demo для построения
    графиков без единого реального ПДн."""
    rng = random.Random(seed)
    bins = ["220220", "220100", "400000", "400111", "510000", "555500", "340000"]
    records: list[NormalizedRecord] = []
    now = datetime.now(timezone.utc)
    for _ in range(n):
        card = _make_demo_card(rng, rng.choice(bins))
        phone = _make_demo_phone(rng)
        months_back = rng.randint(0, months_span - 1)
        # В текущем месяце (months_back=0) день не должен уходить в будущее
        # относительно "сегодня" — иначе запись выглядит "добавленной завтра".
        day = rng.randint(1, now.day) if months_back == 0 else rng.randint(1, 28)
        # Через divmod, а не ручной модуль/минус — при переходе через границу
        # года прежняя формула (now.year - ...) переворачивала знак и уводила
        # дату в будущее вместо прошлого (см. разбор бага в истории проекта).
        total_months = now.year * 12 + (now.month - 1) - months_back
        year, month_idx = divmod(total_months, 12)
        month_idx += 1
        added_at = f"{year}-{month_idx:02d}-{day:02d}"
        records.append(
            NormalizedRecord(
                card=card,
                card_bin=extract_bin(card),
                payment_system=detect_payment_system(card),
                phone=phone,
                added_at=added_at,
                source_note="demo",
            )
        )
    return records


def build_demo_html(n_rows: int = 25, seed: int = 42) -> str:
    """Генерирует синтетическую HTML-таблицу в разметке, ожидаемой
    parse_table(), для воспроизводимого офлайн-прогона конвейера."""
    rng = random.Random(seed)
    bins = ["220220", "400000", "510000", "340000"]
    rows_html = []
    for i in range(n_rows):
        card = _make_demo_card(rng, rng.choice(bins))
        phone = _make_demo_phone(rng)
        day = 1 + (i % 27)
        rows_html.append(
            f"<tr><td>{card}</td><td>{phone}</td><td>2026-0{1 + i % 6}-{day:02d}</td>"
            f"<td>demo</td></tr>"
        )
    return (
        "<html><body><table class='drops-table'>"
        "<thead><tr><th>Карта</th><th>Телефон</th><th>Дата</th><th>Комментарий</th></tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody></table></body></html>"
    )


# CLI


def collect_records_from_html(html: str, bin_lookup: BinLookup | None = None) -> list[NormalizedRecord]:
    """parse_table -> normalize_record для всей страницы, с отбросом
    записей, где не распознались ни карта, ни телефон. Общая точка входа
    для scraper.py (run_pipeline) и monitor.py (диф с уже сохранённым)."""
    records: list[NormalizedRecord] = []
    for raw in parse_table(html):
        record = normalize_record(raw, bin_lookup=bin_lookup)
        if record.card is None and record.phone is None:
            continue
        records.append(record)
    return records


def run_pipeline_pages(html_pages: Iterable[str], db_path: str, csv_path: str | None) -> tuple[int, int]:
    """Прогоняет parse -> normalize -> store -> export по нескольким уже
    полученным HTML-страницам в одну и ту же БД (дедупликация сквозная,
    по identity_hash — см. Storage.insert). Возвращает (число_страниц,
    число_новых_записей)."""
    storage = Storage(db_path)
    inserted = 0
    pages = 0
    try:
        for html in html_pages:
            pages += 1
            for record in collect_records_from_html(html):
                if storage.insert(record):
                    inserted += 1
        if csv_path:
            storage.export_masked_csv(csv_path)
    finally:
        storage.close()
    return pages, inserted


def run_pipeline(html: str, db_path: str, csv_path: str | None) -> int:
    """Прогоняет полный конвейер parse -> normalize -> store -> export
    для одной уже полученной HTML-страницы. Возвращает число новых записей."""
    _, inserted = run_pipeline_pages([html], db_path, csv_path)
    return inserted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="URL страницы с таблицей дропов (реальный сбор; с --all-pages — стартовая/базовая страница истории)")
    parser.add_argument("--demo", action="store_true", help="Синтетический прогон без сети")
    parser.add_argument("--db", default="drops.sqlite3", help="Путь к SQLite-хранилищу")
    parser.add_argument("--export-csv", default=None, help="Путь для маскированного CSV-экспорта")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--all-pages", action="store_true", help="Обойти всю историю (пагинация/архив), а не одну страницу")
    parser.add_argument(
        "--pagination-mode", choices=["next-link", "query-param"], default="next-link",
        help="Способ обхода: следовать по ссылке SELECTORS['next_page'] или перебирать ?page=N (см. --page-param)",
    )
    parser.add_argument("--page-param", default="page", help="Имя query-параметра страницы для --pagination-mode query-param")
    parser.add_argument("--max-pages", type=int, default=500, help="Предохранитель: максимум страниц при --all-pages")
    args = parser.parse_args()

    if args.demo:
        html = build_demo_html()
        inserted = run_pipeline(html, args.db, args.export_csv)
        logger.info("Демо-прогон завершён: добавлено %d новых записей в %s", inserted, args.db)
        return

    if not args.url:
        parser.error("Укажите --url для реального сбора или --demo для синтетического прогона")

    fetcher = PoliteFetcher(user_agent=args.user_agent)

    if args.all_pages:
        if args.pagination_mode == "next-link":
            html_pages = collect_all_pages_next_link(fetcher, args.url, max_pages=args.max_pages)
        else:
            html_pages = collect_all_pages_query_param(
                fetcher, args.url, page_param=args.page_param, max_pages=args.max_pages
            )
        pages, inserted = run_pipeline_pages(html_pages, args.db, args.export_csv)
        logger.info("Обход всей истории завершён: %d страниц, добавлено %d новых записей в %s", pages, inserted, args.db)
        return

    html = fetcher.fetch(args.url)
    inserted = run_pipeline(html, args.db, args.export_csv)
    logger.info("Сбор завершён: добавлено %d новых записей в %s", inserted, args.db)


if __name__ == "__main__":
    main()
