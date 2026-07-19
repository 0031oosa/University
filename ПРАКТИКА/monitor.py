"""Мониторинг новых дроп-реквизитов с алертом на почту.

Одноразовый CLI-скрипт под внешний планировщик (cron / Windows Task
Scheduler). При каждом запуске:

1. Собирает актуальный снимок записей (реюз сбора из scraper.py:
   PoliteFetcher + parse_table + normalize_record, либо --demo/--html-file
   для офлайн-прогона на синтетике).
2. Сравнивает его с уже сохранённым в SQLite по ключу идентичности —
   SHA-256 от нормализованной пары (card, phone) (scraper.compute_identity_hash).
   В базу добавляются ТОЛЬКО новые по этому ключу записи; first_seen_at
   фиксируется в момент первой вставки (см. Storage._migrate_schema).
3. Формирует обезличенный (маскированный + агрегированный) e-mail-алерт
   и отправляет его через smtplib, либо печатает («--dry-run»).

Идемпотентность: повторный запуск с тем же снимком источника не добавит
новых записей (0 новых) и, без --always-send, не отправит письмо — это
то, что нужно для запуска по расписанию без ручного контроля.

152-ФЗ: в письмо, логи и stdout попадают ТОЛЬКО маскированные
(220220******1234, +7(910)***-**-67) и агрегированные значения. Полные
PAN/телефоны никогда не покидают локальный SQLite-файл.
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from analyze import aggregate_operators, aggregate_payment_systems, aggregate_top_bins
from scraper import (
    DEFAULT_USER_AGENT,
    NormalizedRecord,
    PoliteFetcher,
    Storage,
    collect_records_from_html,
    compute_identity_hash,
    generate_demo_records,
    mask_card,
    mask_phone,
)

logger = logging.getLogger("monitor")

PREVIEW_LIMIT = 20


# 1. Сбор актуального снимка


def collect_snapshot(args: argparse.Namespace) -> list[NormalizedRecord]:
    """Возвращает текущий снимок нормализованных записей из ровно одного
    источника, заданного аргументами CLI."""
    if args.demo:
        records = generate_demo_records(n=args.n, seed=args.seed)
        logger.info("Синтетический снимок: %d записей (seed=%d)", len(records), args.seed)
        return records

    if args.html_file:
        html = Path(args.html_file).read_text(encoding="utf-8")
        records = collect_records_from_html(html)
        logger.info("Снимок из локального файла %s: %d записей", args.html_file, len(records))
        return records

    fetcher = PoliteFetcher(user_agent=args.user_agent)
    html = fetcher.fetch(args.url)
    records = collect_records_from_html(html)
    logger.info("Снимок с %s: %d записей", args.url, len(records))
    return records


# 2. Диф по ключу идентичности


def diff_new_records(batch: list[NormalizedRecord], existing_hashes: set[str]) -> list[NormalizedRecord]:
    """Возвращает записи из batch, чей identity_hash (SHA-256 от пары
    card/phone) отсутствует в existing_hashes. Дедуплицирует и внутри
    самого batch (повторы на одной странице), сохраняя порядок первого
    появления."""
    seen_in_batch: set[str] = set()
    new_records: list[NormalizedRecord] = []
    for record in batch:
        h = compute_identity_hash(record.card, record.phone)
        if h in existing_hashes or h in seen_in_batch:
            continue
        seen_in_batch.add(h)
        new_records.append(record)
    return new_records


def store_new_records(storage: Storage, new_records: list[NormalizedRecord]) -> int:
    """Вставляет уже отфильтрованные новые записи. Возвращает число
    фактически вставленных (может быть меньше len(new_records) только
    в редком случае гонки с другим процессом — INTEGRITY ERROR молча
    пропускается, как и в scraper.Storage.insert)."""
    inserted = 0
    for record in new_records:
        if storage.insert(record):
            inserted += 1
    return inserted


# 3. Тело алерта — только маскированные/агрегированные значения


def build_alert_email(new_records: list[NormalizedRecord], run_date: str) -> tuple[str, str]:
    """Формирует (тема, тело) письма. В тело попадают только счётчики,
    агрегаты по категориям и маскированное превью — см. докстринг модуля."""
    n = len(new_records)
    subject = f"[P2P-alert] Новых: {n} ({run_date})"

    lines: list[str] = []
    lines.append(f"Мониторинг дроп-реквизитов: обнаружено новых записей — {n} ({run_date}).")
    lines.append("")

    lines.append("Разбивка по платёжным системам:")
    for system, count in aggregate_payment_systems(new_records).most_common():
        lines.append(f"  {system}: {count}")
    lines.append("")

    lines.append("Топ БИН-групп (условный прокси банка-эмитента):")
    for bin_code, count in aggregate_top_bins(new_records, top_n=10):
        lines.append(f"  {bin_code}: {count}")
    lines.append("")

    lines.append("Разбивка по операторам СБП (по префиксу номера, приблизительно):")
    for operator, count in aggregate_operators(new_records).most_common():
        lines.append(f"  {operator}: {count}")
    lines.append("")

    preview = new_records[:PREVIEW_LIMIT]
    lines.append(f"Маскированное превью первых {len(preview)} новых записей:")
    for r in preview:
        card_m = mask_card(r.card) if r.card else "—"
        phone_m = mask_phone(r.phone) if r.phone else "—"
        lines.append(f"  {card_m} | {phone_m} | {r.payment_system or '—'} | {r.added_at or '—'}")
    lines.append("")

    lines.append("---")
    lines.append(
        "Автоматическое уведомление. В письме отсутствуют полные номера карт и "
        "телефонов — только маскированные и агрегированные значения (152-ФЗ)."
    )

    return subject, "\n".join(lines)


# 4. Отправка через smtplib, конфигурация из окружения


@dataclass
class SmtpConfig:
    host: str
    port: int
    username: str | None
    password: str | None
    use_tls: bool
    from_addr: str
    to_addrs: list[str]


_REQUIRED_ENV_VARS = ["SMTP_HOST", "ALERT_FROM_EMAIL", "ALERT_TO_EMAILS"]


def load_smtp_config_from_env() -> SmtpConfig:
    """Читает параметры SMTP и адреса ТОЛЬКО из переменных окружения —
    в репозитории и коде кредов быть не должно (см. докстринг модуля и
    README)."""
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise OSError(
            "Не заданы обязательные переменные окружения для SMTP: " + ", ".join(missing)
        )
    to_addrs = [addr.strip() for addr in os.environ["ALERT_TO_EMAILS"].split(",") if addr.strip()]
    if not to_addrs:
        raise OSError("ALERT_TO_EMAILS пуст после разбора списка адресов")
    return SmtpConfig(
        host=os.environ["SMTP_HOST"],
        port=int(os.environ.get("SMTP_PORT", "587")),
        username=os.environ.get("SMTP_USERNAME"),
        password=os.environ.get("SMTP_PASSWORD"),
        use_tls=os.environ.get("SMTP_USE_TLS", "1") not in ("0", "false", "False", ""),
        from_addr=os.environ["ALERT_FROM_EMAIL"],
        to_addrs=to_addrs,
    )


def send_email(subject: str, body: str, config: SmtpConfig) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.from_addr
    msg["To"] = ", ".join(config.to_addrs)
    msg.set_content(body)

    with smtplib.SMTP(config.host, config.port, timeout=30) as smtp:
        if config.use_tls:
            smtp.starttls()
        if config.username and config.password:
            smtp.login(config.username, config.password)
        smtp.send_message(msg)
    logger.info("Письмо отправлено: %s -> %s", config.from_addr, ", ".join(config.to_addrs))


# CLI


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="URL страницы с таблицей дропов (реальный сбор через scraper.PoliteFetcher)")
    source.add_argument("--demo", action="store_true", help="Синтетический снимок без сети (scraper.generate_demo_records)")
    source.add_argument("--html-file", help="Локальный HTML-файл вместо сетевого запроса (для тестов/офлайн)")

    parser.add_argument("--db", default="platshield_data.sqlite", help="Путь к SQLite-хранилищу")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--seed", type=int, default=42, help="Seed для --demo")
    parser.add_argument("--n", type=int, default=50, help="Число синтетических записей для --demo")
    parser.add_argument("--dry-run", action="store_true", help="Напечатать письмо вместо отправки")
    parser.add_argument("--always-send", action="store_true", help="Отправлять письмо даже при 0 новых записей")
    args = parser.parse_args()

    snapshot = collect_snapshot(args)

    storage = Storage(args.db)
    try:
        existing_hashes = storage.existing_identity_hashes()
        new_records = diff_new_records(snapshot, existing_hashes)
        inserted = store_new_records(storage, new_records)
    finally:
        storage.close()

    logger.info("Новых записей: %d (вставлено %d) из снимка размером %d", len(new_records), inserted, len(snapshot))

    if len(new_records) == 0 and not args.always_send:
        logger.info("Новых записей нет — письмо не отправляется (используйте --always-send для принудительной отправки)")
        return

    run_date = datetime.now().strftime("%Y-%m-%d")
    subject, body = build_alert_email(new_records, run_date)

    if args.dry_run:
        print(f"Subject: {subject}\n\n{body}")
        return

    config = load_smtp_config_from_env()
    send_email(subject, body, config)


if __name__ == "__main__":
    main()
