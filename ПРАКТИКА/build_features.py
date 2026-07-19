"""Признаки для задачи детекции счетов-посредников («мулов», см. ТЗ.md п.1.3
и ARCHITECTURE.md): не отдельная строка-транзакция, а агрегат по одной
уникальной сущности (номер карты либо номер СБП) — частота повторного
появления, разброс во времени, связность через общие площадки.

Почему признаки строятся именно так:

- Частота повторного появления (occurrences) — прямое обоснование, ради
  которого дедупликация в sheet_collector.py сделана по позиции строки, а
  не по значению card/phone (см. ARCHITECTURE.md): если бы повторы
  схлопывались при хранении, этот признак было бы неоткуда взять.
- distinct_sheets/distinct_casinos/*_domains — сущность, встречающаяся на
  многих разных площадках/шлюзах, структурно отличается от сущности,
  использованной один раз в одном месте — типичный сигнал дроп-счёта,
  который "прогоняют" через несколько точек.
- max_shared_casino_degree — связность через общую площадку: если одна и
  та же площадка обслуживает много разных card/phone одновременно, каждая
  из них получает более высокий признак связности, чем сущность на
  малоиспользуемой площадке.

152-ФЗ: сам номер карты/телефона используется только для группировки
внутри этого процесса и не должен покидать локальную среду. В
build_entity_features() он ещё присутствует как entity_value (для
локального сохранения и обучения модели), но export_masked_features_csv()
всегда заменяет его на entity_hash (SHA-256) — признаки строятся по
группировке, а не по значению самого номера, поэтому для обучения модели
хэша достаточно, а восстановить исходный номер из него практически
невозможно.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from analyze import _OPERATOR_PREFIXES

logger = logging.getLogger("build_features")

_FEATURE_QUERY = """
    SELECT
        value_kind, card, card_bin, payment_system, phone,
        casino_name, mirror_url, payment_channel_url,
        sheet_slug, date_iso
    FROM sheet_drops
    WHERE value_kind IN ('card', 'phone')
"""


def _split_db_paths(db_path: str) -> list[str]:
    """"a.sqlite,b.sqlite" -> ["a.sqlite", "b.sqlite"] — тот же формат, что и
    в visualize.load_dataframe/analyze.load_records_from_db, для единой
    аналитики по нескольким источникам без создания их копии."""
    return [p.strip() for p in db_path.split(",") if p.strip()]


def load_raw(db_path: str) -> pd.DataFrame:
    frames = []
    for path in _split_db_paths(db_path):
        conn = sqlite3.connect(path)
        try:
            frames.append(pd.read_sql_query(_FEATURE_QUERY, conn))
        finally:
            conn.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "value_kind", "card", "card_bin", "payment_system", "phone",
        "casino_name", "mirror_url", "payment_channel_url", "sheet_slug", "date_iso",
    ])


def _extract_domain(url: str | float | None) -> str | None:
    """Домен из URL-подобной строки — только для подсчёта distinct-доменов
    (сами домены не попадают в итоговый файл, только их количество).
    Источник не всегда пишет схему аккуратно (см.
    sheet_collector.sanitize_url_field), поэтому парсинг нестрогий.

    pd.isna(), а не только `not url`: колонки, прочитанные pandas из SQL,
    приходят как float('nan') для отсутствующих значений, а не None (см.
    ту же находку в visualize.resolve_bank) — `not nan` в Python равно
    False, поэтому одной проверки на пустоту недостаточно."""
    if pd.isna(url) or not url:
        return None
    candidate = url if "://" in url else f"//{url}"
    return urlparse(candidate).netloc or None


def _phone_prefix(phone: str | float | None) -> str | None:
    if pd.isna(phone) or not phone:
        return None
    digits = "".join(c for c in phone if c.isdigit())
    return digits[1:4] if len(digits) == 11 else None


def _entity_hash(value_kind: str, value: str) -> str:
    return hashlib.sha256(f"{value_kind}:{value}".encode("utf-8")).hexdigest()


def build_entity_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Возвращает по одной строке на уникальную сущность (card или phone).
    Результат ещё содержит entity_value (полный номер) — это осознанно, для
    локального обучения модели; при экспорте наружу используйте
    export_masked_features_csv()."""
    df = raw.copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "value_kind", "entity_value", "occurrences", "distinct_sheets",
            "distinct_casinos", "distinct_mirror_domains", "distinct_payment_domains",
            "first_seen", "last_seen", "active_span_days", "card_bin", "payment_system",
            "phone_prefix", "operator", "max_shared_casino_degree", "entity_hash",
        ])

    df["entity_value"] = df["card"].where(df["value_kind"] == "card", df["phone"])
    df = df.dropna(subset=["entity_value"])
    df["mirror_domain"] = df["mirror_url"].apply(_extract_domain)
    df["payment_domain"] = df["payment_channel_url"].apply(_extract_domain)
    df["phone_prefix"] = df["phone"].apply(_phone_prefix)

    features = df.groupby(["value_kind", "entity_value"]).agg(
        occurrences=("entity_value", "size"),
        distinct_sheets=("sheet_slug", "nunique"),
        distinct_casinos=("casino_name", "nunique"),
        distinct_mirror_domains=("mirror_domain", "nunique"),
        distinct_payment_domains=("payment_domain", "nunique"),
        first_seen=("date_iso", "min"),
        last_seen=("date_iso", "max"),
        card_bin=("card_bin", "first"),
        payment_system=("payment_system", "first"),
        phone_prefix=("phone_prefix", "first"),
    ).reset_index()

    features["active_span_days"] = (
        (pd.to_datetime(features["last_seen"]) - pd.to_datetime(features["first_seen"])).dt.days.fillna(0).astype(int)
    )
    features["operator"] = features["phone_prefix"].map(_OPERATOR_PREFIXES).fillna(
        features["value_kind"].map({"phone": "Прочие/неизвестно"})
    )

    # Связность через общие площадки: для каждой сущности — самая "многолюдная"
    # площадка, на которой она встречается, минус сама сущность (см. докстринг
    # модуля). Считается на уникальных парах (сущность, площадка), чтобы
    # многократное появление на одной и той же площадке не завышало счёт.
    casino_pairs = (
        df.dropna(subset=["casino_name"])[["value_kind", "entity_value", "casino_name"]].drop_duplicates()
    )
    casino_pairs["co_entities"] = (
        casino_pairs.groupby("casino_name")["entity_value"].transform("nunique") - 1
    )
    max_co = (
        casino_pairs.groupby(["value_kind", "entity_value"])["co_entities"].max().rename("max_shared_casino_degree")
    )
    features = features.merge(max_co, on=["value_kind", "entity_value"], how="left")
    features["max_shared_casino_degree"] = features["max_shared_casino_degree"].fillna(0).astype(int)

    features["entity_hash"] = [
        _entity_hash(vk, ev) for vk, ev in zip(features["value_kind"], features["entity_value"])
    ]
    return features


def export_masked_features_csv(features: pd.DataFrame, out_path: str | Path) -> None:
    """Единственный маршрут выгрузки признаков за пределы локального
    процесса: entity_value (полный номер) не попадает в файл вообще —
    только entity_hash. Остальные признаки строятся по группировке, а не
    по значению самого номера, поэтому для обучения/анализа хэша
    достаточно."""
    safe = features.drop(columns=["entity_value", "phone_prefix"])
    cols = ["entity_hash"] + [c for c in safe.columns if c != "entity_hash"]
    safe[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Экспортировано %d сущностей (без сырых card/phone) в %s", len(safe), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="platshield_data.sqlite,platshield_gsheets_test.sqlite,platshield_archive_local.sqlite",
        help="Путь(и) к SQLite-хранилищу через запятую",
    )
    parser.add_argument("--export-csv", default="entity_features_masked.csv", help="Путь для маскированного CSV")
    args = parser.parse_args()

    raw = load_raw(args.db)
    logger.info("Загружено %d строк card/phone из %s", len(raw), args.db)
    features = build_entity_features(raw)
    logger.info(
        "Построено %d признаковых сущностей (%d карт, %d телефонов)",
        len(features), (features["value_kind"] == "card").sum(), (features["value_kind"] == "phone").sum(),
    )
    export_masked_features_csv(features, args.export_csv)


if __name__ == "__main__":
    main()
