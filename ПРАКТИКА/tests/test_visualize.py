"""Тесты visualize.py: агрегаты на синтетике, отсутствие полных PAN/
телефонов на любом этапе конвейера (DataFrame -> PNG), корректность
export_static()."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import visualize as v
from scraper import generate_demo_records
from sheet_collector import SheetStorage, normalize_sheet_row

REAL_CARD = "4111111111111111"  # Luhn-валидный тестовый номер


class LoadDataframeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = v.load_dataframe(demo=True, demo_n=150, demo_seed=1, demo_months_span=12)

    def test_no_raw_card_or_phone_columns(self) -> None:
        self.assertNotIn("card", self.df.columns)
        self.assertNotIn("phone", self.df.columns)

    def test_no_raw_pan_or_phone_values_anywhere_in_dataframe(self) -> None:
        records = generate_demo_records(n=150, seed=1, months_span=12)
        as_text = self.df.astype(str).apply(lambda col: col.str.cat(sep=""), axis=0).str.cat(sep="")
        for r in records:
            if r.card:
                self.assertNotIn(r.card, as_text)
            if r.phone:
                self.assertNotIn(r.phone, as_text)

    def test_expected_derived_columns_present(self) -> None:
        for col in ["card_bin", "payment_system", "bank", "operator", "event_date", "first_seen_date", "has_card"]:
            self.assertIn(col, self.df.columns)

    def test_phone_prefix_is_at_most_3_digits(self) -> None:
        # phone_prefix — только 3-значный префикс оператора, не номер целиком.
        lengths = self.df["phone_prefix"].dropna().str.len().unique()
        self.assertTrue(all(length <= 3 for length in lengths))


class LoadDataframeRealDbSanityTests(unittest.TestCase):
    """Регрессия на реальную находку (2026-07-14, локальный архив
    additional_data/): пара строк там несёт опечатку в дате источника
    ("26.05.2038", "05.06.2040" внутри вкладок за 2025 год) — без защиты
    такие даты растягивали ось времени графиков на 15 лет вперёд."""

    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_visualize_real.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()
        self.storage = SheetStorage(self.db_path)

    def tearDown(self) -> None:
        self.storage.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_event_date_after_first_seen_is_nulled_as_implausible(self) -> None:
        # Дата источника (2040) заведомо позже момента сбора (first_seen_at
        # проставляется автоматически "сейчас" при insert) — это не может
        # быть правдой, значит опечатка в исходнике, а не факт.
        record = normalize_sheet_row(
            ["05.06.2040", REAL_CARD, "Casino", "", "", "", "", ""],
            sheet_slug="s1", sheet_title="Т", tab_name="Карты", tab_index=0, row_number=1,
        )
        self.storage.insert(record)
        df = v.load_dataframe(db_path=str(self.db_path))
        self.assertTrue(df["event_date"].isna().all())
        # Запись при этом никуда не пропадает — просто не участвует в
        # графиках по времени.
        self.assertEqual(len(df), 1)

    def test_plausible_past_date_is_kept(self) -> None:
        record = normalize_sheet_row(
            ["01.07.2026", REAL_CARD, "Casino", "", "", "", "", ""],
            sheet_slug="s1", sheet_title="Т", tab_name="Карты", tab_index=0, row_number=1,
        )
        self.storage.insert(record)
        df = v.load_dataframe(db_path=str(self.db_path))
        self.assertFalse(df["event_date"].isna().any())


class KpiTests(unittest.TestCase):
    def test_kpis_on_known_synthetic_set(self) -> None:
        df = v.load_dataframe(demo=True, demo_n=100, demo_seed=7, demo_months_span=6)
        kpis = v.compute_kpis(df)
        self.assertEqual(kpis["total_records"], 100)
        self.assertGreater(kpis["unique_bins"], 0)
        self.assertGreaterEqual(kpis["luhn_valid_share"], 0.0)
        self.assertLessEqual(kpis["luhn_valid_share"], 1.0)
        self.assertIsNotNone(kpis["last_update"])

    def test_kpis_on_empty_dataframe(self) -> None:
        df = v.load_dataframe(demo=True, demo_n=0, demo_seed=1)
        kpis = v.compute_kpis(df)
        self.assertEqual(kpis["total_records"], 0)
        self.assertEqual(kpis["luhn_valid_share"], 0.0)
        self.assertIsNone(kpis["last_update"])


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = v.load_dataframe(demo=True, demo_n=250, demo_seed=3, demo_months_span=18)

    def test_cumulative_growth_is_monotonic_and_ends_at_total(self) -> None:
        growth = v.cumulative_growth(self.df)
        self.assertTrue((growth["cumulative"].diff().dropna() >= 0).all())
        self.assertEqual(growth["cumulative"].iloc[-1], len(self.df.dropna(subset=["event_date"])))

    def test_new_per_period_weekly_sums_to_total(self) -> None:
        weekly = v.new_per_period(self.df, freq="W")
        self.assertEqual(weekly["count"].sum(), len(self.df.dropna(subset=["event_date"])))

    def test_new_per_period_omits_zero_count_gaps(self) -> None:
        # Реальный случай (2026-07-14): сбор шёл отдельными "островами"
        # месяцев с многомесячными разрывами — pd.Grouper по умолчанию
        # заполнял разрывы нулями, растягивая график в пустую полосу.
        # Строим df с двумя изолированными днями данных, разделёнными
        # трёхмесячным разрывом, и проверяем, что "пустой" месяц между
        # ними не попадает в выдачу вовсе (а не попадает со count=0).
        df = pd.DataFrame(
            {
                "event_date": pd.to_datetime(["2025-01-15", "2025-01-16", "2025-04-10"]),
            }
        )
        monthly = v.new_per_period(df, freq="ME")
        self.assertEqual(len(monthly), 2)  # только январь и апрель, февраль/март отсутствуют
        self.assertTrue((monthly["count"] > 0).all())

    def test_calendar_heatmap_grid_counts_match_total(self) -> None:
        grid = v.calendar_heatmap_grid(self.df)
        self.assertEqual(grid["count"].sum(), len(self.df.dropna(subset=["event_date"])))

    def test_bank_system_matrix_sums_to_total(self) -> None:
        pivot = v.bank_system_matrix(self.df)
        self.assertEqual(int(pivot.values.sum()), len(self.df))

    def test_bank_system_matrix_caps_rows_at_top_n_plus_other(self) -> None:
        pivot = v.bank_system_matrix(self.df, top_n_banks=2)
        self.assertLessEqual(len(pivot.index), 3)  # top 2 + "Прочие"

    def test_treemap_data_sums_to_total(self) -> None:
        data = v.treemap_data(self.df)
        self.assertEqual(int(data["count"].sum()), len(self.df))

    def test_operator_share_over_time_sums_to_one_per_period(self) -> None:
        shares = v.operator_share_over_time(self.df)
        totals = shares.groupby("period")["share"].sum()
        for total in totals:
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_rolling_trend_matches_daily_counts(self) -> None:
        # На синтетике (250 случайных записей за 18 месяцев) изолированные
        # "нулевые" дни — обычная статистическая разреженность, а не
        # реальный многомесячный разрыв сбора: rolling_trend всё равно
        # вставит для них NaN-разрыв (см. test_rolling_trend_breaks_line_
        # across_multimonth_gap) — поэтому здесь проверяем только реальные
        # (не NaN) строки.
        trend = v.rolling_trend(self.df, freq="D", window=7)
        real_rows = trend[trend["count"].notna()]
        self.assertEqual(real_rows["count"].sum(), len(self.df.dropna(subset=["event_date"])))
        self.assertTrue(real_rows["rolling_mean"].notna().all())

    def test_rolling_trend_breaks_line_across_multimonth_gap(self) -> None:
        # Реальный случай (2026-07-14): без разрыва line-plot рисовал прямую
        # (ложный "тренд") через месяцы, где данных физически нет, между
        # "островами" сбора. Проверяем, что между двумя изолированными
        # днями данных вставлена NaN-строка, а не прямое соединение.
        df = pd.DataFrame({"event_date": pd.to_datetime(["2025-01-15", "2025-04-10"])})
        trend = v.rolling_trend(df, freq="D", window=7)
        self.assertEqual(len(trend), 3)  # 2 реальные точки + 1 разрыв между ними
        gap_row = trend.iloc[1]
        self.assertTrue(pd.isna(gap_row["count"]))
        self.assertTrue(pd.isna(gap_row["rolling_mean"]))
        self.assertTrue(pd.Timestamp("2025-01-15") < gap_row["date"] < pd.Timestamp("2025-04-10"))


class ResolveBankOperatorTests(unittest.TestCase):
    def test_resolve_bank_none_bin(self) -> None:
        self.assertEqual(v.resolve_bank(None, demo=False), v.BANK_UNKNOWN)

    def test_resolve_bank_nan_bin_is_unknown_not_literal_nan(self) -> None:
        # Реальная находка (2026-07-15): pd.read_sql_query отдаёт для
        # телефонных строк (нет card_bin) float('nan'), не None. Старая
        # проверка `if not card_bin` пропускала NaN (в Python `not nan`
        # равно False) — 261 893 строки схлопывались в фиктивный банк
        # "БИН nan" вместо "Неизвестно".
        self.assertEqual(v.resolve_bank(float("nan"), demo=False), v.BANK_UNKNOWN)

    def test_resolve_bank_real_mode_is_bin_literal(self) -> None:
        self.assertEqual(v.resolve_bank("400000", demo=False), "БИН 400000")

    def test_resolve_bank_demo_mode_uses_hint_table(self) -> None:
        self.assertIn("демо", v.resolve_bank("400000", demo=True))

    def test_resolve_operator_unknown_prefix(self) -> None:
        self.assertEqual(v.resolve_operator("111"), v.OPERATOR_UNKNOWN)

    def test_resolve_operator_nan_prefix_is_unknown(self) -> None:
        # Тот же класс NaN-truthiness бага, что и в resolve_bank (см. выше) —
        # для строк-карт phone_prefix приходит как float('nan').
        self.assertEqual(v.resolve_operator(float("nan")), v.OPERATOR_UNKNOWN)

    def test_resolve_operator_known_prefix(self) -> None:
        self.assertEqual(v.resolve_operator("910"), "МТС")


class ExportStaticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.out_dir = Path(__file__).parent / "_tmp_report_assets"
        for p in self.out_dir.glob("*.png") if self.out_dir.exists() else []:
            p.unlink()
        self.df = v.load_dataframe(demo=True, demo_n=200, demo_seed=42, demo_months_span=18)

    def tearDown(self) -> None:
        if self.out_dir.exists():
            for p in self.out_dir.glob("*.png"):
                p.unlink()
            self.out_dir.rmdir()

    def test_export_static_produces_six_pngs(self) -> None:
        paths = v.export_static(self.df, self.out_dir)
        self.assertEqual(len(paths), 6)
        for p in paths:
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 0)

    def test_export_static_handles_empty_dataframe_without_raising(self) -> None:
        empty_df = v.load_dataframe(demo=True, demo_n=0, demo_seed=1)
        paths = v.export_static(empty_df, self.out_dir)
        self.assertEqual(len(paths), 6)


if __name__ == "__main__":
    unittest.main()
