"""Headless-тесты dashboard.py через streamlit.testing.v1.AppTest — без
браузера, но с реальным выполнением скрипта (в отличие от простого
HTTP-пинга сервера). Проверяем: скрипт стартует без исключений на демо,
фильтры в сайдбаре реально пересчитывают графики, а в сериализованном
Plotly-JSON (то, что реально уходит в браузер) нет ни одного полного
PAN/телефона из синтетического набора."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

from scraper import generate_demo_records


def _run_demo_app() -> AppTest:
    """Запускает dashboard.py headless и переключает сайдбар-тумблер
    "Синтетика" в True — так же, как это сделал бы пользователь в
    браузере. AppTest не пробрасывает sys.argv в исполняемый скрипт,
    поэтому CLI-флаг --demo здесь недоступен; тумблер — тот же самый
    элемент управления, что и в реальном UI, так что это не обходной
    путь мимо интерфейса, а его прямое использование."""
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "dashboard.py"))
    at.run(timeout=60)  # первый прогон: тумблер выключен -> попытка открыть SQLite по умолчанию
    at.sidebar.toggle[0].set_value(True).run(timeout=60)
    return at


class DashboardSmokeTests(unittest.TestCase):
    def test_loads_without_exceptions_on_demo_data(self) -> None:
        at = _run_demo_app()
        self.assertFalse(at.exception, msg=[str(e) for e in at.exception])

    def test_renders_seven_plotly_charts(self) -> None:
        at = _run_demo_app()
        charts = at.get("plotly_chart")
        self.assertEqual(len(charts), 7)

    def test_kpi_cards_present(self) -> None:
        at = _run_demo_app()
        combined = "".join(m.value for m in at.markdown)
        for label in ["Всего записей", "Уникальных БИН", "Банков", "Операторов СБП", "валидной по Луну"]:
            self.assertIn(label, combined)


class DashboardFilterReactivityTests(unittest.TestCase):
    def test_payment_system_filter_reduces_record_count(self) -> None:
        at = _run_demo_app()
        self.assertFalse(at.exception)
        caption_before = at.caption[-1].value

        at.sidebar.multiselect[0].set_value(["Visa"]).run(timeout=60)
        self.assertFalse(at.exception, msg=[str(e) for e in at.exception])
        caption_after = at.caption[-1].value

        self.assertNotEqual(caption_before, caption_after)

        charts_after = at.get("plotly_chart")
        self.assertEqual(len(charts_after), 7)  # фильтр не ломает набор графиков

    def test_period_radio_switches_new_per_period_chart(self) -> None:
        at = _run_demo_app()
        weekly_spec = at.get("plotly_chart")[1].proto.spec

        at.radio[0].set_value("Месяц").run(timeout=60)
        self.assertFalse(at.exception, msg=[str(e) for e in at.exception])
        monthly_spec = at.get("plotly_chart")[1].proto.spec

        self.assertNotEqual(weekly_spec, monthly_spec)


class DashboardNoRawPdnTests(unittest.TestCase):
    """Ключевая проверка 152-ФЗ-требования: то, что реально сериализуется
    в Plotly-спеки (и уходит в браузер), не должно содержать полных PAN
    или телефонов ни одной синтетической записи."""

    def test_no_full_card_or_phone_in_plotly_specs(self) -> None:
        at = _run_demo_app()
        self.assertFalse(at.exception)

        all_specs = "".join(chart.proto.spec for chart in at.get("plotly_chart"))
        all_markdown = "".join(m.value for m in at.markdown)
        haystack = all_specs + all_markdown

        records = generate_demo_records(n=400, seed=42)
        leaked_cards = [r.card for r in records if r.card and r.card in haystack]
        leaked_phones = [r.phone for r in records if r.phone and r.phone in haystack]

        self.assertEqual(leaked_cards, [], "Полный номер карты просочился в вывод дашборда")
        self.assertEqual(leaked_phones, [], "Полный телефон просочился в вывод дашборда")


if __name__ == "__main__":
    unittest.main()
