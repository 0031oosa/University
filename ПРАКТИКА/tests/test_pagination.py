"""Тесты обхода всей истории (пагинация/архив) — без единого сетевого
запроса: используется фиктивный fetcher, дублирующий интерфейс
PoliteFetcher.fetch(url) -> str."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper import (
    RobotsDisallowedError,
    Storage,
    collect_all_pages_next_link,
    collect_all_pages_query_param,
    run_pipeline_pages,
)


def _page(card: str, phone: str, next_href: str | None) -> str:
    next_html = f'<a rel="next" href="{next_href}">Далее</a>' if next_href else ""
    return (
        "<html><body><table><thead><tr><th>Карта</th><th>Телефон</th></tr></thead>"
        f"<tbody><tr><td>{card}</td><td>{phone}</td></tr></tbody></table>{next_html}</body></html>"
    )


CARD_A, PHONE_A = "4111111111111111", "+79101234567"
CARD_B, PHONE_B = "5555555555554444", "+79151234567"
CARD_C, PHONE_C = "340000000000009", "+79251234567"


class FakeFetcher:
    """Возвращает заранее заданный HTML по URL; считает число вызовов."""

    def __init__(self, pages: dict[str, str], fail_on: set[str] | None = None, disallow: set[str] | None = None) -> None:
        self.pages = pages
        self.fail_on = fail_on or set()
        self.disallow = disallow or set()
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        if url in self.disallow:
            raise RobotsDisallowedError(f"robots.txt запрещает {url}")
        if url in self.fail_on:
            raise RuntimeError(f"сеть недоступна для {url}")
        return self.pages[url]


class NextLinkPaginationTests(unittest.TestCase):
    def test_follows_next_link_until_absent(self) -> None:
        pages = {
            "https://x/p1": _page(CARD_A, PHONE_A, "https://x/p2"),
            "https://x/p2": _page(CARD_B, PHONE_B, "https://x/p3"),
            "https://x/p3": _page(CARD_C, PHONE_C, None),
        }
        fetcher = FakeFetcher(pages)
        html_list = list(collect_all_pages_next_link(fetcher, "https://x/p1"))
        self.assertEqual(len(html_list), 3)
        self.assertEqual(fetcher.calls, ["https://x/p1", "https://x/p2", "https://x/p3"])

    def test_relative_next_link_resolved(self) -> None:
        pages = {
            "https://x/p1": _page(CARD_A, PHONE_A, "/p2"),
            "https://x/p2": _page(CARD_B, PHONE_B, None),
        }
        fetcher = FakeFetcher(pages)
        html_list = list(collect_all_pages_next_link(fetcher, "https://x/p1"))
        self.assertEqual(len(html_list), 2)

    def test_cycle_guard_stops(self) -> None:
        pages = {
            "https://x/p1": _page(CARD_A, PHONE_A, "https://x/p2"),
            "https://x/p2": _page(CARD_B, PHONE_B, "https://x/p1"),  # цикл
        }
        fetcher = FakeFetcher(pages)
        html_list = list(collect_all_pages_next_link(fetcher, "https://x/p1"))
        self.assertEqual(len(html_list), 2)  # не уходит в бесконечный цикл

    def test_max_pages_cap(self) -> None:
        pages = {
            "https://x/p1": _page(CARD_A, PHONE_A, "https://x/p2"),
            "https://x/p2": _page(CARD_B, PHONE_B, "https://x/p3"),
            "https://x/p3": _page(CARD_C, PHONE_C, "https://x/p4"),
        }
        fetcher = FakeFetcher(pages)
        html_list = list(collect_all_pages_next_link(fetcher, "https://x/p1", max_pages=2))
        self.assertEqual(len(html_list), 2)

    def test_robots_disallowed_propagates_immediately(self) -> None:
        pages = {
            "https://x/p1": _page(CARD_A, PHONE_A, "https://x/p2"),
            "https://x/p2": _page(CARD_B, PHONE_B, None),
        }
        fetcher = FakeFetcher(pages, disallow={"https://x/p2"})
        gen = collect_all_pages_next_link(fetcher, "https://x/p1")
        results = []
        with self.assertRaises(RobotsDisallowedError):
            for html in gen:
                results.append(html)
        self.assertEqual(len(results), 1)  # первая страница успела попасть в выдачу

    def test_network_error_stops_gracefully_without_raising(self) -> None:
        pages = {
            "https://x/p1": _page(CARD_A, PHONE_A, "https://x/p2"),
            "https://x/p2": _page(CARD_B, PHONE_B, None),
        }
        fetcher = FakeFetcher(pages, fail_on={"https://x/p2"})
        html_list = list(collect_all_pages_next_link(fetcher, "https://x/p1"))
        self.assertEqual(len(html_list), 1)  # то, что успели собрать, не теряется


class QueryParamPaginationTests(unittest.TestCase):
    def test_stops_on_empty_page(self) -> None:
        pages = {
            "https://x/list?page=1": _page(CARD_A, PHONE_A, None),
            "https://x/list?page=2": _page(CARD_B, PHONE_B, None),
            "https://x/list?page=3": "<html><body><table><tbody></tbody></table></body></html>",
        }
        fetcher = FakeFetcher(pages)
        html_list = list(collect_all_pages_query_param(fetcher, "https://x/list", page_param="page"))
        self.assertEqual(len(html_list), 2)

    def test_appends_param_correctly_when_query_exists(self) -> None:
        pages = {
            "https://x/list?lang=ru&page=1": _page(CARD_A, PHONE_A, None),
            "https://x/list?lang=ru&page=2": "<html><body><table><tbody></tbody></table></body></html>",
        }
        fetcher = FakeFetcher(pages)
        html_list = list(collect_all_pages_query_param(fetcher, "https://x/list?lang=ru", page_param="page"))
        self.assertEqual(len(html_list), 1)
        self.assertEqual(fetcher.calls, ["https://x/list?lang=ru&page=1", "https://x/list?lang=ru&page=2"])


class RunPipelinePagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).parent / "_tmp_test_pagination_pipeline.sqlite3"
        if self.db_path.exists():
            self.db_path.unlink()

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_aggregates_across_pages_with_cross_page_dedup(self) -> None:
        page1 = _page(CARD_A, PHONE_A, None)
        page2 = _page(CARD_B, PHONE_B, None)
        page3_repeat = _page(CARD_A, PHONE_A, None)  # повтор записи со стр.1

        pages, inserted = run_pipeline_pages([page1, page2, page3_repeat], self.db_path, None)
        self.assertEqual(pages, 3)
        self.assertEqual(inserted, 2)  # повтор не должен задвоиться

        storage = Storage(self.db_path)
        try:
            self.assertEqual(storage.count(), 2)
        finally:
            storage.close()


if __name__ == "__main__":
    unittest.main()
