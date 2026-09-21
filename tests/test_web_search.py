"""Tests for web_search.py - no network needed (providers are mocked)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web_search as ws  # noqa: E402
from web_search import SearchError, SearchResult, web_search  # noqa: E402


import logging  # noqa: E402


def setUpModule():
    logging.disable(logging.CRITICAL)  # the code logs expected warnings; keep test output clean


def tearDownModule():
    logging.disable(logging.NOTSET)


def tavily_payload(*items):
    return {"results": [dict(title=t, url=u, content=c, **extra) for t, u, c, extra in items]}


def fake_ddgs(text=None, news=None):
    """A stand-in for ddgs.DDGS that works as a context manager."""
    instance = MagicMock()
    instance.text.return_value = text or []
    instance.news.return_value = news or []
    cls = MagicMock()
    cls.return_value.__enter__.return_value = instance
    return cls, instance


class HelperTests(unittest.TestCase):
    def test_clip_collapses_whitespace_and_cuts_on_word_boundary(self):
        self.assertEqual(ws._clip("a   b\n c"), "a b c")
        clipped = ws._clip("word " * 200, 50)
        self.assertLessEqual(len(clipped), 51)
        self.assertTrue(clipped.endswith("…"))
        self.assertEqual(ws._clip(None), "")

    def test_clean_rejects_non_http_urls(self):
        items = [SearchResult("js", "javascript:alert(1)", "x"), SearchResult("ok", "https://ok.example", "x")]
        self.assertEqual([r.title for r in ws._clean(items)], ["ok"])

    def test_clean_drops_empty_and_duplicate_results(self):
        items = [
            SearchResult("a", "http://x", "text"),
            SearchResult("dupe", "http://x", "other"),
            SearchResult("no url", "", "text"),
            SearchResult("no snippet", "http://y", ""),
        ]
        self.assertEqual([r.title for r in ws._clean(items)], ["a"])


class ProviderTests(unittest.TestCase):
    def test_tavily_results_are_normalized(self):
        payload = tavily_payload(("Vijay sworn in", "http://t/1", "CM since 10 May 2026", {"published_date": "May 2026"}))
        with patch("tavily.TavilyClient") as client_cls:
            client_cls.return_value.search.return_value = payload
            resp = web_search("tn cm", tavily_api_key="tvly-x")
        self.assertEqual(resp.provider, "Tavily")
        self.assertEqual(resp.results[0], SearchResult("Vijay sworn in", "http://t/1", "CM since 10 May 2026", "May 2026"))
        kwargs = client_cls.return_value.search.call_args.kwargs
        self.assertEqual(kwargs["topic"], "general")
        client_cls.assert_called_once_with(api_key="tvly-x")

    def test_duckduckgo_text_results_are_normalized(self):
        cls, inst = fake_ddgs(text=[{"title": "T", "href": "http://d/1", "body": "B"}])
        with patch("ddgs.DDGS", cls):
            resp = web_search("q")
        self.assertEqual(resp.provider, "DuckDuckGo")
        self.assertEqual(resp.results, [SearchResult("T", "http://d/1", "B")])
        inst.text.assert_called_once()

    def test_duckduckgo_news_results_use_url_and_date_fields(self):
        cls, inst = fake_ddgs(news=[{"title": "N", "url": "http://n/1", "body": "B", "date": "2026-09-20"}])
        with patch("ddgs.DDGS", cls):
            resp = web_search("q", news=True)
        self.assertEqual(resp.results[0].url, "http://n/1")
        self.assertEqual(resp.results[0].published, "2026-09-20")
        inst.news.assert_called_once()


class FallbackTests(unittest.TestCase):
    def test_without_a_tavily_key_tavily_is_never_used(self):
        cls, _ = fake_ddgs(text=[{"title": "T", "href": "http://d", "body": "B"}])
        with patch("tavily.TavilyClient") as tav, patch("ddgs.DDGS", cls):
            web_search("q")
        tav.assert_not_called()

    def test_tavily_failure_falls_back_to_duckduckgo(self):
        cls, _ = fake_ddgs(text=[{"title": "T", "href": "http://d", "body": "B"}])
        with patch("tavily.TavilyClient") as tav, patch("ddgs.DDGS", cls):
            tav.return_value.search.side_effect = RuntimeError("quota exceeded")
            resp = web_search("q", tavily_api_key="tvly-x")
        self.assertEqual(resp.provider, "DuckDuckGo")

    def test_tavily_empty_results_fall_back_to_duckduckgo(self):
        cls, _ = fake_ddgs(text=[{"title": "T", "href": "http://d", "body": "B"}])
        with patch("tavily.TavilyClient") as tav, patch("ddgs.DDGS", cls):
            tav.return_value.search.return_value = {"results": []}
            resp = web_search("q", tavily_api_key="tvly-x")
        self.assertEqual(resp.provider, "DuckDuckGo")

    def test_all_providers_failing_raises_search_error_naming_both(self):
        cls, inst = fake_ddgs()
        inst.text.side_effect = RuntimeError("ratelimited")
        with patch("tavily.TavilyClient") as tav, patch("ddgs.DDGS", cls):
            tav.return_value.search.side_effect = RuntimeError("bad key")
            with self.assertRaises(SearchError) as ctx:
                web_search("q", tavily_api_key="tvly-x")
        self.assertIn("Tavily", str(ctx.exception))
        self.assertIn("DuckDuckGo", str(ctx.exception))

    def test_empty_query_is_rejected(self):
        with self.assertRaises(SearchError):
            web_search("   ")


class NewsTests(unittest.TestCase):
    def test_news_flag_uses_news_topic(self):
        payload = tavily_payload(("T", "http://t", "C", {}))
        with patch("tavily.TavilyClient") as tav:
            tav.return_value.search.return_value = payload
            web_search("q", tavily_api_key="k", news=True)
        self.assertEqual(tav.return_value.search.call_args.kwargs["topic"], "news")

    def test_empty_news_search_retries_as_general_search(self):
        general = tavily_payload(("T", "http://t", "C", {}))
        with patch("tavily.TavilyClient") as tav:
            tav.return_value.search.side_effect = [{"results": []}, general]
            resp = web_search("q", tavily_api_key="k", news=True)
        topics = [c.kwargs["topic"] for c in tav.return_value.search.call_args_list]
        self.assertEqual(topics, ["news", "general"])
        self.assertEqual(len(resp.results), 1)


if __name__ == "__main__":
    unittest.main()
