"""
Live web search for the chatbot.

Why this exists: an LLM only knows what it saw during training, so it happily
gives *old* answers to questions like "who is the CM of Tamil Nadu?". Searching
the web first and handing the results to the model fixes that.

Provider order
  1. Tavily      - used when TAVILY_API_KEY is set (built for LLM apps, free tier)
  2. DuckDuckGo  - needs no API key (default / automatic fallback)

Both providers are converted to the same `SearchResult` shape, so the rest of
the app never needs to know which one answered.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 600  # keep the prompt small: ~5 results x 600 chars is plenty


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published: str = ""  # only filled in when the provider supplies a date


@dataclass
class SearchResponse:
    query: str
    provider: str
    results: list[SearchResult] = field(default_factory=list)


class SearchError(RuntimeError):
    """Every configured provider failed or returned nothing."""


# ── helpers ───────────────────────────────────────────────────────────────────
def _clip(text: str | None, limit: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and cut to `limit` characters on a word boundary."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _clean(results: list[SearchResult]) -> list[SearchResult]:
    """Drop results without an http(s) URL or a snippet, and remove duplicate URLs."""
    seen: set[str] = set()
    cleaned = []
    for r in results:
        if not r.url.startswith(("http://", "https://")) or not r.snippet or r.url in seen:
            continue
        seen.add(r.url)
        cleaned.append(r)
    return cleaned


# ── providers ─────────────────────────────────────────────────────────────────
def _search_tavily(query: str, api_key: str, max_results: int, news: bool) -> list[SearchResult]:
    from tavily import TavilyClient  # imported lazily so the app still starts without it

    response = TavilyClient(api_key=api_key).search(
        query=query,
        topic="news" if news else "general",
        search_depth="basic",
        max_results=max_results,
        timeout=15,
    )
    return [
        SearchResult(
            title=_clip(item.get("title"), 160) or item.get("url", ""),
            url=item.get("url", ""),
            snippet=_clip(item.get("content")),
            published=_clip(item.get("published_date"), 40),
        )
        for item in response.get("results", [])
    ]


def _search_duckduckgo(query: str, max_results: int, news: bool) -> list[SearchResult]:
    from ddgs import DDGS  # imported lazily so the app still starts without it

    with DDGS(timeout=10) as ddgs:
        if news:
            raw = ddgs.news(query, max_results=max_results)
            return [
                SearchResult(
                    title=_clip(item.get("title"), 160),
                    url=item.get("url", ""),
                    snippet=_clip(item.get("body")),
                    published=_clip(item.get("date"), 40),
                )
                for item in raw
            ]
        raw = ddgs.text(query, max_results=max_results)
        return [
            SearchResult(
                title=_clip(item.get("title"), 160),
                url=item.get("href", ""),
                snippet=_clip(item.get("body")),
            )
            for item in raw
        ]


# ── public API ────────────────────────────────────────────────────────────────
def web_search(
    query: str,
    *,
    max_results: int = 5,
    tavily_api_key: str | None = None,
    news: bool = False,
) -> SearchResponse:
    """
    Search the web and return normalized results.

    Tries Tavily first (if a key is given), then DuckDuckGo. `news=True` asks
    the provider for recent news articles and quietly falls back to a normal
    search if the news search comes back empty.

    Raises SearchError only when *every* provider failed or found nothing.
    """
    query = (query or "").strip()
    if not query:
        raise SearchError("Empty search query.")

    providers = []
    if tavily_api_key:
        providers.append(("Tavily", lambda n: _search_tavily(query, tavily_api_key, max_results, n)))
    providers.append(("DuckDuckGo", lambda n: _search_duckduckgo(query, max_results, n)))

    problems: list[str] = []
    for name, run in providers:
        try:
            results = _clean(run(news))
            if news and not results:
                results = _clean(run(False))
            if results:
                return SearchResponse(query=query, provider=name, results=results)
            problems.append(f"{name}: no results")
        except Exception as exc:  # network error, rate limit, bad key, package missing...
            logger.warning("%s search failed: %s", name, exc)
            problems.append(f"{name}: {exc}")

    raise SearchError("; ".join(problems))
