"""
Web search tool with a multi-provider fallback chain.

Priority order:
  1. Tavily          – if TAVILY_API_KEY is set
  2. DuckDuckGo DDGS text search
  3. DuckDuckGo DDGS news search
  4. Stub results    – guaranteed non-empty list of placeholder items

No exception ever escapes this module. The returned list is ALWAYS non-empty
when called via `search_with_fallback`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain(url: str) -> str:
    """Extract bare domain (no www.) from a URL."""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Provider 1 – Tavily
# ---------------------------------------------------------------------------

def _tavily_search(
    query: str,
    max_results: int,
    api_key: str,
) -> List[Dict[str, Any]]:
    """
    Query the Tavily Search API.

    Returns an empty list (not an exception) on any failure.
    """
    results: List[Dict[str, Any]] = []
    if not api_key:
        return results

    try:
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "domain": _domain(item.get("url", "")),
                    "published_date": item.get("published_date"),
                    "score": item.get("score", 0.0),
                    "source": "tavily",
                }
            )
        logger.info(
            "[web_search] Tavily returned %d results for '%s'",
            len(results),
            query[:60],
        )
    except ImportError:
        logger.warning("[web_search] tavily-python not installed; skipping Tavily")
    except Exception as exc:
        logger.warning("[web_search] Tavily failed for '%s': %s", query[:60], exc)

    return results


# ---------------------------------------------------------------------------
# Provider 2 – DuckDuckGo DDGS text search
# ---------------------------------------------------------------------------

def _ddgs_text_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Query DuckDuckGo via the `duckduckgo-search` DDGS text API.

    Returns an empty list (not an exception) on any failure.
    """
    results: List[Dict[str, Any]] = []
    try:
        from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        for item in raw:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                    "domain": _domain(item.get("href", "")),
                    "source": "duckduckgo_text",
                }
            )
        logger.info(
            "[web_search] DDGS text returned %d results for '%s'",
            len(results),
            query[:60],
        )
    except ImportError:
        logger.warning("[web_search] duckduckgo-search not installed; skipping DDGS text")
    except Exception as exc:
        logger.warning("[web_search] DDGS text failed for '%s': %s", query[:60], exc)

    return results


# ---------------------------------------------------------------------------
# Provider 3 – DuckDuckGo DDGS news search
# ---------------------------------------------------------------------------

def _ddgs_news_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Query DuckDuckGo via the `duckduckgo-search` DDGS news API.

    Returns an empty list (not an exception) on any failure.
    """
    results: List[Dict[str, Any]] = []
    try:
        from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            raw = list(ddgs.news(query, max_results=max_results))

        for item in raw:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("body", ""),
                    "domain": _domain(item.get("url", "")),
                    "published_date": item.get("date"),
                    "source": "duckduckgo_news",
                }
            )
        logger.info(
            "[web_search] DDGS news returned %d results for '%s'",
            len(results),
            query[:60],
        )
    except ImportError:
        logger.warning("[web_search] duckduckgo-search not installed; skipping DDGS news")
    except Exception as exc:
        logger.warning("[web_search] DDGS news failed for '%s': %s", query[:60], exc)

    return results


# ---------------------------------------------------------------------------
# Provider 4 – Stub / URL-builder fallback (never empty)
# ---------------------------------------------------------------------------

def _stub_results(query: str, topic: str = "") -> List[Dict[str, Any]]:
    """
    Build minimal stub results so callers always receive a non-empty list.

    The URLs are real search engine query pages the user (or a browser)
    could open manually. This guarantees at least 3 results.
    """
    label = topic.strip() or query.strip()
    encoded = quote_plus(query)
    stubs = [
        {
            "title": f"Search results for: {label}",
            "url": f"https://www.google.com/search?q={encoded}",
            "snippet": (
                f"Automated search providers are unavailable. "
                f"Open this URL to search Google for '{label}'."
            ),
            "domain": "google.com",
            "source": "stub_fallback",
        },
        {
            "title": f"DuckDuckGo: {label}",
            "url": f"https://duckduckgo.com/?q={encoded}",
            "snippet": (
                f"Open this URL to search DuckDuckGo for '{label}'."
            ),
            "domain": "duckduckgo.com",
            "source": "stub_fallback",
        },
        {
            "title": f"Bing News: {label}",
            "url": f"https://www.bing.com/news/search?q={encoded}",
            "snippet": (
                f"Open this URL to search Bing News for '{label}'."
            ),
            "domain": "bing.com",
            "source": "stub_fallback",
        },
    ]
    logger.warning(
        "[web_search] All providers failed for '%s'; returning %d stub results",
        query[:60],
        len(stubs),
    )
    return stubs


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_search_queries(topic: str) -> List[str]:
    """
    Generate a rich set of targeted search queries for competitive intelligence research.

    Covers: pricing, AI capabilities, product launches, funding/M&A,
    analyst coverage, customer trends, and competitive comparisons.

    Args:
        topic: Company name, product, or market topic.

    Returns:
        List of query strings ordered from most to least specific.
    """
    base = topic.strip()
    return [
        # Core competitive landscape
        f"{base} competitive analysis market share 2025",
        f"{base} top competitors comparison 2024 2025",
        # Pricing intelligence
        f"{base} pricing plans cost per user 2025",
        f"{base} pricing changes discount enterprise tier",
        # Product & AI capabilities
        f"{base} AI features product launch announcement 2025",
        f"{base} new product release roadmap generative AI copilot",
        # M&A and funding
        f"{base} acquisition merger funding round investment 2025",
        f"{base} partnership integration strategic deal 2024 2025",
        # Market analysis
        f"{base} market size growth rate industry report 2025",
        f"{base} market trends analyst report Gartner Forrester",
        # Customer and business signals
        f"{base} customer reviews complaints churn retention",
        f"{base} revenue growth ARR customer wins enterprise",
    ]


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def web_search(
    query: str,
    max_results: int = 10,
    tavily_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a web search using Tavily (if key available) or DuckDuckGo.

    This function preserves backward-compatibility with the original signature.
    It does NOT guarantee a non-empty list by itself — use `search_with_fallback`
    for guaranteed results.

    Args:
        query:          Search query string.
        max_results:    Maximum number of results to return.
        tavily_api_key: Optional Tavily API key. Falls back to TAVILY_API_KEY env var.

    Returns:
        List of result dicts with keys: title, url, snippet, domain, source.
    """
    key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")

    start = time.time()

    if key:
        results = _tavily_search(query, max_results, api_key=key)
        engine = "tavily"
    else:
        results = _ddgs_text_search(query, max_results)
        engine = "duckduckgo_text"

    elapsed = round((time.time() - start) * 1000, 1)
    logger.info(
        "[web_search] '%s' → %d results in %.1f ms (engine=%s)",
        query[:60],
        len(results),
        elapsed,
        engine,
    )
    return results[:max_results]


def search_with_fallback(
    topic: str,
    max_results: int = 10,
    tavily_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search across all providers with automatic fallback. NEVER returns empty list.

    Fallback chain:
      1. Tavily       (if TAVILY_API_KEY env var or tavily_api_key argument is set)
      2. DDGS text    (duckduckgo-search package)
      3. DDGS news    (duckduckgo-search package)
      4. Stub results (URL-builder, always succeeds)

    Args:
        topic:          Topic or query string.
        max_results:    Maximum results desired.
        tavily_api_key: Optional override for the Tavily API key.

    Returns:
        Non-empty list of result dicts.
    """
    key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
    results: List[Dict[str, Any]] = []

    # --- Provider 1: Tavily ---------------------------------------------------
    if key:
        results = _tavily_search(topic, max_results, api_key=key)
        if results:
            return results[:max_results]
        logger.info("[web_search] Tavily returned 0 results; trying DDGS text")

    # --- Provider 2: DDGS text -----------------------------------------------
    results = _ddgs_text_search(topic, max_results)
    if results:
        return results[:max_results]
    logger.info("[web_search] DDGS text returned 0 results; trying DDGS news")

    # --- Provider 3: DDGS news -----------------------------------------------
    results = _ddgs_news_search(topic, max_results)
    if results:
        return results[:max_results]
    logger.info("[web_search] DDGS news returned 0 results; using stub fallback")

    # --- Provider 4: Stub (guaranteed non-empty) -----------------------------
    return _stub_results(topic, topic)
