"""
Source extractor – validates URLs, scores relevance, deduplicates,
and converts raw search results into SourceDocument objects.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from agents.state import SourceDocument
from config import search_config

logger = logging.getLogger(__name__)


def _is_valid_url(url: str) -> bool:
    """Check whether a URL is syntactically valid and uses http/https."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _extract_domain(url: str) -> str:
    """Extract bare domain (without www.) from a URL."""
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return ""


def _score_relevance(text: str, topic: str) -> float:
    """
    Simple keyword-based relevance scoring.
    Returns a float between 0.0 and 1.0.
    """
    if not text or not topic:
        return 0.0
    topic_words = set(re.findall(r"\w+", topic.lower()))
    text_words = set(re.findall(r"\w+", text.lower()[:2000]))
    if not topic_words:
        return 0.0
    overlap = len(topic_words & text_words)
    return min(1.0, overlap / len(topic_words))


def _is_trusted(domain: str) -> bool:
    """Check whether the domain is in the trusted domain list."""
    return any(
        domain.endswith(td) or td in domain
        for td in search_config.trusted_domains
    )


def extract_sources(
    raw_results: List[Dict[str, Any]],
    topic: str,
    max_sources: int = 10,
) -> List[SourceDocument]:
    """
    Convert raw search result dicts into validated SourceDocument objects.

    Args:
        raw_results: List of dicts from web_search tool.
        topic: The research topic, used for relevance scoring.
        max_sources: Maximum number of sources to return.

    Returns:
        Deduplicated, scored list of SourceDocument objects.
    """
    seen_urls: set[str] = set()
    sources: List[SourceDocument] = []

    for item in raw_results:
        url = item.get("url", "").strip()
        if not url or not _is_valid_url(url):
            logger.debug("Skipping invalid URL: %s", url)
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        domain = _extract_domain(url)
        title = item.get("title", url)
        summary = item.get("snippet", "") or item.get("content", "") or ""

        relevance = _score_relevance(title + " " + summary, topic)
        # Boost trusted domains
        if _is_trusted(domain):
            relevance = min(1.0, relevance + 0.2)

        source = SourceDocument(
            title=title,
            url=url,
            summary=summary[:500],
            domain=domain,
            relevance_score=round(relevance, 3),
            publication_date=item.get("published_date"),
            is_trusted_domain=_is_trusted(domain),
        )
        sources.append(source)

    # Sort by relevance (trusted + relevant first)
    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    return sources[:max_sources]


def deduplicate_sources(sources: List[SourceDocument]) -> List[SourceDocument]:
    """Remove duplicate sources by URL."""
    seen: set[str] = set()
    unique: List[SourceDocument] = []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            unique.append(s)
    return unique
