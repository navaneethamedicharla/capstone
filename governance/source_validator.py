"""
Source validator – checks that sources are real, reachable URLs and
assigns trust scores based on domain reputation.
"""

from __future__ import annotations

import logging
from typing import List, Tuple
from urllib.parse import urlparse

from agents.state import SourceDocument
from config import search_config

logger = logging.getLogger(__name__)


def validate_url_syntax(url: str) -> bool:
    """Return True if the URL is syntactically valid."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def is_trusted_domain(url: str) -> bool:
    """Return True if the URL's domain appears in the trusted domain list."""
    try:
        domain = urlparse(url).netloc.replace("www.", "").lower()
        return any(td in domain for td in search_config.trusted_domains)
    except Exception:
        return False


def validate_sources(sources: List[SourceDocument]) -> Tuple[List[SourceDocument], List[str]]:
    """
    Validate a list of sources, filtering out invalid URLs.

    Args:
        sources: List of SourceDocuments to validate.

    Returns:
        Tuple of (valid_sources, list_of_validation_issues).
    """
    valid: List[SourceDocument] = []
    issues: List[str] = []

    for src in sources:
        if not src.url:
            issues.append(f"Source '{src.title}' has no URL – skipped.")
            continue
        if not validate_url_syntax(src.url):
            issues.append(f"Source '{src.title}' has invalid URL: {src.url} – skipped.")
            continue
        if not src.summary and not src.raw_content:
            issues.append(f"Source '{src.title}' has no content – low trust.")
        src.is_trusted_domain = is_trusted_domain(src.url)
        valid.append(src)

    return valid, issues


def source_trust_score(sources: List[SourceDocument]) -> float:
    """
    Compute an aggregate trust score for a set of sources.
    Trusted domains add +0.2 bonus over their base relevance score.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not sources:
        return 0.0
    scores = []
    for src in sources:
        base = src.relevance_score
        if src.is_trusted_domain:
            base = min(1.0, base + 0.2)
        scores.append(base)
    return round(sum(scores) / len(scores), 3)
