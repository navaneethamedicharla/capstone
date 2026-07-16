"""
Citation generator – formats source documents into numbered citations
and injects citation markers into report text.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from agents.state import Citation, SourceDocument

logger = logging.getLogger(__name__)


def generate_citations(sources: List[SourceDocument]) -> List[Citation]:
    """
    Convert a list of SourceDocuments into numbered Citation objects.

    Args:
        sources: List of source documents to cite.

    Returns:
        List of Citation objects with incrementing numbers.
    """
    citations: List[Citation] = []
    for idx, source in enumerate(sources, start=1):
        citation = Citation(
            id=source.id,
            number=idx,
            title=source.title or source.url,
            url=source.url,
            domain=source.domain,
            accessed_date=datetime.utcnow().strftime("%Y-%m-%d"),
        )
        citations.append(citation)
    return citations


def format_citation_inline(citation_number: int) -> str:
    """Return an inline citation marker like [1]."""
    return f"[{citation_number}]"


def build_citation_map(citations: List[Citation]) -> Dict[str, int]:
    """
    Build a mapping from source_id → citation_number.

    Args:
        citations: List of Citation objects.

    Returns:
        Dict mapping source ID to citation number.
    """
    return {c.id: c.number for c in citations}


def format_references_section(citations: List[Citation]) -> str:
    """
    Format citations as a compact numbered References section in Markdown.

    Each citation shows: [N] Short title. domain (accessed date)
    Long titles are truncated to 60 chars. Full URLs are omitted to keep
    the section readable — domain gives enough context for sourcing.

    Args:
        citations: List of Citation objects.

    Returns:
        Formatted references Markdown string.
    """
    if not citations:
        return "## References\n\n_No sources cited._\n"

    lines = ["## References\n"]
    for c in citations:
        # Truncate long titles
        title = c.title or c.url or "Untitled"
        if len(title) > 65:
            title = title[:62].rstrip() + "..."

        # Show domain instead of full URL
        domain = c.domain or ""
        if not domain and c.url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(c.url).netloc.replace("www.", "")
            except Exception:
                domain = c.url[:40]

        date_str = f" · {c.accessed_date}" if c.accessed_date else ""
        lines.append(f"[{c.number}] {title} — _{domain}_{date_str}")

    return "\n".join(lines)


def inject_citations_into_text(text: str, citation_map: Dict[str, int]) -> str:
    """
    Replace source ID placeholders «source_id» with [N] inline citations.

    Args:
        text: Report text possibly containing «id» placeholders.
        citation_map: Mapping of source_id → citation_number.

    Returns:
        Text with placeholders replaced by [N] markers.
    """
    for source_id, number in citation_map.items():
        placeholder = f"«{source_id}»"
        text = text.replace(placeholder, f"[{number}]")
    return text


def calculate_citation_coverage(text: str, citations: List[Citation]) -> float:
    """
    Estimate what fraction of the report body contains citation markers.

    Args:
        text: Report body text.
        citations: Citations used in the report.

    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if not citations:
        return 0.0
    found = sum(1 for c in citations if f"[{c.number}]" in text)
    return round(found / len(citations), 3)
