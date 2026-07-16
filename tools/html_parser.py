"""
HTML parser utility – extracts clean text, title, and meta information
from raw HTML content using BeautifulSoup.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# Tags whose content should be removed entirely
_JUNK_TAGS = [
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "form", "iframe", "button", "svg",
    "figure", "figcaption", "picture",
]


def parse_html(html: str, base_url: str = "") -> Dict[str, Any]:
    """
    Parse raw HTML and extract clean text content.

    Args:
        html: Raw HTML string.
        base_url: Base URL for resolving relative links.

    Returns:
        Dict with keys: title, text, meta_description, links.
    """
    result: Dict[str, Any] = {
        "title": None,
        "text": "",
        "meta_description": None,
        "links": [],
    }

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            result["title"] = title_tag.get_text(strip=True)

        # Extract meta description
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            result["meta_description"] = meta["content"].strip()

        # Remove junk tags
        for tag_name in _JUNK_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Try to find main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"id": re.compile(r"content|main|body", re.I)})
            or soup.find("div", {"class": re.compile(r"content|main|body|article", re.I)})
            or soup.body
        )

        if main_content:
            # Get paragraphs for cleaner text
            paragraphs: List[str] = []
            for elem in main_content.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
                text = elem.get_text(separator=" ", strip=True)
                if len(text) > 30:  # filter short/empty lines
                    paragraphs.append(text)
            result["text"] = "\n\n".join(paragraphs)

        # Extract links
        if base_url:
            links: List[str] = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith("http"):
                    links.append(href)
                elif href.startswith("/"):
                    links.append(urljoin(base_url, href))
            result["links"] = list(dict.fromkeys(links))[:50]  # deduplicate, cap at 50

        if not result["text"] and soup.body:
            result["text"] = soup.body.get_text(separator="\n", strip=True)

    except ImportError:
        logger.error("beautifulsoup4 not installed – cannot parse HTML")
        result["text"] = _fallback_strip(html)
    except Exception as exc:
        logger.warning("HTML parsing error: %s", exc)
        result["text"] = _fallback_strip(html)

    return result


def _fallback_strip(html: str) -> str:
    """Minimal HTML stripper using regex when bs4 is unavailable."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()[:5000]


def extract_publication_date(html: str) -> Optional[str]:
    """Attempt to extract a publication date from HTML meta tags."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Common date meta tags
        for attr_name, attr_val in [
            ("property", "article:published_time"),
            ("name", "date"),
            ("name", "pubdate"),
            ("name", "publication_date"),
            ("itemprop", "datePublished"),
        ]:
            tag = soup.find("meta", attrs={attr_name: attr_val})
            if tag and tag.get("content"):
                return tag["content"][:10]
    except Exception:
        pass
    return None
