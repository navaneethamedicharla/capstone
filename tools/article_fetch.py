"""
Article fetch tool – retrieves and aggressively cleans web page content.

Parser chain (tried in order, first result with >150 chars wins):
  1. trafilatura   – best quality extraction (if installed)
  2. BeautifulSoup – removes ALL nav/script/ads/boilerplate, extracts article text
  3. Raw regex     – minimal HTML-tag stripping, last resort

Content quality guarantees:
  - Scripts, styles, nav menus, cookie banners, ads, footers are stripped.
  - Duplicate paragraphs are removed.
  - Pages with <150 chars of clean text are marked as failed (not included).
  - Publication date and publisher extracted when available.

Never raises an exception. Always returns a dict:
  { url, title, text, publisher, publication_date, error, fetched_at }

Also exposes `concurrent_fetch(urls, max_workers)` for bulk fetching.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

_MAX_CONTENT_BYTES = 500_000   # 500 KB ceiling on raw download
_MIN_TEXT_LENGTH   = 150       # minimum chars of CLEAN text to accept a page
_MIN_PARA_LENGTH   = 40        # minimum chars for a paragraph to be kept

# Tags to fully remove before any text extraction
_JUNK_TAGS = [
    "script", "style", "noscript", "header", "footer", "nav",
    "aside", "form", "iframe", "button", "svg", "figure",
    "figcaption", "picture", "meta", "link", "input", "select",
    "textarea", "label", "fieldset", "legend", "video", "audio",
    "canvas", "map", "object", "embed", "applet",
]

# CSS class/id patterns that indicate boilerplate
_BOILERPLATE_PATTERNS = re.compile(
    r"(cookie|consent|gdpr|banner|popup|modal|overlay|newsletter|"
    r"subscribe|signup|login|sidebar|widget|advertisement|sponsor|"
    r"promo|social|share|comment|related|recommended|trending|"
    r"breadcrumb|pagination|tag-cloud|author-bio|back-to-top|"
    r"skip-to|menu|navbar|topbar|bottom-bar|site-footer|site-header)",
    re.I,
)

# Content-bearing tags
_CONTENT_TAGS = ["p", "article", "section", "h1", "h2", "h3", "h4", "h5", "li", "td", "blockquote"]


# ---------------------------------------------------------------------------
# HTTP session helper
# ---------------------------------------------------------------------------

def _build_session(retries: int = 2) -> requests.Session:
    """Return a requests Session with retry logic and browser-like headers."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(_HEADERS)
    return session


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalize whitespace, remove null bytes and control chars,
    collapse excessive blank lines.
    """
    # Remove null bytes and non-printable control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse whitespace within lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedup_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Remove duplicate and near-duplicate paragraphs.
    Uses a fingerprint of the first 80 chars lowercased.
    """
    seen: set = set()
    unique: List[str] = []
    for p in paragraphs:
        fp = hashlib.md5(p[:80].lower().strip().encode()).hexdigest()
        if fp not in seen:
            seen.add(fp)
            unique.append(p)
    return unique


def _is_boilerplate_tag(tag) -> bool:
    """Return True if a BeautifulSoup tag looks like navigation/ad/boilerplate."""
    tag_id = tag.get("id", "") or ""
    tag_classes = " ".join(tag.get("class", []) or [])
    combined = f"{tag_id} {tag_classes}"
    return bool(_BOILERPLATE_PATTERNS.search(combined))


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_publication_date(soup) -> Optional[str]:
    """
    Try multiple locations to find a publication date.
    Returns ISO-ish string or None.
    """
    # 1. Standard <meta> date tags
    for meta_name in (
        "article:published_time", "og:published_time", "publishdate",
        "date", "DC.date", "article:modified_time", "datePublished",
    ):
        tag = soup.find("meta", {"property": meta_name}) or soup.find("meta", {"name": meta_name})
        if tag and tag.get("content"):
            return tag["content"][:20]

    # 2. JSON-LD
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            import json
            data = json.loads(script.string or "")
            for key in ("datePublished", "dateCreated", "dateModified"):
                if key in data:
                    return str(data[key])[:20]
        except Exception:
            pass

    # 3. <time> element
    time_tag = soup.find("time")
    if time_tag:
        dt = time_tag.get("datetime") or time_tag.get_text(strip=True)
        if dt:
            return dt[:20]

    return None


def _extract_publisher(soup, url: str) -> Optional[str]:
    """
    Try to extract the publisher/site name.
    Falls back to domain.
    """
    # og:site_name
    tag = soup.find("meta", {"property": "og:site_name"})
    if tag and tag.get("content"):
        return tag["content"].strip()

    # application-name
    tag = soup.find("meta", {"name": "application-name"})
    if tag and tag.get("content"):
        return tag["content"].strip()

    # JSON-LD publisher
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            import json
            data = json.loads(script.string or "")
            pub = data.get("publisher", {})
            if isinstance(pub, dict) and pub.get("name"):
                return pub["name"].strip()
        except Exception:
            pass

    # Fall back to domain
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return None


def _extract_title_bs4(soup) -> Optional[str]:
    """Extract page title from og:title, <title>, or first <h1>."""
    tag = soup.find("meta", {"property": "og:title"})
    if tag and tag.get("content"):
        return tag["content"].strip()
    title_tag = soup.find("title")
    if title_tag:
        t = title_tag.get_text(strip=True)
        # Strip " | Site Name" suffixes
        t = re.split(r"\s*[|\-–—]\s*", t)[0].strip()
        if t:
            return t
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(strip=True)
        if t:
            return t
    return None


# ---------------------------------------------------------------------------
# Parser 1 – trafilatura
# ---------------------------------------------------------------------------

def _parse_with_trafilatura(html_bytes: bytes, url: str) -> Optional[str]:
    """
    Extract main article text using trafilatura.
    Returns the extracted text string, or None on failure.
    """
    try:
        import trafilatura  # type: ignore

        html_str = html_bytes.decode("utf-8", errors="replace")
        text = trafilatura.extract(
            html_str,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            url=url,
        )
        if text:
            text = _clean_text(text)
        if text and len(text) > _MIN_TEXT_LENGTH:
            logger.debug("[article_fetch] trafilatura: %d chars from %s", len(text), url[:60])
            return text
    except ImportError:
        logger.debug("[article_fetch] trafilatura not installed")
    except Exception as exc:
        logger.debug("[article_fetch] trafilatura failed for %s: %s", url[:60], exc)
    return None


# ---------------------------------------------------------------------------
# Parser 2 – BeautifulSoup (aggressive cleaning)
# ---------------------------------------------------------------------------

def _parse_with_bs4(html_bytes: bytes, url: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract text, title, publisher, and publication_date with BeautifulSoup.
    Aggressively removes navigation, ads, boilerplate, scripts, and styles.

    Returns (title, text, publisher, publication_date).
    """
    try:
        from bs4 import BeautifulSoup, Comment  # type: ignore

        soup = BeautifulSoup(html_bytes, "html.parser")

        # -- Metadata extraction (before stripping) --
        title = _extract_title_bs4(soup)
        pub_date = _extract_publication_date(soup)
        publisher = _extract_publisher(soup, url)

        # -- Strip all junk tags completely --
        for tag_name in _JUNK_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # -- Remove HTML comments --
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # -- Remove boilerplate elements by class/id pattern --
        for tag in soup.find_all(True):
            try:
                if _is_boilerplate_tag(tag):
                    tag.decompose()
            except Exception:
                pass

        # -- Find the best content container --
        container = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"id": re.compile(r"^(content|main|article|post|story|body)$", re.I)})
            or soup.find("div", {"class": re.compile(r"(article|post|story|content|entry)(-body|-content|-text|-main)?$", re.I)})
            or soup.body
        )

        paragraphs: List[str] = []
        if container:
            for elem in container.find_all(_CONTENT_TAGS):
                raw = elem.get_text(separator=" ", strip=True)
                raw = _clean_text(raw)
                if len(raw) >= _MIN_PARA_LENGTH:
                    paragraphs.append(raw)

        # Deduplicate paragraphs
        paragraphs = _dedup_paragraphs(paragraphs)
        text = "\n\n".join(paragraphs)

        # Fallback: use full body text if paragraphs too sparse
        if len(text) < _MIN_TEXT_LENGTH and soup.body:
            raw_body = soup.body.get_text(separator="\n", strip=True)
            text = _clean_text(raw_body)

        if text and len(text) > _MIN_TEXT_LENGTH:
            logger.debug("[article_fetch] bs4: %d chars, %d paras from %s", len(text), len(paragraphs), url[:60])
            return title, text, publisher, pub_date

        return title, None, publisher, pub_date

    except ImportError:
        logger.debug("[article_fetch] beautifulsoup4 not installed")
    except Exception as exc:
        logger.warning("[article_fetch] bs4 failed for %s: %s", url[:60], exc)
    return None, None, None, None


# ---------------------------------------------------------------------------
# Parser 3 – Raw regex fallback
# ---------------------------------------------------------------------------

def _parse_raw(html_bytes: bytes, url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Minimal HTML stripper using regex – always returns something.
    Removes script/style blocks explicitly before stripping tags.
    Returns (title, text).
    """
    try:
        html_str = html_bytes.decode("utf-8", errors="replace")

        # Remove <script>...</script> and <style>...</style> blocks
        html_str = re.sub(r"<script[^>]*>.*?</script>", " ", html_str, flags=re.I | re.S)
        html_str = re.sub(r"<style[^>]*>.*?</style>", " ", html_str, flags=re.I | re.S)

        # Extract title
        title: Optional[str] = None
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html_str, re.I | re.S)
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip()

        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", html_str)
        text = _clean_text(text)

        if text and len(text) > _MIN_TEXT_LENGTH:
            logger.debug("[article_fetch] raw regex fallback for %s (%d chars)", url[:60], len(text))
            return title, text

    except Exception as exc:
        logger.warning("[article_fetch] raw regex parser failed for %s: %s", url[:60], exc)
    return None, None


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_article(
    url: str,
    timeout: int = 20,
    max_chars: int = 8000,
) -> Dict[str, Optional[str]]:
    """
    Fetch and extract the main text of a web page.

    Tries parsers in order: trafilatura → BeautifulSoup → raw regex.
    Returns the first result with >150 chars of CLEAN text.

    Pages that produce <150 chars of clean text are rejected with an
    error message so the research agent can skip them.

    Args:
        url:       The URL to retrieve.
        timeout:   HTTP request timeout (seconds).
        max_chars: Maximum characters of text to return.

    Returns:
        Dict with keys: url, title, text, publisher, publication_date, error, fetched_at.
        Never raises.
    """
    result: Dict[str, Optional[str]] = {
        "url": url,
        "title": None,
        "text": None,
        "publisher": None,
        "publication_date": None,
        "error": None,
        "fetched_at": None,
    }

    # --- Skip obviously non-fetchable URLs ---
    if not url or not url.startswith(("http://", "https://")):
        result["error"] = "Invalid URL scheme"
        return result

    # --- HTTP request --------------------------------------------------------
    raw_html: Optional[bytes] = None
    try:
        session = _build_session()
        start = time.time()
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            result["error"] = f"Non-HTML content type: {content_type[:40]}"
            return result

        raw_html = response.content[:_MAX_CONTENT_BYTES]
        elapsed_ms = round((time.time() - start) * 1000, 1)
        logger.debug("[article_fetch] Downloaded %s in %.1f ms (%d bytes)", url[:60], elapsed_ms, len(raw_html))

    except requests.exceptions.Timeout:
        result["error"] = f"Timeout after {timeout}s"
        return result
    except requests.exceptions.TooManyRedirects:
        result["error"] = "Too many redirects"
        return result
    except requests.exceptions.ConnectionError as exc:
        result["error"] = f"Connection error: {str(exc)[:80]}"
        return result
    except requests.exceptions.HTTPError as exc:
        result["error"] = f"HTTP {exc.response.status_code}"
        return result
    except Exception as exc:
        result["error"] = f"Fetch error: {str(exc)[:80]}"
        return result

    result["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- Parser 1: trafilatura ------------------------------------------------
    traf_text = _parse_with_trafilatura(raw_html, url)
    if traf_text and len(traf_text) > _MIN_TEXT_LENGTH:
        # Extract metadata with bs4 since trafilatura doesn't return it
        title: Optional[str] = None
        publisher: Optional[str] = None
        pub_date: Optional[str] = None
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(raw_html, "html.parser")
            title = _extract_title_bs4(soup)
            publisher = _extract_publisher(soup, url)
            pub_date = _extract_publication_date(soup)
        except Exception:
            pass
        result.update({
            "title": title,
            "text": traf_text[:max_chars],
            "publisher": publisher,
            "publication_date": pub_date,
        })
        return result

    # --- Parser 2: BeautifulSoup (aggressive) --------------------------------
    bs4_title, bs4_text, bs4_publisher, bs4_date = _parse_with_bs4(raw_html, url)
    if bs4_text and len(bs4_text) > _MIN_TEXT_LENGTH:
        result.update({
            "title": bs4_title,
            "text": bs4_text[:max_chars],
            "publisher": bs4_publisher,
            "publication_date": bs4_date,
        })
        return result

    # --- Parser 3: Raw regex (last resort) -----------------------------------
    raw_title, raw_text = _parse_raw(raw_html, url)
    if raw_text and len(raw_text) > _MIN_TEXT_LENGTH:
        result.update({
            "title": raw_title or bs4_title,
            "text": raw_text[:max_chars],
        })
        return result

    # All parsers failed quality gate
    result["error"] = f"Insufficient content extracted (<{_MIN_TEXT_LENGTH} chars)"
    logger.info("[article_fetch] Quality gate failed for %s — skipping", url[:60])
    return result


# ---------------------------------------------------------------------------
# Concurrent fetch
# ---------------------------------------------------------------------------

def concurrent_fetch(
    urls: List[str],
    max_workers: int = 5,
    timeout: int = 20,
    max_chars: int = 8000,
) -> List[Dict[str, Optional[str]]]:
    """
    Fetch multiple URLs concurrently using a thread pool.

    Args:
        urls:        List of URLs to fetch.
        max_workers: Maximum parallel worker threads.
        timeout:     Per-request timeout in seconds.
        max_chars:   Maximum characters of text per article.

    Returns:
        List of result dicts in the same order as `urls`.
        Each dict has: url, title, text, publisher, publication_date, error, fetched_at.
        Never raises.
    """
    if not urls:
        return []

    ordered: Dict[int, Dict[str, Optional[str]]] = {}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(fetch_article, url, timeout, max_chars): idx
                for idx, url in enumerate(urls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered[idx] = future.result()
                except Exception as exc:
                    ordered[idx] = {
                        "url": urls[idx],
                        "title": None,
                        "text": None,
                        "publisher": None,
                        "publication_date": None,
                        "error": str(exc),
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
    except Exception as exc:
        logger.error("[article_fetch] concurrent_fetch pool error: %s", exc)
        for idx, url in enumerate(urls):
            if idx not in ordered:
                ordered[idx] = {
                    "url": url,
                    "title": None,
                    "text": None,
                    "publisher": None,
                    "publication_date": None,
                    "error": f"Pool error: {exc}",
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

    results = [ordered[i] for i in range(len(urls))]
    success = sum(1 for r in results if r.get("text"))
    failed  = sum(1 for r in results if r.get("error"))
    logger.info(
        "[article_fetch] concurrent_fetch done: %d/%d clean articles, %d failed",
        success, len(urls), failed,
    )
    return results
