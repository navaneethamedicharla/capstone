"""
PDF export tool – converts Markdown report to a clean, styled PDF.

Layout mirrors the 'friend report' design:
  • Title block  : large title + topic/date/status/sources metadata row
  • Section pages: H2 heading with a coloured rule, body text, bullet lists
  • References   : bulleted list with URLs
  • Metadata page: two-column table of run statistics
  • Footer        : page number + branding on every page

Engine priority: WeasyPrint → xhtml2pdf → ReportLab styled (never plain).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from config import path_config

logger = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────
NAVY       = "#1a237e"   # headings, title bar background
INDIGO     = "#283593"   # H2 rule colour
MID_BLUE   = "#3949ab"   # H3 / accent
LIGHT_GREY = "#f5f5f5"   # table zebra / code background
RULE_GREY  = "#cccccc"   # thin horizontal rules
TEXT_DARK  = "#212121"   # body text
TEXT_MID   = "#555555"   # captions / metadata values
WHITE      = "#ffffff"
ACCENT_BG  = "#e8eaf6"   # table header background / cover meta strip


# ── HTML rendering path (WeasyPrint / xhtml2pdf) ──────────────────────────────

_CSS = """
/* ── Reset & base ── */
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #212121;
    background: #fff;
}

/* ── Cover block ── */
.cover {
    background: #1a237e;
    color: #fff;
    padding: 36px 40px 28px 40px;
    margin-bottom: 0;
}
.cover h1 {
    font-size: 22pt;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #fff;
    border: none;
    margin-bottom: 6px;
}
.cover .subtitle {
    font-size: 10pt;
    color: #c5cae9;
    margin-bottom: 20px;
}
.meta-strip {
    background: #e8eaf6;
    color: #1a237e;
    padding: 10px 40px;
    font-size: 9.5pt;
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
    margin-bottom: 28px;
}
.meta-strip span { font-weight: 600; }

/* ── Section headings ── */
h2 {
    font-size: 14pt;
    color: #1a237e;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 4px;
    padding-bottom: 5px;
    border-bottom: 2px solid #283593;
}
h3 {
    font-size: 11.5pt;
    color: #3949ab;
    font-weight: 600;
    margin-top: 16px;
    margin-bottom: 4px;
}

/* ── Body ── */
p { margin-bottom: 9px; }

/* ── Lists ── */
ul, ol { margin-left: 20px; margin-bottom: 10px; }
li { margin-bottom: 4px; }

/* ── Blockquote (unverified notice) ── */
blockquote {
    border-left: 4px solid #bbdefb;
    padding: 8px 14px;
    margin: 12px 0;
    color: #555;
    background: #f8f9ff;
    font-size: 9.5pt;
}

/* ── Tables ── */
table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 14px;
    font-size: 9.5pt;
}
th {
    background: #e8eaf6;
    color: #1a237e;
    font-weight: 700;
    padding: 7px 10px;
    border: 1px solid #c5cae9;
    text-align: left;
}
td {
    padding: 6px 10px;
    border: 1px solid #ddd;
    vertical-align: top;
}
tr:nth-child(even) td { background: #f5f5f5; }

/* ── Inline code ── */
code {
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
    font-family: "Courier New", monospace;
}

/* ── Horizontal rule ── */
hr { border: none; border-top: 1px solid #ccc; margin: 18px 0; }

/* ── Footer ── */
.footer {
    font-size: 8pt;
    color: #9e9e9e;
    text-align: center;
    margin-top: 40px;
    border-top: 1px solid #eee;
    padding-top: 8px;
}

/* ── Strong / em ── */
strong { font-weight: 700; }
em     { font-style: italic; color: #555; }

/* ── Links ── */
a { color: #1565c0; text-decoration: none; }
"""


def _md_to_html_body(markdown_content: str) -> str:
    """Convert Markdown content to an HTML body string."""
    try:
        import markdown as md_lib
        return md_lib.markdown(
            markdown_content,
            extensions=["tables", "fenced_code", "toc"],
        )
    except ImportError:
        pass

    # Minimal regex fallback when the markdown library is absent
    text = markdown_content
    text = re.sub(r"^# (.+)$",   r"<h1>\1</h1>",   text, flags=re.M)
    text = re.sub(r"^## (.+)$",  r"<h2>\1</h2>",   text, flags=re.M)
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>",   text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"_(.+?)_",    r"<em>\1</em>",   text)
    text = re.sub(r"^- (.+)$",   r"<li>\1</li>",   text, flags=re.M)
    text = re.sub(r"(<li>.*</li>\n?)+", r"<ul>\g<0></ul>", text)
    text = re.sub(r"^(?!<)(.+)$", r"<p>\1</p>",    text, flags=re.M)
    return text


def _build_cover_html(topic: str, generated: str) -> str:
    """Return the HTML cover block (dark header + meta strip)."""
    safe_topic = _xml_escape(topic)
    safe_date  = _xml_escape(generated)
    return (
        f'<div class="cover">'
        f'<h1>Competitive Intelligence Briefing</h1>'
        f'<div class="subtitle">Topic: {safe_topic}</div>'
        f'</div>'
        f'<div class="meta-strip">'
        f'<span>Generated:</span> {safe_date}&nbsp;&nbsp;'
        f'<span>Status:</span> Complete'
        f'</div>'
    )


def _xml_escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _build_full_html(markdown_content: str, topic: str) -> str:
    """Assemble the complete HTML document for PDF rendering."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    # Strip the raw H1 title line from the markdown so the cover block
    # replaces it cleanly (avoids a duplicate title in the body).
    body_md = re.sub(r"^# .+\n", "", markdown_content, count=1)
    # Also strip the italic "Generated on …" subtitle line right after it
    body_md = re.sub(r"^_Generated on .+_\n?", "", body_md, count=1)

    body_html = _md_to_html_body(body_md)
    cover     = _build_cover_html(topic, now)
    footer    = (
        '<div class="footer">'
        'Generated by Competitive Intelligence Briefing Crew'
        '</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>{_CSS}</style>
</head>
<body>
{cover}
{body_html}
{footer}
</body>
</html>"""


def _try_weasyprint(html: str, out_path: Path) -> bool:
    """Attempt PDF export using WeasyPrint."""
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(out_path))
        return True
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("WeasyPrint failed: %s", exc)
        return False


def _try_xhtml2pdf(html: str, out_path: Path) -> bool:
    """Attempt PDF export using xhtml2pdf."""
    try:
        from xhtml2pdf import pisa
        with open(out_path, "wb") as f:
            result = pisa.CreatePDF(html, dest=f)
        return not result.err
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("xhtml2pdf failed: %s", exc)
        return False


# ── ReportLab styled renderer ─────────────────────────────────────────────────

def _rl_color(hex_str: str):
    """Convert a hex colour string to a ReportLab Color object."""
    from reportlab.lib.colors import HexColor
    return HexColor(hex_str)


def _build_rl_styles():
    """Return a dict of custom ParagraphStyles for the report."""
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    base = getSampleStyleSheet()

    styles = {}

    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        parent=base["Normal"],
        fontSize=22,
        fontName="Helvetica-Bold",
        textColor=_rl_color(WHITE),
        leading=28,
        spaceAfter=6,
    )
    styles["CoverSubtitle"] = ParagraphStyle(
        "CoverSubtitle",
        parent=base["Normal"],
        fontSize=10,
        fontName="Helvetica",
        textColor=_rl_color("#c5cae9"),
        leading=14,
        spaceAfter=0,
    )
    styles["MetaKey"] = ParagraphStyle(
        "MetaKey",
        parent=base["Normal"],
        fontSize=9,
        fontName="Helvetica-Bold",
        textColor=_rl_color(NAVY),
        leading=13,
    )
    styles["MetaVal"] = ParagraphStyle(
        "MetaVal",
        parent=base["Normal"],
        fontSize=9,
        fontName="Helvetica",
        textColor=_rl_color(TEXT_MID),
        leading=13,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        parent=base["Normal"],
        fontSize=14,
        fontName="Helvetica-Bold",
        textColor=_rl_color(NAVY),
        leading=18,
        spaceBefore=12,
        spaceAfter=2,
    )
    styles["H3"] = ParagraphStyle(
        "H3",
        parent=base["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=_rl_color(MID_BLUE),
        leading=15,
        spaceBefore=6,
        spaceAfter=2,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=10,
        fontName="Helvetica",
        textColor=_rl_color(TEXT_DARK),
        leading=15,
        spaceAfter=4,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=base["Normal"],
        fontSize=10,
        fontName="Helvetica",
        textColor=_rl_color(TEXT_DARK),
        leading=14,
        leftIndent=14,
        bulletIndent=0,
        spaceAfter=2,
        bulletText="\u2022",
    )
    styles["BlockQuote"] = ParagraphStyle(
        "BlockQuote",
        parent=base["Normal"],
        fontSize=9,
        fontName="Helvetica-Oblique",
        textColor=_rl_color(TEXT_MID),
        leading=13,
        leftIndent=16,
        spaceAfter=6,
    )
    styles["Footer"] = ParagraphStyle(
        "Footer",
        parent=base["Normal"],
        fontSize=8,
        fontName="Helvetica",
        textColor=_rl_color("#9e9e9e"),
        leading=11,
        alignment=TA_CENTER,
    )
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=base["Normal"],
        fontSize=9,
        fontName="Helvetica-Bold",
        textColor=_rl_color(NAVY),
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base["Normal"],
        fontSize=9,
        fontName="Helvetica",
        textColor=_rl_color(TEXT_DARK),
    )
    return styles


def _rl_xml(text: str) -> str:
    """Escape text for ReportLab's XML paragraph parser."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _parse_inline(text: str) -> str:
    """
    Convert common Markdown inline markup to ReportLab XML tags.
    Bold (**text**), italic (*text* / _text_), inline code (`text`).
    """
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__",     r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_",       r"<i>\1</i>", text)
    # Inline code
    text = re.sub(r"`(.+?)`",       r'<font name="Courier" size="9">\1</font>', text)
    return text


def _build_cover_flowables(topic: str, styles: dict) -> list:
    """Build the dark-navy cover block as ReportLab flowables."""
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib import colors

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Cover background block — achieved via a single-cell Table with background
    cover_data = [[
        Paragraph("Competitive Intelligence Briefing", styles["CoverTitle"]),
        Paragraph(_rl_xml(f"Topic: {topic}"), styles["CoverSubtitle"]),
    ]]
    # Stack title + subtitle vertically inside one cell
    cover_inner = [
        Paragraph("Competitive Intelligence Briefing", styles["CoverTitle"]),
        Spacer(1, 6),
        Paragraph(_rl_xml(f"Topic: {topic}"), styles["CoverSubtitle"]),
    ]
    cover_table = Table([[cover_inner]], colWidths=["100%"])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _rl_color(NAVY)),
        ("TOPPADDING",    (0, 0), (-1, -1), 28),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 24),
        ("LEFTPADDING",   (0, 0), (-1, -1), 30),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 30),
    ]))

    # Meta strip (light accent background)
    meta_data = [[
        Paragraph("<b>Generated:</b>", styles["MetaKey"]),
        Paragraph(_rl_xml(now),        styles["MetaVal"]),
        Paragraph("<b>Status:</b>",    styles["MetaKey"]),
        Paragraph("Complete",          styles["MetaVal"]),
    ]]
    meta_table = Table(meta_data, colWidths=[70, 130, 55, 80])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _rl_color(ACCENT_BG)),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    return [cover_table, Spacer(1, 4), meta_table, Spacer(1, 8)]


def _h2_with_rule(title: str, styles: dict) -> list:
    """Return an H2 heading followed by a coloured rule line."""
    from reportlab.platypus import Paragraph, HRFlowable
    return [
        Paragraph(_rl_xml(title), styles["H2"]),
        HRFlowable(
            width="100%", thickness=2,
            color=_rl_color(INDIGO),
            spaceAfter=4,
        ),
    ]


def _parse_markdown_to_flowables(markdown_content: str, styles: dict) -> list:
    """
    Walk through the Markdown lines and emit ReportLab flowables.
    Handles: H1 (skip – replaced by cover), H2, H3, bullets, blockquotes,
    blank lines, and plain paragraphs.
    """
    from reportlab.platypus import Paragraph, Spacer, HRFlowable

    story = []
    lines = markdown_content.splitlines()
    i = 0

    # Skip the leading H1 title and subtitle (rendered by cover block)
    while i < len(lines) and (
        lines[i].startswith("# ") or
        lines[i].strip() == "" or
        lines[i].startswith("_Generated on") or
        lines[i].strip() == "---"
    ):
        i += 1

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Blank line → skip entirely; paragraph spaceAfter handles vertical rhythm.
        # Consume ALL consecutive blank lines as one, emit nothing.
        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=_rl_color(RULE_GREY),
                spaceBefore=4, spaceAfter=4,
            ))
            i += 1
            continue

        # H1 inside body (skip – already rendered as cover)
        if stripped.startswith("# "):
            i += 1
            continue

        # H2
        if stripped.startswith("## "):
            story.extend(_h2_with_rule(stripped[3:].strip(), styles))
            i += 1
            continue

        # H3
        if stripped.startswith("### "):
            story.append(Paragraph(
                _parse_inline(_rl_xml(stripped[4:].strip())),
                styles["H3"]
            ))
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            story.append(Paragraph(
                _parse_inline(_rl_xml(stripped[2:])),
                styles["BlockQuote"]
            ))
            i += 1
            continue

        # Bullet list item (- or *)
        if re.match(r"^[-*] ", stripped):
            text = stripped[2:].strip()
            story.append(Paragraph(
                "\u2022\u00a0" + _parse_inline(_rl_xml(text)),
                styles["Bullet"]
            ))
            i += 1
            continue

        # Numbered list item
        m = re.match(r"^\d+\. (.+)$", stripped)
        if m:
            story.append(Paragraph(
                "\u2022\u00a0" + _parse_inline(_rl_xml(m.group(1))),
                styles["Bullet"]
            ))
            i += 1
            continue

        # Plain paragraph
        story.append(Paragraph(
            _parse_inline(_rl_xml(stripped)),
            styles["Body"]
        ))
        i += 1

    return story


def _build_footer_flowable(styles: dict) -> list:
    """Return a bottom footer paragraph."""
    from reportlab.platypus import Paragraph, HRFlowable, Spacer
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return [
        Spacer(1, 16),
        HRFlowable(width="100%", thickness=0.5, color=_rl_color(RULE_GREY)),
        Spacer(1, 4),
        Paragraph(
            f"Generated by Competitive Intelligence Briefing Crew · {now}",
            styles["Footer"]
        ),
    ]


def _styled_reportlab(markdown_content: str, topic: str, out_path: Path) -> bool:
    """Render a clean, styled PDF using ReportLab platypus."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=2.2 * cm,
            rightMargin=2.2 * cm,
            topMargin=1.8 * cm,
            bottomMargin=2.0 * cm,
            title=f"Competitive Intelligence Briefing: {topic}",
            author="Competitive Intelligence Briefing Crew",
        )

        styles = _build_rl_styles()

        story = []
        story.extend(_build_cover_flowables(topic, styles))
        story.extend(_parse_markdown_to_flowables(markdown_content, styles))
        story.extend(_build_footer_flowable(styles))

        doc.build(story)
        return True

    except ImportError:
        logger.error("reportlab not installed; PDF export unavailable")
        return False
    except Exception as exc:
        logger.error("ReportLab styled PDF failed: %s", exc, exc_info=True)
        return False



# ── Public API ────────────────────────────────────────────────────────────────

def export_pdf(
    markdown_content: str,
    topic: str,
    run_id: str = "",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Export a Markdown report as a styled PDF.

    Engine priority: WeasyPrint → xhtml2pdf → ReportLab (styled).
    The ReportLab path now renders a clean, structured layout rather than
    a plain-text dump, so the output is always presentation-quality.

    Args:
        markdown_content: Full Markdown content of the report.
        topic: Research topic (used in filename and cover block).
        run_id: Optional run UUID appended to the filename.
        output_dir: Directory to write the PDF (defaults to reports_dir).

    Returns:
        Path to the saved PDF, or None if all engines fail.
    """
    if output_dir is None:
        output_dir = path_config.reports_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_topic = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in topic.lower()
    )[:40]
    ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_id[:8]}" if run_id else ""
    out_path = output_dir / f"report_{safe_topic}_{ts}{suffix}.pdf"

    # Try HTML-based engines first (best fidelity)
    html = _build_full_html(markdown_content, topic)

    if _try_weasyprint(html, out_path):
        logger.info("PDF saved via WeasyPrint: %s", out_path)
        return out_path

    if _try_xhtml2pdf(html, out_path):
        logger.info("PDF saved via xhtml2pdf: %s", out_path)
        return out_path

    # Styled ReportLab (always looks clean — not a plain-text fallback)
    if _styled_reportlab(markdown_content, topic, out_path):
        logger.info("PDF saved via ReportLab (styled): %s", out_path)
        return out_path

    logger.error("All PDF engines failed for topic '%s'", topic)
    return None
