"""
PDF export tool – converts Markdown report to PDF.
Tries weasyprint, then xhtml2pdf, and falls back to a plain-text PDF
via reportlab so the export never fails hard.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import path_config

logger = logging.getLogger(__name__)


def _markdown_to_html(markdown_content: str) -> str:
    """Convert Markdown to HTML for PDF rendering."""
    try:
        import markdown as md_lib

        html_body = md_lib.markdown(
            markdown_content,
            extensions=["tables", "fenced_code", "toc"],
        )
    except ImportError:
        # Simple fallback
        import re

        html_body = markdown_content
        html_body = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html_body, flags=re.M)
        html_body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_body, flags=re.M)
        html_body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_body, flags=re.M)
        html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
        html_body = re.sub(r"^- (.+)$", r"<li>\1</li>", html_body, flags=re.M)
        html_body = f"<ul>{html_body}</ul>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto;
         font-size: 14px; line-height: 1.6; color: #222; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 6px; }}
  h2 {{ color: #283593; margin-top: 2em; }}
  h3 {{ color: #3949ab; }}
  a {{ color: #1565c0; }}
  code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
  pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }}
  blockquote {{ border-left: 4px solid #bbdefb; margin-left: 0; padding-left: 16px; color: #555; }}
  li {{ margin-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
  th {{ background: #e8eaf6; }}
  hr {{ border: none; border-top: 1px solid #ccc; margin: 2em 0; }}
</style>
</head>
<body>
{html_body}
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


def _fallback_reportlab(markdown_content: str, out_path: Path) -> bool:
    """Last-resort PDF using reportlab plain text."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(str(out_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        for line in markdown_content.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 8))
                continue
            # Sanitize for reportlab XML parser
            line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if line.startswith("# "):
                story.append(Paragraph(line[2:], styles["Title"]))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], styles["Heading3"]))
            else:
                story.append(Paragraph(line, styles["Normal"]))

        doc.build(story)
        return True
    except ImportError:
        logger.error("reportlab not installed; PDF export unavailable")
        return False
    except Exception as exc:
        logger.error("reportlab PDF failed: %s", exc)
        return False


def export_pdf(
    markdown_content: str,
    topic: str,
    run_id: str = "",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Export a Markdown report as a PDF.

    Tries rendering engines in order: WeasyPrint → xhtml2pdf → ReportLab.
    Returns the path to the saved PDF, or None if all engines fail.

    Args:
        markdown_content: Full Markdown content.
        topic: Topic used in the filename.
        run_id: Optional run ID appended to filename.
        output_dir: Directory to write the PDF.

    Returns:
        Path to the saved PDF or None.
    """
    if output_dir is None:
        output_dir = path_config.reports_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic.lower())[:40]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_id[:8]}" if run_id else ""
    out_path = output_dir / f"report_{safe_topic}_{ts}{suffix}.pdf"

    html = _markdown_to_html(markdown_content)

    if _try_weasyprint(html, out_path):
        logger.info("PDF saved via WeasyPrint: %s", out_path)
        return out_path
    if _try_xhtml2pdf(html, out_path):
        logger.info("PDF saved via xhtml2pdf: %s", out_path)
        return out_path
    if _fallback_reportlab(markdown_content, out_path):
        logger.info("PDF saved via ReportLab: %s", out_path)
        return out_path

    logger.error("All PDF engines failed for topic '%s'", topic)
    return None
