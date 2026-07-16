"""
Markdown export tool – saves a report as a .md file and returns its path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from config import path_config

logger = logging.getLogger(__name__)


def export_markdown(
    markdown_content: str,
    topic: str,
    run_id: str = "",
    output_dir: Path = None,
) -> Path:
    """
    Save a Markdown report to disk.

    Args:
        markdown_content: Full Markdown text.
        topic: Topic string used in the filename.
        run_id: Optional run ID appended to the filename.
        output_dir: Directory to write the file (defaults to reports/).

    Returns:
        Path to the saved file.
    """
    if output_dir is None:
        output_dir = path_config.reports_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a safe filename
    safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic.lower())[:40]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_id[:8]}" if run_id else ""
    filename = f"report_{safe_topic}_{ts}{suffix}.md"
    filepath = output_dir / filename

    try:
        filepath.write_text(markdown_content, encoding="utf-8")
        logger.info("Markdown report saved to %s", filepath)
    except IOError as exc:
        logger.error("Failed to save Markdown report: %s", exc)
        raise

    return filepath
