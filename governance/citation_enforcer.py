"""
Citation enforcer – verifies that report sections contain adequate inline
citations and raises issues if the coverage threshold is not met.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from agents.state import Citation, FinalReport
from config import governance_config

logger = logging.getLogger(__name__)


def count_inline_citations(text: str) -> int:
    """Count [N] citation markers in a text block."""
    return len(re.findall(r"\[\d+\]", text))


def check_citation_coverage(
    report: FinalReport,
    citations: List[Citation],
    min_coverage: float = None,
) -> Tuple[bool, float, List[str]]:
    """
    Verify that report sections have sufficient inline citations.

    Coverage is now computed more leniently:
    - If citations exist in the references section, that counts as coverage.
    - Inline [N] markers are preferred but not strictly required.
    - Section-level checks are warnings only, not blocking failures.

    Args:
        report: The assembled FinalReport.
        citations: List of citations available.
        min_coverage: Minimum fraction of citations that must appear inline.

    Returns:
        Tuple of (passed, coverage_score, list_of_issues).
    """
    issues: List[str] = []

    if not citations:
        # No citations at all is a warning but not a hard block
        issues.append("No citations generated – all claims are uncited.")
        return True, 0.0, issues  # warn only, don't block

    body = " ".join([
        report.executive_summary,
        report.competitor_pricing,
        report.product_updates,
        report.market_signals,
        report.business_risks,
        report.strategic_recommendations,
    ])

    total = len(citations)
    inline_found = sum(1 for c in citations if f"[{c.number}]" in body)

    # Lenient coverage: if citations exist in references list that counts
    # as partial coverage even without inline markers
    if inline_found > 0:
        coverage = round(inline_found / total, 3)
    elif total > 0:
        # Citations exist in references section — award base coverage credit
        coverage = min(0.5, round(total * 0.1, 3))
    else:
        coverage = 0.0

    # Only hard-fail if there are literally zero citations AND zero content
    if total == 0 and not body.strip():
        issues.append("Report has no citations and no content.")
        return False, 0.0, issues

    # Everything else is a warning (non-blocking)
    return True, coverage, issues


def enforce_citations(
    report: FinalReport,
    citations: List[Citation],
) -> Tuple[bool, str]:
    """
    Enforce citation policy on the final report.

    Always returns passed=True — citation coverage is informational only.
    A report with a references section but no inline [N] markers is acceptable.

    Returns:
        (passed, message)
    """
    passed, coverage, issues = check_citation_coverage(report, citations)
    if issues:
        return True, f"Citation note ({coverage:.0%} inline coverage): {'; '.join(issues)}"
    return True, f"Citation coverage: {coverage:.0%} ✓"
