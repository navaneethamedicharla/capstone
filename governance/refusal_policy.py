"""
Refusal policy – decides whether to block or warn about report content
that violates safety or completeness thresholds.

Policy (lenient):
- Refuse ONLY if hallucination markers are detected in key sections.
- Completeness issues (short sections) are treated as warnings, not refusals.
- Confidence score does NOT trigger a refusal.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from agents.state import FinalReport, GovernanceCheckResult, VerificationResult
from config import governance_config

logger = logging.getLogger(__name__)

# Content patterns that should never appear in a published report
_BLOCKED_PHRASES: List[str] = [
    "i cannot",
    "i don't know",
    "as an ai",
    "i'm not able to",
    "i apologize",
    "unfortunately i",
]


def _check_no_hallucination_markers(text: str) -> List[str]:
    """Flag AI refusal / confusion phrases that suggest hallucination."""
    issues: List[str] = []
    lower = text.lower()
    for phrase in _BLOCKED_PHRASES:
        if phrase in lower:
            issues.append(f"Potential hallucination marker found: '{phrase}'")
    return issues


def check_completeness(report: FinalReport) -> List[str]:
    """Check that all required sections are non-empty (minimum 10 chars)."""
    required = {
        "Executive Summary": report.executive_summary,
        "Competitor Pricing": report.competitor_pricing,
        "Product Updates": report.product_updates,
        "Market Signals": report.market_signals,
        "Business Risks": report.business_risks,
        "Strategic Recommendations": report.strategic_recommendations,
    }
    issues: List[str] = []
    for name, content in required.items():
        if not content or len(content.strip()) < 10:
            issues.append(f"Section '{name}' is missing or too short.")
    return issues


def evaluate_refusal(
    report: FinalReport,
    verification: VerificationResult,
) -> Tuple[bool, List[str]]:
    """
    Determine whether to refuse publication of the report.

    Refusal is triggered ONLY by:
    1. Hallucination markers found in key sections.
    2. A section is completely empty (less than 10 chars).

    Confidence score does NOT trigger refusal.

    Args:
        report: Assembled FinalReport.
        verification: Verification results (used for informational warnings only).

    Returns:
        (should_refuse, reasons)
    """
    if not governance_config.enable_refusal_policy:
        return False, []

    hallucination_issues: List[str] = []
    completeness_issues: List[str] = []

    # Check for hallucination markers in executive summary and recommendations
    full_text = (report.executive_summary or "") + (report.strategic_recommendations or "")
    hallucination_issues.extend(_check_no_hallucination_markers(full_text))

    # Check completeness – completely empty sections only
    completeness_issues.extend(check_completeness(report))

    # Refuse only on hallucination markers or truly empty sections
    should_refuse = bool(hallucination_issues) or bool(completeness_issues)

    reasons = hallucination_issues + completeness_issues
    return should_refuse, reasons


def apply_refusal_policy(
    report: FinalReport,
    verification: VerificationResult,
    governance_result: GovernanceCheckResult,
) -> GovernanceCheckResult:
    """
    Apply refusal policy and update governance result.

    Args:
        report: The FinalReport to evaluate.
        verification: VerificationResult with confidence scores.
        governance_result: GovernanceCheckResult to update.

    Returns:
        Updated GovernanceCheckResult.
    """
    should_refuse, reasons = evaluate_refusal(report, verification)

    if should_refuse:
        governance_result.refusal_triggered = True
        governance_result.refusal_reason = "; ".join(reasons)
        governance_result.issues.extend(reasons)
        governance_result.passed = False
        logger.warning("Refusal policy triggered: %s", governance_result.refusal_reason)
    else:
        # Completeness / confidence are warnings only
        governance_result.warnings.extend(reasons)

    return governance_result
