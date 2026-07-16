"""
Confidence scorer – computes an aggregate confidence score for the
final report based on source quality, claim verification, and coverage.

Floor values ensure a completed workflow never scores 0.0:
- No sources but analysis exists  → source_score = 0.4
- No verification                 → claim_score  = 0.4
- Minimum returned score          → 0.3
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.state import FinalReport, ResearchResult, VerificationResult

logger = logging.getLogger(__name__)


def compute_confidence(
    research: Optional[ResearchResult],
    verification: Optional[VerificationResult],
    report: Optional[FinalReport],
) -> float:
    """
    Compute an overall confidence score (0.0–1.0) for the report.

    Weighted formula:
    - Source quality:      25%
    - Citation coverage:   30%
    - Claim verification:  30%
    - Report completeness: 15%

    Floor values:
    - If no sources but analysis exists: source_score = 0.4
    - If no verification data:           claim_score  = 0.4
    - Minimum returned score:            0.3

    Args:
        research: ResearchResult from the research agent.
        verification: VerificationResult from the fact verification agent.
        report: FinalReport from the writer agent.

    Returns:
        Float confidence score between 0.3 and 1.0 for a completed workflow.
    """
    # ── Source quality (25%) ──────────────────────────────────────────────────
    if research and research.sources:
        avg_relevance = sum(s.relevance_score for s in research.sources) / len(research.sources)
        trusted_ratio = (
            sum(1 for s in research.sources if s.is_trusted_domain) / len(research.sources)
        )
        source_score = avg_relevance * 0.6 + trusted_ratio * 0.4
    else:
        # No sources, but the workflow has produced analysis – give a floor value
        source_score = 0.4

    # ── Citation coverage (30%) ───────────────────────────────────────────────
    citation_score = verification.citation_coverage if verification else 0.0

    # ── Claim verification (30%) ──────────────────────────────────────────────
    if verification and (verification.overall_confidence or 0.0) > 0.0:
        claim_score = verification.overall_confidence
    else:
        # No verification data or zero confidence – give a floor value
        claim_score = 0.4

    # ── Report completeness (15%) ─────────────────────────────────────────────
    completeness_score = 0.0
    if report:
        sections = [
            report.executive_summary,
            report.competitor_pricing,
            report.product_updates,
            report.market_signals,
            report.business_risks,
            report.strategic_recommendations,
        ]
        filled = sum(1 for s in sections if s and len(s.strip()) > 10)
        completeness_score = filled / len(sections)

    overall = (
        source_score * 0.25
        + citation_score * 0.30
        + claim_score * 0.30
        + completeness_score * 0.15
    )

    # Never return 0.0 for a completed workflow
    return round(min(1.0, max(0.3, overall)), 3)


def confidence_label(score: float) -> str:
    """Return a human-readable label for a confidence score."""
    if score >= 0.85:
        return "High"
    if score >= 0.65:
        return "Medium"
    if score >= 0.40:
        return "Low"
    return "Very Low"
