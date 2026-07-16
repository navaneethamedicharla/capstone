"""
Governance Agent – runs all governance checks after the Writer Agent.
Enforces citation coverage, confidence thresholds, and refusal policy
before the report is presented for human approval.
"""

from __future__ import annotations

import logging
import time

from agents.base_agent import (
    add_audit,
    add_error,
    add_trace,
    check_runaway,
    increment_step,
    set_agent_status,
)
from agents.state import (
    AgentStatus,
    BriefingState,
    GovernanceCheckResult,
    WorkflowPhase,
)
from governance.citation_enforcer import enforce_citations
from governance.confidence_scorer import compute_confidence, confidence_label
from governance.refusal_policy import apply_refusal_policy
from governance.source_validator import validate_sources

logger = logging.getLogger(__name__)
AGENT_NAME = "governance_agent"


def governance_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Governance Agent.

    Runs citation enforcement, confidence scoring, source validation,
    and refusal policy. Updates state.governance_result.

    Args:
        state: Current BriefingState.

    Returns:
        Updated BriefingState.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    if check_runaway(state):
        return state

    set_agent_status(state, "governance", AgentStatus.RUNNING)
    add_trace(state, AGENT_NAME, "governance", "Governance checks started")

    report = state.get("final_report")
    verification = state.get("verification_result")
    research = state.get("research_result")

    if not report or not verification:
        msg = "Governance: missing report or verification results"
        add_error(state, msg)
        add_trace(state, AGENT_NAME, "failed", msg)
        set_agent_status(state, "governance", AgentStatus.FAILED)
        return state

    issues: list[str] = []
    warnings: list[str] = []

    # ── 1. Source validation ───────────────────────────────────────────────────
    if research and research.sources:
        _, source_issues = validate_sources(research.sources)
        warnings.extend(source_issues[:5])  # Cap to avoid flooding

    # ── 2. Citation enforcement ───────────────────────────────────────────────
    citations = report.references
    citation_passed, citation_msg = enforce_citations(report, citations)
    # Citation coverage is informational — always a warning, never a blocking issue
    if citation_msg:
        warnings.append(citation_msg)

    # ── 3. Confidence scoring ─────────────────────────────────────────────────
    confidence = compute_confidence(research, verification, report)
    label = confidence_label(confidence)

    # ── 4. Build initial governance result ────────────────────────────────────
    governance_result = GovernanceCheckResult(
        passed=len(issues) == 0,
        citation_coverage=verification.citation_coverage,
        confidence_score=confidence,
        issues=issues,
        warnings=warnings,
    )

    # ── 5. Apply refusal policy ───────────────────────────────────────────────
    governance_result = apply_refusal_policy(report, verification, governance_result)

    # Update report with final confidence
    report.overall_confidence = confidence
    report.citation_coverage = verification.citation_coverage

    state["governance_result"] = governance_result
    state["final_report"] = report

    # Transition
    if governance_result.refusal_triggered:
        state["workflow_phase"] = WorkflowPhase.FAILED.value
        add_audit(
            state,
            event_type="governance_refusal",
            agent=AGENT_NAME,
            description=f"Report refused: {governance_result.refusal_reason}",
            severity="error",
        )
    else:
        state["workflow_phase"] = WorkflowPhase.AWAITING_APPROVAL.value
        add_audit(
            state,
            event_type="governance_passed",
            agent=AGENT_NAME,
            description=f"Governance passed. Confidence: {confidence:.0%} ({label}). "
                        f"Coverage: {verification.citation_coverage:.0%}",
            data={"confidence": confidence, "coverage": verification.citation_coverage},
        )

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
    add_trace(
        state,
        AGENT_NAME,
        "completed",
        f"Governance {'PASSED' if governance_result.passed else 'FAILED'}. "
        f"Confidence: {confidence:.0%} ({label}). "
        f"{'Refusal triggered.' if governance_result.refusal_triggered else 'Ready for approval.'}",
        duration_ms=elapsed_ms,
        metadata={
            "passed": governance_result.passed,
            "confidence": confidence,
            "refusal": governance_result.refusal_triggered,
        },
    )
    set_agent_status(state, "governance", AgentStatus.COMPLETED)
    logger.info(
        "Governance %s. Confidence=%.0f%%, Coverage=%.0f%%",
        "PASSED" if governance_result.passed else "FAILED",
        confidence * 100,
        verification.citation_coverage * 100,
    )
    return state
