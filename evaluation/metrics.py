"""
Evaluation metrics for the Competitive Intelligence Briefing Crew.
Computes task completion, trace correctness, citation coverage,
faithfulness, and completion time metrics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.state import BriefingState, FinalReport, VerificationResult


def task_completion_score(state: BriefingState) -> float:
    """
    Score how many pipeline stages completed successfully (0.0–1.0).

    Checks: research, analysis, verification, writing, governance.
    """
    tracker = state.get("agent_status")
    if not tracker:
        return 0.0

    stages = ["research", "analyst", "fact_verification", "writer", "governance"]
    completed = sum(
        1 for s in stages
        if getattr(tracker, s, "pending") in ("completed", "skipped")
    )
    return round(completed / len(stages), 3)


def trace_correctness_score(state: BriefingState) -> float:
    """
    Score whether the execution trace contains all expected phase entries.
    """
    trace = state.get("execution_trace", [])
    expected_phases = {"researching", "analyzing", "verifying", "writing", "governance"}
    found_phases = {t.phase for t in trace}
    overlap = expected_phases & found_phases
    return round(len(overlap) / len(expected_phases), 3)


def citation_coverage_score(state: BriefingState) -> float:
    """Return citation coverage from verification results."""
    verification = state.get("verification_result")
    if not verification:
        return 0.0
    return verification.citation_coverage


def faithfulness_score(state: BriefingState) -> float:
    """
    Faithfulness: ratio of verified to total claims.
    High faithfulness means most claims were backed by sources.
    """
    verification = state.get("verification_result")
    if not verification:
        return 0.0
    total = (
        len(verification.verified_claims)
        + len(verification.rejected_claims)
        + len(verification.unverified_claims)
    )
    if total == 0:
        return 0.0
    return round(len(verification.verified_claims) / total, 3)


def tool_call_accuracy(state: BriefingState) -> float:
    """
    Ratio of successful tool calls to total tool calls.
    Approximated by comparing errors to tool calls in metadata.
    """
    meta = state.get("run_metadata")
    if not meta or meta.tool_calls == 0:
        return 1.0
    error_rate = min(1.0, meta.errors / meta.tool_calls)
    return round(1.0 - error_rate, 3)


def compute_all_metrics(state: BriefingState, elapsed_seconds: float = 0.0) -> Dict[str, Any]:
    """
    Compute and return all evaluation metrics as a dict.

    Args:
        state: Completed BriefingState.
        elapsed_seconds: Wall-clock time for the run.

    Returns:
        Dict of metric_name → value.
    """
    tc = task_completion_score(state)
    trace = trace_correctness_score(state)
    cit = citation_coverage_score(state)
    faith = faithfulness_score(state)
    tool = tool_call_accuracy(state)
    meta = state.get("run_metadata")
    duration = elapsed_seconds or (meta.duration_seconds if meta else 0.0)

    return {
        "task_completion": tc,
        "trace_correctness": trace,
        "citation_coverage": cit,
        "faithfulness": faith,
        "tool_call_accuracy": tool,
        "completion_time_seconds": duration,
        "overall_score": round((tc + trace + cit + faith + tool) / 5, 3),
        "sources_found": meta.total_sources if meta else 0,
        "claims_verified": meta.verified_claims if meta else 0,
        "claims_total": meta.total_claims if meta else 0,
        "errors": meta.errors if meta else 0,
    }
