"""
Edge routing functions for the LangGraph workflow.
Each function inspects the current state and returns the name of the
next node to execute.
"""

from __future__ import annotations

import logging
from typing import Literal

from agents.state import BriefingState, WorkflowPhase

logger = logging.getLogger(__name__)


def route_after_supervisor(
    state: BriefingState,
) -> Literal["research", "end"]:
    """Route from supervisor: go to research or terminate."""
    if state.get("should_terminate"):
        logger.info("Routing: supervisor → END (terminate=%s)", state.get("termination_reason"))
        return "end"
    return "research"


def route_after_research(
    state: BriefingState,
) -> Literal["analyst", "end"]:
    """
    Route from research: go to analyst or terminate.

    NOTE: We NEVER route to 'end' just because sources is empty.
    Even with 0 live sources the analyst can operate on LLM general
    knowledge, and the research agent always injects a fallback source
    anyway.  The only reason to terminate here is an explicit
    should_terminate flag set upstream.
    """
    if state.get("should_terminate"):
        logger.info(
            "Routing: research → END (should_terminate=True, reason=%s)",
            state.get("termination_reason"),
        )
        return "end"
    # Always proceed to analyst – even with 0 sources, LLM can use general knowledge
    return "analyst"


def route_after_analyst(
    state: BriefingState,
) -> Literal["fact_verification", "end"]:
    """Route from analyst: go to fact verification or terminate."""
    if state.get("should_terminate"):
        return "end"
    return "fact_verification"


def route_after_verification(
    state: BriefingState,
) -> Literal["writer", "end"]:
    """Route from fact verification: go to writer or terminate."""
    if state.get("should_terminate"):
        return "end"
    return "writer"


def route_after_writer(
    state: BriefingState,
) -> Literal["governance", "end"]:
    """Route from writer: go to governance check or terminate."""
    if state.get("should_terminate"):
        return "end"
    report = state.get("final_report")
    if not report:
        logger.warning("No final report – routing to END")
        return "end"
    return "governance"


def route_after_governance(
    state: BriefingState,
) -> Literal["awaiting_approval", "writer", "end"]:
    """
    Route from governance:
    - If refusal triggered → end
    - If governance checks passed → awaiting_approval
    - Otherwise → end
    """
    if state.get("should_terminate"):
        return "end"
    gov = state.get("governance_result")
    if gov and gov.refusal_triggered:
        logger.warning("Governance refusal triggered – routing to END")
        return "end"
    return "awaiting_approval"


def route_after_approval(
    state: BriefingState,
) -> Literal["end", "writer"]:
    """
    Route from human approval gate:
    - Approved → end (report complete)
    - Not approved / revision requested → writer (regenerate)
    """
    if state.get("should_terminate"):
        return "end"

    approved = state.get("human_approved")
    revision_requested = state.get("revision_requested", False)
    revision_count = state.get("revision_count", 0)

    if revision_requested and revision_count < 2:
        logger.info("Human requested revision (count=%d) – re-routing to writer", revision_count)
        state["revision_count"] = revision_count + 1
        state["revision_requested"] = False
        return "writer"

    if approved is True:
        logger.info("Report approved by human – workflow complete")
        return "end"

    # If no decision yet (shouldn't reach here in sync flow), default to end
    return "end"
