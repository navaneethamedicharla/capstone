"""
Extra LangGraph nodes not covered by agents:
- awaiting_approval_node: Human approval gate (records decision in state)
- end_node: Finalises run metadata
"""

from __future__ import annotations

import logging
from datetime import datetime

from agents.base_agent import add_audit, add_trace
from agents.state import BriefingState, WorkflowPhase

logger = logging.getLogger(__name__)


def awaiting_approval_node(state: BriefingState) -> BriefingState:
    """
    Human approval gate node.

    In the Streamlit UI this node is NOT executed as a blocking LangGraph step;
    the UI intercepts the state at AWAITING_APPROVAL phase and shows the preview.
    When the user clicks Approve / Reject the UI manually sets human_approved
    and resumes the graph.

    This node is included so the graph compiles correctly and can be used
    in automated/testing mode where human_approved is pre-set in state.

    Args:
        state: Current BriefingState.

    Returns:
        Unchanged state (approval is set externally).
    """
    approved = state.get("human_approved")
    if approved is None:
        # In automated mode default to approved
        state["human_approved"] = True
        add_trace(
            state,
            "approval_gate",
            "approval",
            "Automated mode: report auto-approved (no human interaction)",
        )
        add_audit(
            state,
            event_type="report_approved",
            agent="approval_gate",
            description="Report auto-approved in automated mode",
        )
    elif approved:
        add_trace(
            state,
            "approval_gate",
            "approval",
            "Report approved by human reviewer",
        )
        add_audit(
            state,
            event_type="report_approved",
            agent="approval_gate",
            description="Report approved by human reviewer",
        )
    else:
        add_trace(
            state,
            "approval_gate",
            "revision",
            "Report rejected by human – requesting revision",
        )
        state["revision_requested"] = True

    state["workflow_phase"] = WorkflowPhase.COMPLETED.value
    return state


def end_node(state: BriefingState) -> BriefingState:
    """
    Terminal node: finalises run metadata.

    Args:
        state: Current BriefingState.

    Returns:
        State with updated run metadata.
    """
    meta = state.get("run_metadata")
    if meta and not meta.completed_at:
        meta.completed_at = datetime.utcnow().isoformat()
        try:
            meta.duration_seconds = round(
                (
                    datetime.fromisoformat(meta.completed_at)
                    - datetime.fromisoformat(meta.started_at)
                ).total_seconds(),
                1,
            )
        except Exception:
            pass
        meta.status = "completed"

    state["workflow_phase"] = WorkflowPhase.COMPLETED.value

    add_trace(
        state,
        "end",
        "completed",
        f"Workflow finished. Duration: {meta.duration_seconds if meta else '?'}s",
    )
    logger.info("Workflow END node reached")
    return state
