"""
Supervisor Agent – initialises the workflow, validates input, coordinates
agents, prevents runaway loops, and handles top-level failures.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict

from agents.base_agent import (
    add_audit,
    add_error,
    add_trace,
    check_runaway,
    increment_step,
    set_agent_status,
)
from agents.state import AgentStatus, BriefingState, WorkflowPhase

logger = logging.getLogger(__name__)

AGENT_NAME = "supervisor"


def supervisor_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Supervisor Agent.

    Responsibilities:
    - Validate topic input.
    - Initialise run metadata.
    - Guard against runaway loops.
    - Set workflow phase.
    - Coordinate hand-off to Research Agent.

    Args:
        state: Current BriefingState.

    Returns:
        Updated BriefingState.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    set_agent_status(state, "supervisor", AgentStatus.RUNNING)
    add_trace(state, AGENT_NAME, "initializing", "Supervisor agent started")

    # ── Input validation ─────────────────────────────────────────────────────
    topic: str = state.get("topic", "").strip()
    if not topic:
        err = "No topic provided. Cannot begin research."
        add_error(state, err)
        set_agent_status(state, "supervisor", AgentStatus.FAILED)
        state["should_terminate"] = True
        state["termination_reason"] = err
        add_trace(state, AGENT_NAME, "failed", err)
        return state

    if len(topic) < 3:
        err = f"Topic too short: '{topic}'. Provide a meaningful topic."
        add_error(state, err)
        set_agent_status(state, "supervisor", AgentStatus.FAILED)
        state["should_terminate"] = True
        state["termination_reason"] = err
        return state

    # ── Runaway guard ─────────────────────────────────────────────────────────
    if check_runaway(state):
        set_agent_status(state, "supervisor", AgentStatus.FAILED)
        return state

    # ── Initialise metadata ───────────────────────────────────────────────────
    meta = state.get("run_metadata")
    if meta:
        meta.topic = topic
        meta.started_at = datetime.utcnow().isoformat()
        meta.status = "running"

    # ── Set initial max limits if not already set ─────────────────────────────
    if not state.get("max_steps"):
        state["max_steps"] = 20
    if not state.get("max_sources"):
        state["max_sources"] = 10

    # ── Transition to research phase ──────────────────────────────────────────
    state["workflow_phase"] = WorkflowPhase.RESEARCHING.value

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
    add_trace(
        state,
        AGENT_NAME,
        "completed",
        f"Supervisor initialised. Topic: '{topic}'. Handing off to Research Agent.",
        duration_ms=elapsed_ms,
        metadata={"topic": topic, "max_steps": state["max_steps"]},
    )
    add_audit(
        state,
        event_type="workflow_start",
        agent=AGENT_NAME,
        description=f"Workflow started for topic: {topic}",
        data={"topic": topic, "run_id": state.get("run_id")},
    )

    set_agent_status(state, "supervisor", AgentStatus.COMPLETED)
    logger.info("Supervisor completed. Topic: '%s'", topic)
    return state
