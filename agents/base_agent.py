"""
Base agent utilities – shared LLM client factory and trace helpers
used by all agents in the pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from agents.state import AgentStatus, AuditLogEntry, BriefingState, TraceEntry
from config import get_active_api_key, get_llm_base_url, llm_config

logger = logging.getLogger(__name__)


def get_llm(temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """
    Build and return a ChatOpenAI client configured for the active provider.

    Args:
        temperature: Override temperature (uses config default if None).
        max_tokens: Override max_tokens (uses config default if None).

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=llm_config.model,
        temperature=temperature if temperature is not None else llm_config.temperature,
        max_tokens=max_tokens or llm_config.max_tokens,
        api_key=get_active_api_key(),
        base_url=get_llm_base_url(),
        timeout=llm_config.request_timeout,
    )


def add_trace(
    state: BriefingState,
    agent: str,
    phase: str,
    message: str,
    duration_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a TraceEntry to the state execution_trace list (mutates state)."""
    entry = TraceEntry(
        agent=agent,
        phase=phase,
        message=message,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )
    trace: List[TraceEntry] = state.get("execution_trace", [])
    trace.append(entry)
    state["execution_trace"] = trace


def add_audit(
    state: BriefingState,
    event_type: str,
    agent: str,
    description: str,
    data: Optional[Dict[str, Any]] = None,
    severity: str = "info",
) -> None:
    """Append an AuditLogEntry to the state audit_log list (mutates state)."""
    entry = AuditLogEntry(
        event_type=event_type,
        agent=agent,
        description=description,
        data=data or {},
        severity=severity,
    )
    audit_log: List[AuditLogEntry] = state.get("audit_log", [])
    audit_log.append(entry)
    state["audit_log"] = audit_log


def add_error(state: BriefingState, message: str) -> None:
    """Append an error message to the state errors list."""
    errors: List[str] = state.get("errors", [])
    errors.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {message}")
    state["errors"] = errors
    meta = state.get("run_metadata")
    if meta:
        meta.errors += 1


def set_agent_status(state: BriefingState, agent_name: str, status: AgentStatus) -> None:
    """Update a named agent's status in the AgentStatusTracker."""
    tracker = state.get("agent_status")
    if tracker and hasattr(tracker, agent_name):
        setattr(tracker, agent_name, status)


def increment_step(state: BriefingState) -> int:
    """Increment and return the current workflow step counter."""
    step = state.get("current_step", 0) + 1
    state["current_step"] = step
    meta = state.get("run_metadata")
    if meta:
        meta.total_steps = step
    return step


def check_runaway(state: BriefingState) -> bool:
    """
    Return True if the workflow has exceeded max_steps and should terminate.
    Sets the termination flag in state if limit exceeded.
    """
    step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 20)
    if step >= max_steps:
        state["should_terminate"] = True
        state["termination_reason"] = f"Max steps ({max_steps}) exceeded at step {step}"
        logger.warning("Runaway guard triggered at step %d/%d", step, max_steps)
        return True
    return False
