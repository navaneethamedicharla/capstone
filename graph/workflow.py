"""
LangGraph workflow definition for the Competitive Intelligence Briefing Crew.

Graph topology:
  START → supervisor → research → analyst → fact_verification
        → writer → governance → awaiting_approval → END

Conditional edges handle failures, runaway guards, and human revision loops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from agents.analyst_agent import analyst_node
from agents.fact_verification_agent import fact_verification_node
from agents.research_agent import research_node
from agents.state import BriefingState, create_initial_state
from agents.supervisor import supervisor_node
from agents.writer_agent import writer_node
from governance.governance_agent import governance_node
from graph.edges import (
    route_after_analyst,
    route_after_approval,
    route_after_governance,
    route_after_research,
    route_after_supervisor,
    route_after_verification,
    route_after_writer,
)
from graph.nodes import awaiting_approval_node, end_node

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns:
        Compiled LangGraph application.
    """
    graph = StateGraph(BriefingState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("fact_verification", fact_verification_node)
    graph.add_node("writer", writer_node)
    graph.add_node("governance", governance_node)
    graph.add_node("awaiting_approval", awaiting_approval_node)
    graph.add_node("end", end_node)

    # ── Entry edge ────────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Conditional edges ─────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"research": "research", "end": "end"},
    )
    graph.add_conditional_edges(
        "research",
        route_after_research,
        {"analyst": "analyst", "end": "end"},
    )
    graph.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {"fact_verification": "fact_verification", "end": "end"},
    )
    graph.add_conditional_edges(
        "fact_verification",
        route_after_verification,
        {"writer": "writer", "end": "end"},
    )
    graph.add_conditional_edges(
        "writer",
        route_after_writer,
        {"governance": "governance", "end": "end"},
    )
    graph.add_conditional_edges(
        "governance",
        route_after_governance,
        {
            "awaiting_approval": "awaiting_approval",
            "writer": "writer",
            "end": "end",
        },
    )
    graph.add_conditional_edges(
        "awaiting_approval",
        route_after_approval,
        {"end": "end", "writer": "writer"},
    )

    # ── Terminal edge ─────────────────────────────────────────────────────────
    graph.add_edge("end", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


# Module-level compiled graph singleton (built lazily)
_COMPILED_GRAPH = None


def get_graph():
    """Return the compiled graph, building it once on first call."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def run_workflow(
    topic: str,
    max_steps: int = 20,
    max_sources: int = 10,
    rag_enabled: bool = True,
    human_approved: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
) -> BriefingState:
    """
    Run the full competitive intelligence workflow.

    Args:
        topic: Research topic string.
        max_steps: Maximum LangGraph steps before runaway guard activates.
        max_sources: Maximum sources to collect per run.
        rag_enabled: Whether to query the RAG knowledge base.
        human_approved: Pre-set approval (None = requires UI interaction).
        config: Optional LangGraph run config (e.g. for LangSmith tracing).

    Returns:
        Final BriefingState after workflow completion.
    """
    initial_state = create_initial_state(
        topic=topic,
        max_steps=max_steps,
        max_sources=max_sources,
        rag_enabled=rag_enabled,
    )
    if human_approved is not None:
        initial_state["human_approved"] = human_approved

    app = get_graph()
    run_config = config or {}

    logger.info("Starting workflow for topic: '%s'", topic)
    final_state: BriefingState = app.invoke(initial_state, config=run_config)
    logger.info(
        "Workflow finished. Phase: %s, Steps: %d",
        final_state.get("workflow_phase"),
        final_state.get("current_step", 0),
    )
    return final_state


def stream_workflow(
    topic: str,
    max_steps: int = 20,
    max_sources: int = 10,
    rag_enabled: bool = True,
    human_approved: Optional[bool] = True,
):
    """
    Stream workflow state updates for real-time UI rendering.

    Yields state snapshots after each node execution.

    Args:
        topic: Research topic.
        max_steps: Maximum workflow steps.
        max_sources: Maximum sources.
        rag_enabled: RAG enabled flag.
        human_approved: Pre-set approval for streaming mode.

    Yields:
        BriefingState dicts after each node.
    """
    initial_state = create_initial_state(
        topic=topic,
        max_steps=max_steps,
        max_sources=max_sources,
        rag_enabled=rag_enabled,
    )
    if human_approved is not None:
        initial_state["human_approved"] = human_approved

    app = get_graph()
    for state_snapshot in app.stream(initial_state):
        yield state_snapshot
