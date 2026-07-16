"""
JSON logger – writes structured execution logs to a per-run JSONL file.
Captures timing, tool calls, agent transitions, and errors for debugging
and evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from config import path_config

logger = logging.getLogger(__name__)


def _run_log_path(run_id: str = "") -> Path:
    """Return the JSONL log file path for a run."""
    if run_id:
        return path_config.logs_dir / f"run_{run_id[:8]}.jsonl"
    return path_config.logs_dir / "run.jsonl"


def log_event(
    event_type: str,
    agent: str,
    message: str,
    run_id: str = "",
    data: Optional[Dict[str, Any]] = None,
    level: str = "info",
    log_path: Optional[Path] = None,
) -> None:
    """
    Append a structured JSON event to the run log.

    Args:
        event_type: Type of event (tool_call, step, error, etc.).
        agent: Name of the agent logging this event.
        message: Human-readable message.
        run_id: Run ID for file routing.
        data: Optional arbitrary metadata dict.
        level: Log level string.
        log_path: Optional explicit file path override.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "level": level,
        "event_type": event_type,
        "agent": agent,
        "message": message,
        "data": data or {},
    }
    path = log_path or _run_log_path(run_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.error("JSON log write failed: %s", exc)


@contextmanager
def timed_step(
    event_type: str,
    agent: str,
    message: str,
    run_id: str = "",
    data: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that logs an event with timing on entry and exit.

    Usage:
        with timed_step("tool_call", "research_agent", "Searching web") as ctx:
            ctx["query"] = "AI CRM tools"
            results = search(...)

    Args:
        event_type: Event type string.
        agent: Agent name.
        message: Log message.
        run_id: Run ID.
        data: Initial data dict (mutated by caller inside the block).
    """
    ctx: Dict[str, Any] = data.copy() if data else {}
    start = time.perf_counter()
    log_event(event_type, agent, f"START: {message}", run_id=run_id, data=ctx, level="info")
    try:
        yield ctx
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        ctx["duration_ms"] = elapsed_ms
        log_event(event_type, agent, f"END: {message}", run_id=run_id, data=ctx, level="info")
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        ctx["duration_ms"] = elapsed_ms
        ctx["error"] = str(exc)
        log_event(event_type, agent, f"ERROR: {message}: {exc}", run_id=run_id, data=ctx, level="error")
        raise


def load_run_log(run_id: str = "", log_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load all log entries for a run.

    Args:
        run_id: Run ID to locate the log file.
        log_path: Optional explicit path override.

    Returns:
        List of log entry dicts.
    """
    path = log_path or _run_log_path(run_id)
    entries: List[Dict[str, Any]] = []
    if not path.exists():
        return entries
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        logger.error("Failed to read run log: %s", exc)
    return entries
