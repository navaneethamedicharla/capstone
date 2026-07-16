"""
Audit logger – writes structured audit log entries to a rotating JSONL file.
Every governance-relevant event (tool call, agent transition, refusal, etc.)
is recorded here for compliance review.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.state import AuditLogEntry
from config import path_config

logger = logging.getLogger(__name__)


def _audit_log_path(run_id: str = "") -> Path:
    """Return path to the audit log JSONL file for a given run."""
    if run_id:
        return path_config.logs_dir / f"audit_{run_id[:8]}.jsonl"
    return path_config.logs_dir / "audit.jsonl"


def write_audit_entry(
    entry: AuditLogEntry,
    run_id: str = "",
    audit_log_path: Optional[Path] = None,
) -> None:
    """
    Append an AuditLogEntry to the JSONL audit file.

    Args:
        entry: The AuditLogEntry to write.
        run_id: Optional run ID used in the filename.
        audit_log_path: Optional explicit path override.
    """
    path = audit_log_path or _audit_log_path(run_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.dict()) + "\n")
    except Exception as exc:
        logger.error("Failed to write audit entry: %s", exc)


def create_audit_entry(
    event_type: str,
    agent: str,
    description: str,
    data: Optional[Dict[str, Any]] = None,
    severity: str = "info",
) -> AuditLogEntry:
    """
    Convenience factory for AuditLogEntry objects.

    Args:
        event_type: Short event type string (e.g. "tool_call", "agent_transition").
        agent: Name of the agent triggering the event.
        description: Human-readable description.
        data: Optional key-value metadata.
        severity: Log severity level (info/warning/error).

    Returns:
        AuditLogEntry object.
    """
    return AuditLogEntry(
        timestamp=datetime.utcnow().isoformat(),
        event_type=event_type,
        agent=agent,
        description=description,
        data=data or {},
        severity=severity,
    )


def load_audit_log(run_id: str = "", audit_log_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Read all audit entries for a run from the JSONL file.

    Args:
        run_id: Run ID to identify the correct file.
        audit_log_path: Optional explicit path override.

    Returns:
        List of dicts (one per audit entry).
    """
    path = audit_log_path or _audit_log_path(run_id)
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
        logger.error("Failed to read audit log: %s", exc)
    return entries


def format_audit_summary(entries: List[AuditLogEntry]) -> str:
    """
    Format audit log entries as a Markdown table.

    Args:
        entries: List of AuditLogEntry objects.

    Returns:
        Markdown table string.
    """
    if not entries:
        return "_No audit entries._"

    rows = ["| Timestamp | Severity | Agent | Event | Description |",
            "|-----------|----------|-------|-------|-------------|"]
    for e in entries:
        ts = e.timestamp[:19]
        rows.append(f"| {ts} | {e.severity} | {e.agent} | {e.event_type} | {e.description[:60]} |")
    return "\n".join(rows)
