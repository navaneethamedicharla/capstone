"""
Evaluation scenarios – defines test cases that exercise different
workflow paths: happy path, failure modes, edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationScenario:
    """A single evaluation test scenario."""

    name: str
    topic: str
    description: str
    expected_phase: str
    expected_min_sources: int = 0
    expected_min_claims: int = 0
    expected_citation_coverage: float = 0.0
    rag_enabled: bool = False
    human_approved: Optional[bool] = True
    max_sources: int = 5
    max_steps: int = 15
    tags: List[str] = field(default_factory=list)
    mock_overrides: Dict[str, Any] = field(default_factory=dict)


# ── Standard test scenarios ────────────────────────────────────────────────────

HAPPY_PATH = EvaluationScenario(
    name="happy_path",
    topic="Artificial Intelligence CRM software market 2025",
    description=(
        "Standard end-to-end run. Should produce a complete report "
        "with citations and pass governance."
    ),
    expected_phase="completed",
    expected_min_sources=3,
    expected_min_claims=2,
    expected_citation_coverage=0.0,  # Relaxed – real search may vary
    tags=["smoke", "e2e"],
)

ELECTRIC_VEHICLES = EvaluationScenario(
    name="electric_vehicles",
    topic="Electric vehicles market competition 2025",
    description="EV market briefing – diverse competitor landscape.",
    expected_phase="completed",
    expected_min_sources=2,
    expected_min_claims=1,
    expected_citation_coverage=0.0,
    tags=["e2e"],
)

CLOUD_COMPUTING = EvaluationScenario(
    name="cloud_computing",
    topic="Cloud computing providers competitive landscape AWS Azure GCP",
    description="Major cloud provider analysis.",
    expected_phase="completed",
    expected_min_sources=2,
    expected_min_claims=1,
    expected_citation_coverage=0.0,
    tags=["e2e"],
)

EMPTY_TOPIC = EvaluationScenario(
    name="empty_topic",
    topic="",
    description="Empty topic should fail gracefully at supervisor.",
    expected_phase="initializing",
    expected_min_sources=0,
    tags=["failure", "validation"],
)

SHORT_TOPIC = EvaluationScenario(
    name="short_topic",
    topic="AI",
    description="Very short topic – supervisor should reject.",
    expected_phase="initializing",
    expected_min_sources=0,
    tags=["failure", "validation"],
)

RUNAWAY_GUARD = EvaluationScenario(
    name="runaway_guard",
    topic="Cybersecurity market competitive analysis",
    description="Max steps set very low to test runaway guard.",
    expected_phase="completed",
    max_steps=3,
    max_sources=3,
    tags=["runaway", "safety"],
)

HEALTHCARE_AI = EvaluationScenario(
    name="healthcare_ai",
    topic="Healthcare AI diagnostics market 2025",
    description="Healthcare AI sector analysis.",
    expected_phase="completed",
    expected_min_sources=2,
    tags=["e2e", "healthcare"],
)

CYBERSECURITY = EvaluationScenario(
    name="cybersecurity",
    topic="Cybersecurity vendors market share endpoint detection 2025",
    description="Cybersecurity sector analysis.",
    expected_phase="completed",
    expected_min_sources=2,
    tags=["e2e"],
)

# All scenarios indexed by name
ALL_SCENARIOS: Dict[str, EvaluationScenario] = {
    s.name: s
    for s in [
        HAPPY_PATH,
        ELECTRIC_VEHICLES,
        CLOUD_COMPUTING,
        EMPTY_TOPIC,
        SHORT_TOPIC,
        RUNAWAY_GUARD,
        HEALTHCARE_AI,
        CYBERSECURITY,
    ]
}

SAMPLE_TOPICS = [
    "Artificial Intelligence CRM software market 2025",
    "Electric Vehicles market competition 2025",
    "Cloud Computing providers AWS Azure GCP 2025",
    "Cybersecurity vendors endpoint detection market",
    "Healthcare AI diagnostics market trends",
]
