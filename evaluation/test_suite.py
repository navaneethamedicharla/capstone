"""
Evaluation test suite for the Competitive Intelligence Briefing Crew.

Runs evaluation scenarios, collects metrics, and generates a summary report.
Can be run standalone: python -m evaluation.test_suite

Usage:
    python -m evaluation.test_suite                     # Run all scenarios
    python -m evaluation.test_suite --scenario happy_path
    python -m evaluation.test_suite --tags smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.metrics import compute_all_metrics
from evaluation.scenarios import ALL_SCENARIOS, EvaluationScenario
from graph.workflow import run_workflow

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("logs")


def run_scenario(scenario: EvaluationScenario) -> Dict[str, Any]:
    """
    Execute a single evaluation scenario and return results.

    Args:
        scenario: EvaluationScenario to run.

    Returns:
        Dict with scenario name, metrics, passed/failed assertions, and state summary.
    """
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario.name}")
    print(f"  Topic:    {scenario.topic or '(empty)'}")
    print(f"  Desc:     {scenario.description}")
    print(f"{'='*60}")

    start = time.perf_counter()
    try:
        final_state = run_workflow(
            topic=scenario.topic,
            max_steps=scenario.max_steps,
            max_sources=scenario.max_sources,
            rag_enabled=scenario.rag_enabled,
            human_approved=scenario.human_approved,
        )
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 2)
        logger.error("Scenario '%s' raised exception: %s", scenario.name, exc)
        return {
            "scenario": scenario.name,
            "topic": scenario.topic,
            "elapsed_seconds": elapsed,
            "exception": str(exc),
            "passed": False,
            "metrics": {},
            "assertions": [{"name": "no_exception", "passed": False, "detail": str(exc)}],
        }

    elapsed = round(time.perf_counter() - start, 2)
    metrics = compute_all_metrics(final_state, elapsed_seconds=elapsed)

    # ── Assertions ─────────────────────────────────────────────────────────────
    assertions: List[Dict[str, Any]] = []

    # 1. Phase assertion
    actual_phase = final_state.get("workflow_phase", "")
    phase_ok = scenario.expected_phase in actual_phase or actual_phase == "completed"
    assertions.append({
        "name": "workflow_phase",
        "passed": phase_ok,
        "detail": f"expected '{scenario.expected_phase}', got '{actual_phase}'",
    })

    # 2. Sources assertion
    research = final_state.get("research_result")
    source_count = research.total_sources if research else 0
    src_ok = source_count >= scenario.expected_min_sources
    assertions.append({
        "name": "min_sources",
        "passed": src_ok,
        "detail": f"got {source_count}, min={scenario.expected_min_sources}",
    })

    # 3. No crash (should_terminate only if intentional)
    if scenario.name in ("empty_topic", "short_topic"):
        crash_ok = final_state.get("should_terminate", False)
    else:
        crash_ok = not final_state.get("should_terminate", False) or scenario.max_steps <= 5
    assertions.append({
        "name": "no_unexpected_termination",
        "passed": crash_ok,
        "detail": final_state.get("termination_reason", "ok"),
    })

    # 4. Error count
    errors = final_state.get("errors", [])
    error_ok = len(errors) < 5
    assertions.append({
        "name": "error_count_acceptable",
        "passed": error_ok,
        "detail": f"{len(errors)} errors",
    })

    all_passed = all(a["passed"] for a in assertions)
    result = {
        "scenario": scenario.name,
        "topic": scenario.topic,
        "elapsed_seconds": elapsed,
        "passed": all_passed,
        "metrics": metrics,
        "assertions": assertions,
        "errors": errors[:5],
        "workflow_phase": actual_phase,
    }

    # Print summary
    status = "✅ PASS" if all_passed else "❌ FAIL"
    print(f"  Result: {status}  ({elapsed}s)")
    print(f"  Metrics: completion={metrics['task_completion']:.0%}  "
          f"faithfulness={metrics['faithfulness']:.0%}  "
          f"sources={metrics['sources_found']}")
    for a in assertions:
        icon = "✓" if a["passed"] else "✗"
        print(f"    {icon} {a['name']}: {a['detail']}")

    return result


def run_suite(
    scenario_names: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run a subset or all evaluation scenarios.

    Args:
        scenario_names: Optional list of scenario names to run.
        tags: Optional list of tags to filter scenarios.

    Returns:
        List of result dicts.
    """
    scenarios = list(ALL_SCENARIOS.values())

    if scenario_names:
        scenarios = [s for s in scenarios if s.name in scenario_names]
    if tags:
        scenarios = [s for s in scenarios if any(t in s.tags for t in tags)]

    if not scenarios:
        print("No matching scenarios found.")
        return []

    print(f"\n{'='*60}")
    print(f"  COMPETITIVE INTELLIGENCE EVALUATION SUITE")
    print(f"  Running {len(scenarios)} scenario(s)")
    print(f"  Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    avg_completion = sum(r.get("metrics", {}).get("task_completion", 0) for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passed}/{total} passed  ({avg_completion:.0%} avg task completion)")
    print(f"{'='*60}\n")

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")

    return results


def main() -> None:
    """CLI entry point for the evaluation suite."""
    parser = argparse.ArgumentParser(description="Competitive Intelligence Evaluation Suite")
    parser.add_argument("--scenario", nargs="*", help="Scenario name(s) to run")
    parser.add_argument("--tags", nargs="*", help="Filter by tag(s)")
    parser.add_argument("--list", action="store_true", help="List all scenarios")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenarios:")
        for name, s in ALL_SCENARIOS.items():
            print(f"  {name:25s} [{', '.join(s.tags)}] – {s.description[:50]}")
        sys.exit(0)

    results = run_suite(scenario_names=args.scenario, tags=args.tags)
    failed = sum(1 for r in results if not r.get("passed"))
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
