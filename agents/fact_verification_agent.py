"""
Fact Verification Agent – verifies every extracted claim against research sources.

Key behaviours:
- Never fails because there are 0 claims — synthetic placeholders are created.
- Verifies claims in batches of 15 to stay within token limits.
- Each batch gets the full source context independently.
- Detailed logs: verified / rejected / unverified counts with reasons.
- Any claim the LLM does not cover defaults to UNVERIFIED (never dropped).
- Errors in one batch do not stop other batches.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import (
    add_audit,
    add_error,
    add_trace,
    call_with_retry,
    check_runaway,
    get_llm,
    increment_step,
    set_agent_status,
)
from agents.state import (
    AgentStatus,
    BriefingState,
    Claim,
    ClaimStatus,
    VerificationResult,
    WorkflowPhase,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "fact_verification_agent"

_BATCH_SIZE = 15   # claims per LLM call

_SYSTEM_PROMPT = """You are a fact-checker for a competitive intelligence firm.

Verify each numbered claim against the provided source excerpts.

Rules:
- VERIFIED (0.7–0.95): The claim is explicitly stated or clearly implied by the sources.
  Always include a supporting_evidence quote or paraphrase.
- UNVERIFIED (0.4–0.65): The claim is plausible, consistent with sources, but not
  directly evidenced in the provided text. Include a short note on what is missing.
- REJECTED (0.0–0.3): ONLY for claims that directly contradict the sources.
  Include the specific contradiction in rejection_reason.

Critical guidance:
- Default to UNVERIFIED, not REJECTED, when evidence is absent.
- A company existing, operating, or having a product is almost always at least UNVERIFIED.
- REJECTED should be rare (under 10% of claims).
- Return exactly one JSON object per claim, in the same order as given.
- Do NOT skip any claim.

Return ONLY a JSON array. No prose.

Schema per item:
{
  "claim_index": 0,
  "status": "verified|unverified|rejected",
  "confidence": 0.0,
  "supporting_evidence": "string or null",
  "rejection_reason": "string or null",
  "verification_note": "one sentence explaining the outcome"
}"""


def _build_batch_prompt(batch: List[Claim], source_texts: List[str]) -> str:
    """Build a verification prompt for one batch of claims."""
    claims_block = "\n".join(
        f"{i}. [{c.category}] {c.text}"
        for i, c in enumerate(batch)
    )
    sources_block = "\n\n---\n\n".join(source_texts[:8])[:10000]
    return (
        f"Verify these {len(batch)} claims against the source excerpts below.\n\n"
        f"CLAIMS:\n{claims_block}\n\n"
        f"SOURCE EXCERPTS:\n{sources_block}\n\n"
        f"Return a JSON array with exactly {len(batch)} objects (indices 0 to {len(batch)-1}). "
        f"No text before or after the JSON."
    )


def _parse_verification_json(raw: str, expected_count: int) -> list:
    """
    Parse a JSON array from LLM response.
    Returns list of dicts, empty list on failure.
    """
    if not raw:
        return []
    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("["):
                text = candidate
                break

    # Find opening bracket
    start = text.find("[")
    if start == -1:
        logger.debug("[fact_verification] No JSON array found in response")
        return []
    text = text[start:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt repair: truncate after last complete object
    last_good = text.rfind("},")
    if last_good > 0:
        repaired = text[:last_good + 1] + "]"
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    logger.debug("[fact_verification] Could not parse JSON: %s...", raw[:100])
    return []


def _build_source_texts(research) -> List[str]:
    """Extract clean source texts for verification context."""
    texts: List[str] = []
    if not research:
        return texts
    for src in research.sources:
        content = (src.raw_content or src.summary or "").strip()
        if len(content) < 50:
            continue
        header = f"[Source: {src.title} | {src.url}]"
        texts.append(f"{header}\n{content[:1200]}")
    return texts


def _apply_batch_results(
    batch: List[Claim],
    parsed: list,
    verified: List[Claim],
    rejected: List[Claim],
    unverified: List[Claim],
) -> None:
    """Apply parsed LLM results to claims, defaulting uncovered claims to UNVERIFIED."""
    # Index parsed results by claim_index (or position)
    result_map: dict = {}
    for item in parsed:
        if isinstance(item, dict):
            idx = item.get("claim_index")
            if idx is None:
                # Fall back to list position
                idx = len(result_map)
            result_map[int(idx)] = item

    for i, claim in enumerate(batch):
        item = result_map.get(i)
        c = claim.copy()

        if item:
            status_str = (item.get("status") or "unverified").lower().strip()
            if status_str not in ("verified", "unverified", "rejected"):
                status_str = "unverified"
            c.status = ClaimStatus(status_str)

            raw_conf = float(item.get("confidence") or 0.0)
            # Default confidence rules
            if c.status == ClaimStatus.VERIFIED and raw_conf < 0.5:
                raw_conf = 0.7
            elif c.status == ClaimStatus.UNVERIFIED and raw_conf == 0.0:
                raw_conf = 0.5
            c.confidence = min(1.0, max(0.0, raw_conf))

            c.supporting_evidence = (item.get("supporting_evidence") or "")[:300] or None
            note = item.get("verification_note") or item.get("rejection_reason") or ""
            c.rejection_reason = (note[:200]) or None
        else:
            # LLM did not cover this claim — mark UNVERIFIED
            c.status = ClaimStatus.UNVERIFIED
            c.confidence = 0.5
            c.rejection_reason = "Not covered by LLM response; defaulted to unverified."

        if c.status == ClaimStatus.VERIFIED:
            verified.append(c)
        elif c.status == ClaimStatus.REJECTED:
            rejected.append(c)
        else:
            unverified.append(c)


def fact_verification_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Fact Verification Agent.

    Verifies claims in batches. Never drops claims — unprocessed ones
    default to UNVERIFIED. Populates state.verification_result.

    Args:
        state: Current BriefingState.
    Returns:
        Updated BriefingState.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    if check_runaway(state):
        return state

    set_agent_status(state, "fact_verification", AgentStatus.RUNNING)
    add_trace(state, AGENT_NAME, "verifying", "Fact verification started")

    analysis  = state.get("analysis_result")
    research  = state.get("research_result")
    topic     = state.get("topic", "competitive intelligence")
    meta      = state.get("run_metadata")

    # --- Handle zero claims --------------------------------------------------
    claims = (analysis.extracted_claims if analysis else []) or []
    if not claims:
        logger.warning(
            "[fact_verification] 0 claims received — creating synthetic placeholders for topic='%s'",
            topic,
        )
        add_trace(state, AGENT_NAME, "synthetic", "No claims found — injecting synthetic placeholders")
        synthetic = [
            Claim(
                text=f"Competitive landscape analysis for {topic}",
                category="market", status=ClaimStatus.UNVERIFIED, confidence=0.4,
                rejection_reason="No claims were extracted by the Analyst agent.",
            ),
            Claim(
                text=f"Market trends observed in the {topic} sector",
                category="market", status=ClaimStatus.UNVERIFIED, confidence=0.4,
                rejection_reason="No claims were extracted by the Analyst agent.",
            ),
            Claim(
                text=f"Strategic opportunities identified in {topic}",
                category="market", status=ClaimStatus.UNVERIFIED, confidence=0.4,
                rejection_reason="No claims were extracted by the Analyst agent.",
            ),
        ]
        state["verification_result"] = VerificationResult(
            unverified_claims=synthetic,
            citation_coverage=0.0,
            overall_confidence=0.4,
            verification_notes=["Analyst produced 0 claims; synthetic placeholders created."],
        )
        state["workflow_phase"] = WorkflowPhase.WRITING.value
        set_agent_status(state, "fact_verification", AgentStatus.COMPLETED)
        return state

    # --- Build source texts --------------------------------------------------
    source_texts = _build_source_texts(research)
    logger.info(
        "[fact_verification] Verifying %d claims against %d source excerpts",
        len(claims), len(source_texts),
    )

    verified:   List[Claim] = []
    rejected:   List[Claim] = []
    unverified: List[Claim] = []

    # --- Process in batches --------------------------------------------------
    try:
        llm = get_llm()
        batches = [claims[i:i + _BATCH_SIZE] for i in range(0, len(claims), _BATCH_SIZE)]
        logger.info(
            "[fact_verification] Processing %d batches of up to %d claims each",
            len(batches), _BATCH_SIZE,
        )

        for b_idx, batch in enumerate(batches):
            try:
                prompt = _build_batch_prompt(batch, source_texts)
                response = call_with_retry(
                    lambda: llm.invoke([
                        SystemMessage(content=_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]),
                    max_retries=3,
                    base_delay=5.0,
                    label=f"verify-batch-{b_idx+1}",
                )
                if meta:
                    meta.tool_calls += 1

                parsed = _parse_verification_json(response.content, len(batch))

                if not parsed:
                    logger.warning(
                        "[fact_verification] Batch %d/%d: LLM returned unparseable response — "
                        "marking all %d claims UNVERIFIED",
                        b_idx + 1, len(batches), len(batch),
                    )
                    for c in batch:
                        cc = c.copy()
                        cc.status = ClaimStatus.UNVERIFIED
                        cc.confidence = 0.5
                        cc.rejection_reason = "Verification batch failed to parse; defaulted to unverified."
                        unverified.append(cc)
                    continue

                _apply_batch_results(batch, parsed, verified, rejected, unverified)

                logger.info(
                    "[fact_verification] Batch %d/%d done — verified=%d rejected=%d unverified=%d",
                    b_idx + 1, len(batches),
                    sum(1 for c in verified   if c in [x for x in batch]),
                    sum(1 for c in rejected   if c in [x for x in batch]),
                    sum(1 for c in unverified if c in [x for x in batch]),
                )

            except Exception as exc:
                logger.warning(
                    "[fact_verification] Batch %d/%d failed: %s — marking batch UNVERIFIED",
                    b_idx + 1, len(batches), exc,
                )
                for c in batch:
                    cc = c.copy()
                    cc.status = ClaimStatus.UNVERIFIED
                    cc.confidence = 0.5
                    cc.rejection_reason = f"Batch verification error: {str(exc)[:80]}"
                    unverified.append(cc)

    except Exception as exc:
        msg = f"Fact verification LLM setup failed: {exc}"
        add_error(state, msg)
        logger.error("[fact_verification] %s", msg, exc_info=True)
        # Carry all claims as UNVERIFIED
        for c in claims:
            cc = c.copy()
            cc.status = ClaimStatus.UNVERIFIED
            cc.confidence = 0.5
            unverified.append(cc)

    total = len(claims)
    coverage = round(len(verified) / total, 3) if total > 0 else 0.0
    all_confident = verified + unverified
    avg_conf = (
        sum(c.confidence for c in all_confident) / len(all_confident)
        if all_confident else 0.0
    )

    result = VerificationResult(
        verified_claims=verified,
        rejected_claims=rejected,
        unverified_claims=unverified,
        citation_coverage=coverage,
        overall_confidence=round(avg_conf, 3),
        verification_notes=[
            f"Verified: {len(verified)}/{total} claims",
            f"Unverified: {len(unverified)}/{total} claims "
            "(plausible but not directly evidenced in fetched text)",
            f"Rejected: {len(rejected)}/{total} claims "
            "(contradicted by sources)",
            f"Coverage: {coverage:.0%}",
            f"Average confidence: {avg_conf:.2f}",
        ],
    )
    state["verification_result"] = result

    if meta:
        meta.verified_claims = len(verified)
        meta.rejected_claims = len(rejected)

    state["workflow_phase"] = WorkflowPhase.WRITING.value
    elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)

    logger.info(
        "[fact_verification] === Done | verified=%d unverified=%d rejected=%d | "
        "coverage=%.0f%% | confidence=%.2f | %.1f ms ===",
        len(verified), len(unverified), len(rejected),
        coverage * 100, avg_conf, elapsed_ms,
    )

    add_trace(
        state, AGENT_NAME, "completed",
        f"Verification done: {len(verified)} verified, {len(unverified)} unverified, "
        f"{len(rejected)} rejected. Coverage: {coverage:.0%}",
        duration_ms=elapsed_ms,
        metadata={"verified": len(verified), "unverified": len(unverified),
                  "rejected": len(rejected), "coverage": coverage},
    )
    add_audit(
        state,
        event_type="fact_verification_complete",
        agent=AGENT_NAME,
        description=f"Coverage: {coverage:.0%}, confidence: {avg_conf:.2f}",
        data={"verified": len(verified), "unverified": len(unverified),
              "rejected": len(rejected), "coverage": coverage},
        severity="info" if coverage >= 0.3 else "warning",
    )
    set_agent_status(state, "fact_verification", AgentStatus.COMPLETED)
    return state
