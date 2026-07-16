"""
Analyst Agent – reads research sources and extracts structured competitive
intelligence using the LLM.

Key design decisions:
- Each source is processed individually so one bad article never blocks others.
- JSON parsing handles markdown fences, truncated JSON, and common LLM quirks.
- Falls back to sentence-level extraction from raw text if LLM returns empty.
- Detailed logging at every step: articles read, claims extracted, failures.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import (
    add_audit,
    add_error,
    add_trace,
    check_runaway,
    get_llm,
    increment_step,
    set_agent_status,
)
from agents.state import (
    AgentStatus,
    AnalysisResult,
    BriefingState,
    Claim,
    ClaimStatus,
    CompetitorProfile,
    WorkflowPhase,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "analyst_agent"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior competitive intelligence analyst at a top-tier strategy consulting firm.

Given a single research article, extract EVERY piece of business intelligence it contains.

Extract the following categories — leave a category as an empty list [] if nothing is found, but look hard:
- competitor_names: List every company name mentioned as a competitor or market player.
- pricing: Specific pricing data (dollar amounts, tier names, % changes, "free trial", "enterprise pricing").
- product_launches: Named product releases, features, or platform updates with dates if mentioned.
- ai_features: Specific AI/ML capabilities named in the article (copilots, automation, generative AI features).
- partnerships: Named partner + what they do together.
- acquisitions: Acquired company + rationale if stated.
- market_trends: Observable market-level facts with evidence (growth %, market size, adoption stats).
- funding: Funding rounds, amounts, investors, valuations.
- customer_adoption: Customer wins, churn stats, NPS data, user counts, enterprise adoption.
- business_risks: Specific risks named in the article (competitive threats, regulatory, pricing pressure).
- strategic_opportunities: Gaps or openings described in the article.

Rules:
- Extract ONLY facts explicitly stated in the article. No fabrication.
- Be SPECIFIC: include exact names, numbers, dates, and product names.
- Each finding is a short factual sentence (max 120 words).
- Confidence: 0.9 = direct quote with numbers; 0.7 = clearly stated; 0.5 = implied.
- source_url must be the exact URL provided.

Return ONLY valid JSON. No prose before or after.

JSON schema:
{
  "competitor_names": ["string"],
  "findings": [
    {
      "category": "pricing|product|ai_feature|partnership|acquisition|market|funding|customer|risk|opportunity",
      "claim": "specific factual sentence",
      "evidence": "exact quote or close paraphrase from article",
      "confidence": 0.0,
      "source_url": "string"
    }
  ]
}"""


def _build_source_prompt(topic: str, source_title: str, source_url: str, source_text: str) -> str:
    """Build a per-source extraction prompt."""
    # Trim text to stay within token budget (~3000 chars ≈ 750 tokens)
    text = source_text[:3000].strip()
    return f"""Topic: {topic}

Article Title: {source_title}
Article URL: {source_url}

Article Text:
{text}

Extract ALL competitive intelligence findings from this article. Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# JSON parsing — handles all common LLM output quirks
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from an LLM response.
    Handles: markdown fences (```json...```), leading/trailing prose,
    truncated JSON (attempts to close unclosed objects/arrays).
    """
    if not raw:
        return {}

    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        # Extract content between first ``` and last ```
        parts = text.split("```")
        # parts[1] is inside the first fence pair
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                text = candidate
                break

    # Find the outermost JSON object
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start != -1:
        text = text[start:]

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to repair truncated JSON by closing unclosed structures
    repaired = _repair_json(text)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    logger.debug("[analyst] JSON parse failed on: %s...", raw[:100])
    return {}


def _repair_json(text: str) -> str:
    """
    Attempt to close unclosed JSON by counting brackets.
    Only fixes truncation — does not fix malformed keys/values.
    """
    # Find the last complete object boundary
    # Truncate at last complete entry (after last }, or last "])
    last_good = max(text.rfind("},"), text.rfind("}]"), text.rfind('"}'))
    if last_good > 0:
        text = text[: last_good + 1]

    # Count and close unclosed brackets
    opens = text.count("{") - text.count("}")
    arr_opens = text.count("[") - text.count("]")
    suffix = ""
    for _ in range(max(0, arr_opens)):
        suffix += "]"
    for _ in range(max(0, opens)):
        suffix += "}"
    return text + suffix


# ---------------------------------------------------------------------------
# Per-source extraction
# ---------------------------------------------------------------------------

def _extract_from_source(
    llm,
    topic: str,
    source_title: str,
    source_url: str,
    source_text: str,
    source_id: str,
) -> Tuple[List[Claim], List[str]]:
    """
    Call the LLM to extract findings from a single source.

    Returns (claims, competitor_names).
    Errors are caught and logged — never propagated.
    """
    claims: List[Claim] = []
    competitor_names: List[str] = []

    if not source_text or len(source_text.strip()) < 100:
        logger.info(
            "[analyst] Skipping %s — insufficient text (%d chars)",
            source_url[:60], len(source_text or ""),
        )
        return claims, competitor_names

    try:
        prompt = _build_source_prompt(topic, source_title, source_url, source_text)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content or ""
        parsed = _parse_llm_json(raw)

        if not parsed:
            logger.info(
                "[analyst] LLM returned empty/unparseable JSON for %s — raw: %s...",
                source_url[:60], raw[:80],
            )
            return claims, competitor_names

        competitor_names = [
            str(n) for n in parsed.get("competitor_names", []) if n
        ]

        for f in parsed.get("findings", []):
            claim_text = (f.get("claim") or "").strip()
            if not claim_text or len(claim_text) < 15:
                continue
            category = (f.get("category") or "market").lower()
            # Normalise category
            category_map = {
                "ai_feature": "technology",
                "ai_features": "technology",
                "funding": "market",
                "customer": "customer",
                "opportunity": "market",
            }
            category = category_map.get(category, category)
            if category not in (
                "pricing", "product", "partnership", "acquisition",
                "market", "technology", "customer", "risk",
            ):
                category = "market"

            claims.append(
                Claim(
                    text=claim_text[:300],
                    category=category,
                    source_ids=[source_id],
                    status=ClaimStatus.UNVERIFIED,
                    confidence=min(1.0, max(0.0, float(f.get("confidence", 0.6)))),
                    supporting_evidence=(f.get("evidence") or "")[:300] or None,
                )
            )

        logger.info(
            "[analyst] Source '%s' → %d claims, %d competitors",
            source_title[:50], len(claims), len(competitor_names),
        )

    except Exception as exc:
        logger.warning(
            "[analyst] LLM extraction failed for %s: %s",
            source_url[:60], exc,
        )

    return claims, competitor_names


# ---------------------------------------------------------------------------
# Fallback: sentence-level extraction from raw text
# ---------------------------------------------------------------------------

def _fallback_extract(sources_data: List[Dict]) -> Tuple[List[Claim], List[str]]:
    """
    When every LLM call fails, extract sentences from raw source text
    as UNVERIFIED claims so the Writer still has content.
    """
    claims: List[Claim] = []
    competitors: List[str] = []

    for item in sources_data:
        text = item.get("text", "")
        if not text:
            continue
        src_id = item.get("id", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
        for sent in sentences[:3]:
            claims.append(
                Claim(
                    text=sent[:250],
                    category="market",
                    source_ids=[src_id] if src_id else [],
                    status=ClaimStatus.UNVERIFIED,
                    confidence=0.35,
                )
            )
        if len(claims) >= 20:
            break

    logger.info(
        "[analyst] Fallback sentence extraction: %d claims from raw text", len(claims)
    )
    return claims, competitors


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def analyst_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Analyst Agent.

    Processes each research source individually with error isolation.
    Aggregates all findings into a single AnalysisResult.
    Never fails the workflow — returns whatever was extracted.

    Args:
        state: Current BriefingState.
    Returns:
        Updated BriefingState with analysis_result populated.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    if check_runaway(state):
        return state

    set_agent_status(state, "analyst", AgentStatus.RUNNING)
    topic = state.get("topic", "")
    add_trace(state, AGENT_NAME, "analyzing", f"Analyst started for: '{topic}'")

    # --- Collect source data -------------------------------------------------
    research = state.get("research_result")
    sources_data: List[Dict] = []
    if research:
        for src in research.sources:
            text = src.raw_content or src.summary or ""
            if text and len(text.strip()) >= 80:
                sources_data.append({
                    "id": src.id,
                    "title": src.title or src.url,
                    "url": src.url,
                    "text": text,
                })

    logger.info(
        "[analyst] === Starting extraction | topic='%s' | %d sources with text ===",
        topic, len(sources_data),
    )

    if not sources_data:
        msg = "No source text available for analysis — all sources lack content."
        add_error(state, msg)
        add_trace(state, AGENT_NAME, "failed", msg)
        set_agent_status(state, "analyst", AgentStatus.FAILED)
        state["analysis_result"] = AnalysisResult()
        logger.warning("[analyst] %s", msg)
        return state

    # --- Per-source LLM extraction -------------------------------------------
    all_claims: List[Claim] = []
    all_competitor_names: List[str] = []
    llm_success_count = 0
    llm_fail_count = 0

    try:
        llm = get_llm()
        meta = state.get("run_metadata")

        for item in sources_data:
            claims, comp_names = _extract_from_source(
                llm=llm,
                topic=topic,
                source_title=item["title"],
                source_url=item["url"],
                source_text=item["text"],
                source_id=item["id"],
            )
            if meta:
                meta.tool_calls += 1

            if claims:
                llm_success_count += 1
                all_claims.extend(claims)
                all_competitor_names.extend(comp_names)
            else:
                llm_fail_count += 1
                logger.info(
                    "[analyst] No claims extracted from '%s' (url: %s)",
                    item["title"][:50], item["url"][:60],
                )

        logger.info(
            "[analyst] LLM extraction done: %d sources succeeded, %d returned 0 claims, "
            "%d total claims, %d unique competitors",
            llm_success_count, llm_fail_count,
            len(all_claims), len(set(all_competitor_names)),
        )

    except Exception as exc:
        msg = f"Analyst LLM setup failed: {exc}"
        add_error(state, msg)
        logger.error("[analyst] %s", msg, exc_info=True)

    # --- Fallback if LLM produced nothing ------------------------------------
    if not all_claims:
        logger.warning(
            "[analyst] LLM produced 0 claims — running sentence-level fallback"
        )
        all_claims, all_competitor_names = _fallback_extract(sources_data)

    # --- Deduplicate claims by text fingerprint ------------------------------
    seen_fps: set = set()
    unique_claims: List[Claim] = []
    for c in all_claims:
        fp = c.text[:60].lower().strip()
        if fp not in seen_fps:
            seen_fps.add(fp)
            unique_claims.append(c)
    if len(unique_claims) < len(all_claims):
        logger.info(
            "[analyst] Deduplication: %d → %d claims",
            len(all_claims), len(unique_claims),
        )
    all_claims = unique_claims

    # --- Build competitor profiles from aggregated data ----------------------
    profiles: List[CompetitorProfile] = []
    seen_competitors: set = set()
    for name in all_competitor_names:
        if not name or name.lower() in seen_competitors:
            continue
        seen_competitors.add(name.lower())
        # Collect claims for this competitor
        comp_pricing = [c.text for c in all_claims if c.category == "pricing" and name.lower() in c.text.lower()]
        comp_products = [c.text for c in all_claims if c.category == "product" and name.lower() in c.text.lower()]
        comp_risks = [c.text for c in all_claims if c.category == "risk" and name.lower() in c.text.lower()]
        comp_partnerships = [c.text for c in all_claims if c.category == "partnership" and name.lower() in c.text.lower()]
        comp_acquisitions = [c.text for c in all_claims if c.category == "acquisition" and name.lower() in c.text.lower()]
        comp_advantages = [c.text for c in all_claims if c.category == "technology" and name.lower() in c.text.lower()]
        profiles.append(
            CompetitorProfile(
                name=name,
                pricing_changes=comp_pricing[:4],
                product_launches=comp_products[:4],
                partnerships=comp_partnerships[:3],
                acquisitions=comp_acquisitions[:3],
                competitive_advantages=comp_advantages[:3],
                business_risks=comp_risks[:3],
            )
        )

    # --- Categorise signals / trends from claims -----------------------------
    market_signals = list(dict.fromkeys(
        c.text for c in all_claims if c.category == "market"
    ))[:12]
    technology_trends = list(dict.fromkeys(
        c.text for c in all_claims if c.category == "technology"
    ))[:8]
    customer_trends = list(dict.fromkeys(
        c.text for c in all_claims if c.category == "customer"
    ))[:8]
    market_movements = list(dict.fromkeys(
        c.text for c in all_claims
        if c.category in ("acquisition", "funding")
    ))[:8]

    result = AnalysisResult(
        competitor_profiles=profiles,
        extracted_claims=all_claims,
        market_signals=market_signals,
        technology_trends=technology_trends,
        customer_trends=customer_trends,
        market_movements=market_movements,
    )
    state["analysis_result"] = result

    meta = state.get("run_metadata")
    if meta:
        meta.total_claims = len(all_claims)

    state["workflow_phase"] = WorkflowPhase.VERIFYING.value
    elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)

    status_str = AgentStatus.COMPLETED if all_claims else AgentStatus.FAILED
    set_agent_status(state, "analyst", status_str)

    add_trace(
        state, AGENT_NAME, "completed",
        f"Extracted {len(all_claims)} claims from {llm_success_count}/{len(sources_data)} sources. "
        f"{len(profiles)} competitor profiles.",
        duration_ms=elapsed_ms,
        metadata={
            "total_claims": len(all_claims),
            "sources_processed": len(sources_data),
            "llm_success": llm_success_count,
            "llm_fail": llm_fail_count,
            "competitors": len(profiles),
        },
    )
    add_audit(
        state,
        event_type="analysis_complete",
        agent=AGENT_NAME,
        description=f"Extracted {len(all_claims)} claims and {len(profiles)} competitor profiles",
        data={"claims": len(all_claims), "competitors": len(profiles)},
    )
    logger.info(
        "[analyst] === Done | %d claims | %d competitors | %d market signals | %.1f ms ===",
        len(all_claims), len(profiles), len(market_signals), elapsed_ms,
    )
    return state
