"""
Writer Agent – generates the professional competitive intelligence report
sections using the LLM, then assembles the complete FinalReport.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional

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
    VerificationResult,
    WorkflowPhase,
)
from tools.citation_generator import generate_citations
from tools.report_generator import assemble_final_report

logger = logging.getLogger(__name__)
AGENT_NAME = "writer_agent"

_SYSTEM_PROMPT = """You are a Principal Analyst at a top-tier competitive intelligence firm, writing for a VP of Strategy and the C-suite. Your reports are read by executives who make million-dollar decisions based on them.

Writing standards:
- Style: Professional consulting (Gartner, McKinsey, Deloitte). Direct, assertive, analytical.
- Never use vague filler like "it is important to note" or "there are several factors." State the finding directly.
- Every paragraph must deliver a specific insight, not just describe what happened — explain WHY it matters competitively.
- Compare competitors explicitly (e.g., "Salesforce's Einstein GPT undercuts HubSpot's AI tier by 30% at the enterprise level").
- Quantify wherever the data supports it. Use specific numbers, percentages, product names, and dates from the sources.
- Rank findings by strategic importance — the most critical insight comes first.
- Distinguish CLEARLY between verified findings and unverified findings:
  * Verified findings: state confidently with citation context
  * Unverified findings: use "According to unconfirmed reports..." or "Sources suggest, though unverified, that..." — NEVER omit them entirely
- If a section has limited data, write what IS known and state the confidence level. NEVER write "No analysis could be generated."
- Use bullet points for enumerated lists. Use prose for analysis and synthesis.
- Include a competitor comparison table (markdown format) whenever 2+ competitors have comparable data points.
- Actionable recommendations must be specific: "Consider launching a mid-market pricing tier at $X/seat to counter Y's recent discount" not "Consider improving pricing strategy."
- Maximum section length: 400 words or 8 bullet points. Be concise."""


# ── Context helpers ───────────────────────────────────────────────────────────


def _claims_text(claims: List[Claim], label: str = "verified") -> str:
    """Format a list of claims as a readable bullet list with status label."""
    if not claims:
        return f"No {label} claims available."
    lines: List[str] = []
    for c in claims[:35]:
        marker = "" if label == "verified" else " [Unverified]"
        evidence = ""
        if c.supporting_evidence and label == "verified":
            evidence = f' — Evidence: "{c.supporting_evidence[:100]}"'
        lines.append(
            f"- [{c.category}]{marker} {c.text} (confidence: {c.confidence:.0%}){evidence}"
        )
    return "\n".join(lines)


def _competitor_profiles_text(analysis: Optional[AnalysisResult]) -> str:
    """Summarise competitor profiles into a detailed, structured string for the LLM."""
    if not analysis or not analysis.competitor_profiles:
        return "No competitor profiles available."
    parts: List[str] = []
    for p in analysis.competitor_profiles[:8]:
        items: List[str] = []
        if hasattr(p, "market_position") and p.market_position:
            items.append(f"Position: {p.market_position}")
        if hasattr(p, "pricing_model") and p.pricing_model:
            items.append(f"Pricing model: {p.pricing_model}")
        if p.pricing_changes:
            items.append("Pricing: " + "; ".join(p.pricing_changes[:4]))
        if p.product_launches:
            items.append("Product launches: " + "; ".join(p.product_launches[:4]))
        if hasattr(p, "ai_capabilities") and p.ai_capabilities:
            items.append("AI capabilities: " + "; ".join(p.ai_capabilities[:4]))
        if p.partnerships:
            items.append("Partnerships: " + "; ".join(p.partnerships[:3]))
        if p.acquisitions:
            items.append("Acquisitions: " + "; ".join(p.acquisitions[:3]))
        if p.competitive_advantages:
            items.append("Advantages: " + "; ".join(p.competitive_advantages[:3]))
        if p.business_risks:
            items.append("Risks: " + "; ".join(p.business_risks[:3]))
        if hasattr(p, "recent_news") and p.recent_news:
            items.append("Recent news: " + "; ".join(p.recent_news[:2]))
        detail = "\n    ".join(items) if items else "Limited data available."
        website = f" ({p.website})" if p.website else ""
        parts.append(f"- {p.name}{website}:\n    {detail}")
    return "\n".join(parts)


def _source_summaries_text(research) -> str:
    """Return the top-8 source summaries with URLs for citation context."""
    if not research or not research.sources:
        return "No source summaries available."
    lines: List[str] = []
    for src in research.sources[:8]:
        summary = (src.summary or "").strip()
        if summary:
            lines.append(f"- [{src.title}] ({src.url}) {summary[:400]}")
    return "\n".join(lines) if lines else "No source summaries available."


def _build_context(
    research,
    analysis: Optional[AnalysisResult],
    verification: Optional[VerificationResult],
) -> str:
    """Assemble a rich context string for the LLM from all pipeline outputs."""

    # Verified claims — sorted by confidence descending
    verified = sorted(
        verification.verified_claims if verification else [],
        key=lambda c: c.confidence,
        reverse=True,
    )
    verified_text = _claims_text(verified, "verified")

    # Unverified claims
    unverified = verification.unverified_claims if verification else []
    unverified_text = _claims_text(unverified, "unverified")

    # Competitor profiles — full detail
    competitors_text = _competitor_profiles_text(analysis)

    # Market signals & trends
    signals: List[str] = []
    tech_trends: List[str] = []
    customer_trends: List[str] = []
    market_movements: List[str] = []
    if analysis:
        signals = analysis.market_signals[:12]
        tech_trends = analysis.technology_trends[:8]
        customer_trends = analysis.customer_trends[:8]
        market_movements = getattr(analysis, "market_movements", [])[:8]
    signals_text = "\n".join(f"- {s}" for s in signals) or "No market signals identified."
    tech_text = "\n".join(f"- {t}" for t in tech_trends) or "No technology trends identified."
    customer_text = "\n".join(f"- {t}" for t in customer_trends) or "No customer trends identified."
    movements_text = "\n".join(f"- {m}" for m in market_movements) or "No market movements identified."

    # Source summaries — top 8
    sources_text = _source_summaries_text(research)

    # High-importance claims flagged separately
    high_priority = [
        c for c in (verified + unverified)
        if getattr(c, "strategic_importance", "medium") == "high" or c.confidence >= 0.8
    ][:10]
    high_priority_text = _claims_text(high_priority, "high-priority") if high_priority else "None flagged."

    return (
        f"=== HIGH-PRIORITY FINDINGS (use these first) ===\n{high_priority_text}\n\n"
        f"=== VERIFIED CLAIMS ===\n{verified_text}\n\n"
        f"=== UNVERIFIED CLAIMS (include with 'unconfirmed reports' language) ===\n{unverified_text}\n\n"
        f"=== COMPETITOR PROFILES ===\n{competitors_text}\n\n"
        f"=== MARKET SIGNALS ===\n{signals_text}\n\n"
        f"=== TECHNOLOGY TRENDS ===\n{tech_text}\n\n"
        f"=== CUSTOMER TRENDS ===\n{customer_text}\n\n"
        f"=== MARKET MOVEMENTS (M&A / Funding / Regulatory) ===\n{movements_text}\n\n"
        f"=== SOURCE SUMMARIES (TOP 8) ===\n{sources_text}"
    )


# ── LLM section writer ────────────────────────────────────────────────────────


def _write_section(
    llm,
    section_name: str,
    topic: str,
    context: str,
    instruction: str,
    fallback: str = "",
) -> str:
    """
    Call the LLM to write a specific report section.

    If the LLM call raises any exception, return *fallback*.
    If the LLM returns a placeholder-like response (empty, too short, or contains
    known fallback phrases), force a second attempt with a stricter prompt.
    """
    _PLACEHOLDER_PHRASES = (
        "no analysis could be generated",
        "limited data was retrieved",
        "could not be generated",
        "no data found",
        "unable to generate",
        "no information available",
    )

    def _is_placeholder(text: str) -> bool:
        t = text.lower().strip()
        return len(t) < 80 or any(p in t for p in _PLACEHOLDER_PHRASES)

    def _call(instruction_override: str) -> str:
        user_msg = (
            f"Topic: {topic}\n\n"
            f"Context:\n{context[:8000]}\n\n"
            f"Task: Write the '{section_name}' section. {instruction_override}\n"
            f"Keep it to 4-6 paragraphs or a focused bullet list. Be specific and analytical."
        )
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        return response.content.strip()

    try:
        result = _call(instruction)

        # If the LLM returns a useless placeholder, try once more with stricter prompt
        if _is_placeholder(result):
            logger.info(
                "[writer] Section '%s' returned placeholder on first attempt — retrying with stricter prompt",
                section_name,
            )
            stricter = (
                f"{instruction} "
                "IMPORTANT: You MUST write substantive content. "
                "The context above contains real research findings — use them. "
                "Do NOT write 'no analysis could be generated' or similar. "
                "If data is limited, summarise what IS known and state confidence level."
            )
            result = _call(stricter)

        if _is_placeholder(result):
            logger.warning(
                "[writer] Section '%s' still returned placeholder after retry — using fallback",
                section_name,
            )
            return fallback or f"Analysis for '{section_name}' could not be completed. Please re-run with additional sources."

        return result

    except Exception as exc:
        logger.warning(
            "[writer] LLM call failed for section '%s': %s — using fallback.", section_name, exc
        )
        return fallback or f"_{section_name} could not be generated due to an error._"


# ── Writer node ───────────────────────────────────────────────────────────────


def writer_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Writer Agent.

    Generates all report sections using the LLM, assembles citations,
    and populates state.final_report.

    Resilient design:
    - Proceeds even if analysis or verification results are None/empty (uses defaults).
    - Includes unverified claims with [Unverified] marker instead of hiding them.
    - Falls back gracefully if any individual LLM section call fails.
    - Writes 6 LLM sections: executive_summary, competitor_pricing, product_updates,
      market_signals, business_risks, strategic_recommendations, AND opportunities.

    Args:
        state: Current BriefingState.

    Returns:
        Updated BriefingState.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    if check_runaway(state):
        return state

    set_agent_status(state, "writer", AgentStatus.RUNNING)
    topic = state.get("topic", "")
    add_trace(state, AGENT_NAME, "writing", f"Writer agent started for topic: '{topic}'")

    research = state.get("research_result")
    analysis: Optional[AnalysisResult] = state.get("analysis_result")
    verification: Optional[VerificationResult] = state.get("verification_result")

    # Warn but continue – do NOT fail if upstream results are missing
    if not analysis:
        add_trace(state, AGENT_NAME, "warning", "No analysis result available; writer will use defaults.")
    if not verification:
        add_trace(state, AGENT_NAME, "warning", "No verification result available; writer will use defaults.")

    try:
        llm = get_llm(temperature=0.2)
        meta = state.get("run_metadata")

        # ── Build rich context for all LLM sections ────────────────────────────
        full_context = _build_context(research, analysis, verification)

        # ── Helper to write + track tool call ─────────────────────────────────
        def write(section_name: str, instruction: str, fallback: str = "") -> str:
            text = _write_section(llm, section_name, topic, full_context, instruction, fallback)
            if meta:
                meta.tool_calls += 1
            return text

        # ── Section 1: Executive Summary ───────────────────────────────────────
        executive_summary = write(
            "Executive Summary",
            "Write a 3-paragraph executive summary that a VP of Strategy can act on immediately. "
            "Paragraph 1: The single most important competitive development and its strategic implication. "
            "Paragraph 2: The 2-3 most significant competitor moves (with names, specific actions, and why they matter). "
            "Paragraph 3: The top strategic recommendation and the most critical risk to monitor. "
            "Use specific competitor names, product names, and numbers. "
            "Mark any unconfirmed findings as 'According to unconfirmed reports...' "
            "Do NOT write generic filler. If data is limited, state that clearly and summarize what IS known.",
            fallback=(
                f"The research pipeline retrieved {len(research.sources) if research else 0} sources on the topic "
                f"'{topic}'. The LLM analysis was unable to generate a full summary. "
                "Key sources retrieved include: "
                + (", ".join(s.title[:50] for s in (research.sources[:3] if research else [])) or "none")
                + ". Please re-run with Tavily enabled for richer data, or try a more specific topic."
            ),
        )

        # ── Section 2: Competitor Pricing ──────────────────────────────────────
        competitor_pricing = write(
            "Competitor Pricing Analysis",
            "Analyze pricing strategies across ALL identified competitors. "
            "1. Start with a comparison table (markdown) if 2+ competitors have pricing data: columns = Competitor | Pricing Model | Key Tier | Price Point | Recent Change. "
            "2. After the table, analyze the strategic implications: who is discounting aggressively, who is moving upmarket, what pricing pressure this creates. "
            "3. Identify any pricing anomalies or opportunities (e.g., 'No competitor offers a $X/month mid-market tier — this is a gap'). "
            "4. Mark any unconfirmed pricing data as 'Unconfirmed:' "
            "If no specific pricing data was found, state: 'Specific pricing data was not retrievable from available sources. Based on public positioning: [summarize what is known about pricing strategy].'",
            fallback=(
                "Specific pricing data could not be extracted from the available sources. "
                "The research sources retrieved did not contain detailed pricing information. "
                "Recommended next steps: search directly on competitor pricing pages, or enable Tavily for real-time pricing data."
            ),
        )

        # ── Section 3: Product Updates ─────────────────────────────────────────
        product_updates = write(
            "Competitor Product & AI Capability Updates",
            "Summarize recent product launches, AI/ML feature releases, platform updates, and technology bets per competitor. "
            "1. Organize by competitor (use subheadings or bold competitor names). "
            "2. For each product launch or AI capability: name it specifically, state when it launched (if known), and explain its competitive significance. "
            "3. Identify capability gaps: 'Competitor X has launched Y — we do not currently have an equivalent.' "
            "4. Note any partnerships or integrations that expand competitive reach. "
            "5. Highlight AI-specific capabilities (copilots, automation, generative AI features) as these are currently the highest-stakes competitive differentiator. "
            "Mark unconfirmed items as 'Unconfirmed:' "
            "Never write 'No analysis could be generated' — if data is thin, summarize what IS available.",
            fallback=(
                "Product update data was limited in the available source articles. "
                "The sources retrieved cover the general market landscape but did not contain specific product launch announcements. "
                "For real-time product updates, monitor competitor blogs, press release feeds, and product changelogs directly."
            ),
        )

        # ── Section 4: Market Signals ──────────────────────────────────────────
        market_signals = write(
            "Market Signals & Trends",
            "Synthesize the key market signals, technology trends, and customer behavior shifts. "
            "1. Lead with the 2-3 strongest signals (those with the most evidence). "
            "2. For each signal: state the observation, explain what is driving it, and describe the strategic implication. "
            "3. Distinguish between: (a) confirmed trends with multiple sources, (b) emerging signals with limited evidence. "
            "4. Include AI/automation adoption trends specifically — these are transforming competitive dynamics. "
            "5. Note any customer behavior shifts (e.g., preference for vertical-specific solutions, demand for AI-native tools). "
            "Mark any unconfirmed signals. "
            "If signals are limited, state the confidence level and recommend additional research areas.",
            fallback=(
                "Market signal analysis was constrained by the content retrievable from available sources. "
                "The research pipeline identified relevant sources but was unable to extract detailed market signals from their text content. "
                "General market context: the topic researched is an active, competitive space. Re-run with Tavily for richer market data."
            ),
        )

        # ── Section 5: Business Risks ──────────────────────────────────────────
        business_risks = write(
            "Business Risks",
            "Identify and rank the top business risks from the competitive landscape. "
            "Format: use a numbered list ranked from highest to lowest severity. "
            "For each risk: (1) name the risk, (2) identify which competitor or trend creates it, "
            "(3) estimate severity (High/Medium/Low) with brief justification, "
            "(4) suggest a mitigation approach. "
            "Categories to consider: pricing pressure, AI capability gap, talent competition, "
            "market share loss, regulatory risk, platform lock-in by competitor ecosystem. "
            "Mark any unconfirmed risks as 'Unconfirmed risk:' "
            "Be specific — 'risk of losing enterprise customers to Salesforce's Einstein GPT suite' is better than 'risk of losing customers to competitors.'",
            fallback=(
                "Risk identification was constrained by available source data. "
                "General risks for this competitive landscape typically include: pricing pressure from incumbents, "
                "AI capability gaps as competitors embed generative AI features, and market consolidation through M&A. "
                "A full risk assessment requires richer source data — enable Tavily or add domain-specific sources."
            ),
        )

        # ── Section 6: Strategic Recommendations ──────────────────────────────
        strategic_recommendations = write(
            "Strategic Recommendations",
            "Provide 5-7 specific, evidence-based strategic recommendations ranked by urgency. "
            "Format each recommendation as: "
            "**[Priority: High/Medium/Low] Recommendation title** "
            "Rationale: Why this is recommended, based on the specific competitive evidence above. "
            "Action: The specific action to take (not vague). "
            "Timeline: Immediate (0-30 days), Near-term (1-3 months), or Strategic (3-12 months). "
            "Base ALL recommendations on evidence from the competitive data above. "
            "Mark any recommendations based on unconfirmed data.",
            fallback=(
                "Strategic recommendations require validated competitive data. "
                "Based on the research conducted, the following general recommendations apply: "
                "1. Conduct a deeper competitive pricing audit using direct competitor websites. "
                "2. Monitor competitor AI feature announcements via press releases and product blogs. "
                "3. Re-run this analysis with Tavily enabled for significantly richer intelligence."
            ),
        )

        # ── Section 7: Opportunities ──────────────────────────────────────────
        opportunities = write(
            "Market & Competitive Opportunities",
            "Identify 4-6 concrete, evidence-based opportunities. "
            "For each opportunity: "
            "(1) State the opportunity specifically (what gap, segment, or capability is underserved). "
            "(2) Cite the evidence (which competitor weakness, which market signal, which customer trend creates it). "
            "(3) Estimate the size or urgency of the opportunity if data supports it. "
            "(4) Note any first-mover advantage window. "
            "Opportunity types to consider: pricing gaps, geographic expansion, underserved customer segment, "
            "AI capability gap in competitor portfolio, partnership opportunity, acquisition target. "
            "Mark speculative opportunities (limited evidence) as 'Potential opportunity (unconfirmed):'",
            fallback=(
                "Opportunity analysis requires validated competitive data. "
                "Potential opportunity areas in this competitive landscape typically include: "
                "underserved mid-market segments, AI-native product differentiation, and vertical-specific solutions. "
                "A detailed opportunity analysis requires richer source data from additional research runs."
            ),
        )

        # ── Generate citations ────────────────────────────────────────────────
        sources = research.sources if research else []
        citations = generate_citations(sources)

        # ── Assemble final report ─────────────────────────────────────────────
        from tools.audit_logger import format_audit_summary
        audit_entries = state.get("audit_log", [])
        audit_summary = format_audit_summary(audit_entries)

        if meta:
            meta.completed_at = datetime.utcnow().isoformat()
            try:
                elapsed_total = (
                    datetime.fromisoformat(meta.completed_at)
                    - datetime.fromisoformat(meta.started_at)
                ).total_seconds()
                meta.duration_seconds = round(elapsed_total, 1)
            except Exception:
                meta.duration_seconds = round(time.perf_counter() - start_ts, 1)
            meta.status = "completed"

        # citation_coverage / overall_confidence – safe defaults if verification missing
        citation_coverage = verification.citation_coverage if verification else 0.0
        overall_confidence = verification.overall_confidence if verification else 0.0

        final_report = assemble_final_report(
            topic=topic,
            executive_summary=executive_summary,
            competitor_pricing=competitor_pricing,
            product_updates=product_updates,
            market_signals=market_signals,
            business_risks=business_risks,
            strategic_recommendations=strategic_recommendations,
            opportunities=opportunities,
            analysis=analysis,
            verification=verification,
            citations=citations,
            run_metadata=meta,
            audit_summary=audit_summary,
        )

        state["final_report"] = final_report
        state["workflow_phase"] = WorkflowPhase.GOVERNANCE.value

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
        add_trace(
            state,
            AGENT_NAME,
            "completed",
            f"Report written. {final_report.word_count} words, "
            f"{len(citations)} citations. Coverage: {final_report.citation_coverage:.0%}",
            duration_ms=elapsed_ms,
            metadata={"word_count": final_report.word_count, "citations": len(citations)},
        )
        add_audit(
            state,
            event_type="report_written",
            agent=AGENT_NAME,
            description=(
                f"Report assembled: {final_report.word_count} words, {len(citations)} citations"
            ),
            data={"word_count": final_report.word_count, "citations": len(citations)},
        )
        set_agent_status(state, "writer", AgentStatus.COMPLETED)
        logger.info(
            "Writer complete: %d words, %d citations", final_report.word_count, len(citations)
        )

    except Exception as exc:
        msg = f"Writer agent failed: {exc}"
        add_error(state, msg)
        add_trace(state, AGENT_NAME, "failed", msg)
        set_agent_status(state, "writer", AgentStatus.FAILED)
        logger.error(msg, exc_info=True)

    return state
