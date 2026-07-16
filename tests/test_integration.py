"""
Integration tests for the Competitive Intelligence Briefing Crew.

These are real integration tests – they make actual network calls and LLM
calls so they require valid API keys in the environment.  They are designed
to be run manually / in CI when credentials are available.

Run with:
    pytest tests/test_integration.py -v

Skip network tests without keys:
    pytest tests/test_integration.py -v -m "not requires_key"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_llm_key() -> bool:
    return bool(
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )


# ---------------------------------------------------------------------------
# 1. Web search
# ---------------------------------------------------------------------------

class TestWebSearch:
    """Tests for tools/web_search.py."""

    def test_build_search_queries_returns_list(self):
        from tools.web_search import build_search_queries
        queries = build_search_queries("AI CRM software 2025")
        assert isinstance(queries, list)
        assert len(queries) >= 3
        assert all(isinstance(q, str) and len(q) > 5 for q in queries)

    def test_web_search_never_raises(self):
        """web_search must never raise an exception."""
        from tools.web_search import web_search
        try:
            results = web_search("nonexistent xyz topic 99999", max_results=3)
            assert isinstance(results, list)
        except Exception as exc:
            pytest.fail(f"web_search raised an exception: {exc}")

    def test_search_with_fallback_always_returns_results(self):
        """search_with_fallback must always return a non-empty list."""
        from tools.web_search import search_with_fallback
        results = search_with_fallback("AI CRM competitors 2025", max_results=5)
        assert isinstance(results, list)
        assert len(results) >= 1, "search_with_fallback must never return empty list"
        first = results[0]
        assert "url" in first
        assert "title" in first

    def test_web_search_returns_dict_with_required_keys(self):
        from tools.web_search import web_search
        results = web_search("OpenAI GPT competitors", max_results=3)
        for r in results:
            assert "title" in r
            assert "url" in r
            assert "snippet" in r


# ---------------------------------------------------------------------------
# 2. Article fetching
# ---------------------------------------------------------------------------

class TestArticleFetch:
    """Tests for tools/article_fetch.py."""

    def test_fetch_article_never_raises(self):
        from tools.article_fetch import fetch_article
        result = fetch_article("https://httpbin.org/html", timeout=10)
        assert isinstance(result, dict)
        assert "url" in result
        assert "text" in result
        assert "error" in result

    def test_fetch_article_extracts_text_from_real_page(self):
        from tools.article_fetch import fetch_article
        result = fetch_article("https://en.wikipedia.org/wiki/Artificial_intelligence", timeout=15, max_chars=2000)
        assert result.get("text") is not None
        assert len(result["text"]) > 100, "Should extract meaningful text"

    def test_fetch_article_handles_bad_url(self):
        from tools.article_fetch import fetch_article
        result = fetch_article("https://this-domain-does-not-exist-xyz.com/page", timeout=5)
        assert result["text"] is None or len(result.get("text") or "") == 0
        assert result["error"] is not None

    def test_concurrent_fetch_returns_all_results(self):
        from tools.article_fetch import concurrent_fetch
        urls = [
            "https://httpbin.org/html",
            "https://this-bad-url-xyz.invalid/page",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        ]
        results = concurrent_fetch(urls, max_workers=3, timeout=10)
        assert len(results) == len(urls), "Must return one result per input URL"
        for r in results:
            assert "url" in r


# ---------------------------------------------------------------------------
# 3. Research agent
# ---------------------------------------------------------------------------

class TestResearchAgent:
    """Tests for agents/research_agent.py."""

    def test_research_always_produces_sources(self):
        """Even with a niche topic, research must return >= 1 source."""
        from agents.state import create_initial_state
        from agents.research_agent import research_node

        state = create_initial_state(
            topic="quantum computing market competitive landscape 2025",
            max_sources=5,
        )
        result_state = research_node(state)
        research = result_state.get("research_result")
        assert research is not None
        assert len(research.sources) >= 1, "research_node must return at least 1 source"

    def test_research_deduplicates_urls(self):
        from agents.state import create_initial_state
        from agents.research_agent import research_node

        state = create_initial_state(topic="AI software tools 2025", max_sources=10)
        result_state = research_node(state)
        research = result_state.get("research_result")
        if research and research.sources:
            urls = [s.url for s in research.sources]
            assert len(urls) == len(set(urls)), "Duplicate URLs found in sources"

    def test_research_records_failed_urls(self):
        from agents.state import create_initial_state
        from agents.research_agent import research_node

        state = create_initial_state(topic="cloud computing vendors", max_sources=5)
        result_state = research_node(state)
        research = result_state.get("research_result")
        assert research is not None
        assert isinstance(research.failed_urls, list)


# ---------------------------------------------------------------------------
# 4. Analyst agent (requires LLM key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_llm_key(), reason="No LLM API key configured")
class TestAnalystAgent:
    """Tests for agents/analyst_agent.py."""

    def test_analyst_extracts_claims(self):
        from agents.state import (
            BriefingState, ResearchResult, SourceDocument, create_initial_state
        )
        from agents.analyst_agent import analyst_node

        state = create_initial_state(topic="Salesforce vs HubSpot CRM 2025", max_sources=3)
        state["research_result"] = ResearchResult(
            sources=[
                SourceDocument(
                    title="Salesforce Q4 2024 Earnings",
                    url="https://investor.salesforce.com/fake",
                    summary="Salesforce reported record revenue of $9.4B in Q4 2024.",
                    raw_content=(
                        "Salesforce reported record revenue of $9.4B in Q4 2024, up 11% YoY. "
                        "The company launched Agentforce, an AI platform for autonomous agents. "
                        "HubSpot grew ARR by 22% and expanded its free CRM tier."
                    ),
                    domain="investor.salesforce.com",
                    relevance_score=0.9,
                    is_trusted_domain=True,
                )
            ],
            total_sources=1,
        )

        result_state = analyst_node(state)
        analysis = result_state.get("analysis_result")
        assert analysis is not None
        assert len(analysis.extracted_claims) >= 1, "Should extract at least 1 claim"


# ---------------------------------------------------------------------------
# 5. Fact verification (requires LLM key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_llm_key(), reason="No LLM API key configured")
class TestFactVerification:
    """Tests for agents/fact_verification_agent.py."""

    def test_verifier_does_not_reject_all_claims(self):
        """Verifier must keep at least some claims as verified or unverified."""
        from agents.state import (
            AnalysisResult, BriefingState, Claim, ClaimStatus,
            ResearchResult, SourceDocument, create_initial_state
        )
        from agents.fact_verification_agent import fact_verification_node

        state = create_initial_state(topic="Salesforce CRM", max_sources=3)
        state["research_result"] = ResearchResult(
            sources=[
                SourceDocument(
                    title="Salesforce News",
                    url="https://salesforce.com/fake",
                    summary="Salesforce revenue $9.4B",
                    raw_content="Salesforce reported record revenue of $9.4B in Q4 2024.",
                    domain="salesforce.com",
                    relevance_score=0.9,
                    is_trusted_domain=True,
                )
            ],
            total_sources=1,
        )
        state["analysis_result"] = AnalysisResult(
            extracted_claims=[
                Claim(text="Salesforce reported record revenue", category="market", confidence=0.8),
                Claim(text="Salesforce is a CRM company", category="market", confidence=0.9),
                Claim(text="CRM market is growing", category="market", confidence=0.7),
            ]
        )

        result_state = fact_verification_node(state)
        verification = result_state.get("verification_result")
        assert verification is not None

        total_not_rejected = len(verification.verified_claims) + len(verification.unverified_claims)
        assert total_not_rejected >= 1, "At least 1 claim must be verified or unverified, not all rejected"

    def test_verifier_produces_synthetic_claims_when_empty(self):
        """When no claims exist, verifier produces synthetic ones."""
        from agents.state import AnalysisResult, create_initial_state
        from agents.fact_verification_agent import fact_verification_node

        state = create_initial_state(topic="AI tools market", max_sources=3)
        state["analysis_result"] = AnalysisResult(extracted_claims=[])

        result_state = fact_verification_node(state)
        verification = result_state.get("verification_result")
        assert verification is not None
        total = (
            len(verification.verified_claims)
            + len(verification.unverified_claims)
            + len(verification.rejected_claims)
        )
        assert total >= 1, "Should produce synthetic claims when analysis is empty"


# ---------------------------------------------------------------------------
# 6. Writer agent (requires LLM key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_llm_key(), reason="No LLM API key configured")
class TestWriterAgent:
    """Tests for agents/writer_agent.py."""

    def test_writer_generates_report(self):
        from agents.state import (
            AnalysisResult, BriefingState, Claim, ClaimStatus,
            ResearchResult, SourceDocument, VerificationResult,
            create_initial_state
        )
        from agents.writer_agent import writer_node

        state = create_initial_state(topic="Salesforce vs HubSpot", max_sources=3)
        state["research_result"] = ResearchResult(
            sources=[
                SourceDocument(
                    title="CRM Comparison 2025",
                    url="https://example.com/crm",
                    summary="Salesforce leads enterprise; HubSpot leads SMB.",
                    raw_content="Salesforce holds 23% market share. HubSpot grew 22% ARR in 2024.",
                    domain="example.com",
                    relevance_score=0.8,
                )
            ],
            total_sources=1,
        )
        verified_claim = Claim(
            text="Salesforce holds 23% market share",
            category="market",
            status=ClaimStatus.VERIFIED,
            confidence=0.85,
        )
        state["analysis_result"] = AnalysisResult(extracted_claims=[verified_claim])
        state["verification_result"] = VerificationResult(
            verified_claims=[verified_claim],
            citation_coverage=0.8,
            overall_confidence=0.75,
        )

        result_state = writer_node(state)
        report = result_state.get("final_report")
        assert report is not None, "Writer must produce a FinalReport"
        assert report.word_count > 100, "Report must have meaningful content"
        assert report.executive_summary, "Executive summary must be non-empty"
        assert report.markdown_content, "Markdown content must be non-empty"


# ---------------------------------------------------------------------------
# 7. Full workflow (requires LLM key, slow)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_llm_key(), reason="No LLM API key configured")
@pytest.mark.slow
class TestFullWorkflow:
    """End-to-end workflow tests."""

    def test_full_workflow_produces_report(self):
        """Run the complete workflow and verify a report is produced."""
        from graph.workflow import run_workflow

        state = run_workflow(
            topic="Salesforce CRM competitors 2025",
            max_steps=20,
            max_sources=5,
            rag_enabled=False,
            human_approved=True,
        )
        assert state is not None
        report = state.get("final_report")
        assert report is not None, "Full workflow must produce a final report"
        assert report.word_count > 200, "Report must have substantial content"
        assert len(report.references) >= 1, "Report must have at least 1 citation"

    def test_workflow_does_not_terminate_at_research(self):
        """Workflow must proceed past research even if sources are thin."""
        from graph.workflow import run_workflow

        state = run_workflow(
            topic="niche widget manufacturing competitive landscape",
            max_steps=20,
            max_sources=3,
            rag_enabled=False,
            human_approved=True,
        )
        phase = state.get("workflow_phase", "")
        assert phase in ("completed", "awaiting_approval"), (
            f"Workflow should reach completed/awaiting_approval, got: {phase}"
        )


# ---------------------------------------------------------------------------
# 8. Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    """Tests for governance/refusal_policy.py."""

    def test_refusal_not_triggered_for_valid_report(self):
        from agents.state import FinalReport, VerificationResult
        from governance.refusal_policy import apply_refusal_policy, evaluate_refusal
        from agents.state import GovernanceCheckResult

        report = FinalReport(
            title="Test Report",
            topic="AI market",
            executive_summary="The AI market is growing rapidly with key players competing.",
            competitor_pricing="Competitor A offers $99/month. Competitor B charges $149/month.",
            product_updates="Competitor A launched a new AI assistant feature in Q1 2025.",
            market_signals="Cloud AI adoption grew 40% YoY. Enterprise demand is accelerating.",
            business_risks="Market saturation and pricing pressure are increasing risks.",
            strategic_recommendations="Differentiate on enterprise features and support SLAs.",
            opportunities="SMB segment remains underpenetrated with high growth potential.",
            markdown_content="test",
        )
        verification = VerificationResult(
            citation_coverage=0.7,
            overall_confidence=0.65,
        )

        should_refuse, reasons = evaluate_refusal(report, verification)
        assert not should_refuse, f"Should NOT refuse a valid report, reasons: {reasons}"

    def test_refusal_triggered_on_hallucination_marker(self):
        from agents.state import FinalReport, VerificationResult
        from governance.refusal_policy import evaluate_refusal

        report = FinalReport(
            title="Bad Report",
            topic="AI market",
            executive_summary="I cannot provide information about this topic as an AI.",
            competitor_pricing="Some pricing.",
            product_updates="Some products.",
            market_signals="Some signals.",
            business_risks="Some risks.",
            strategic_recommendations="Some recommendations.",
            markdown_content="test",
        )
        verification = VerificationResult(citation_coverage=0.8, overall_confidence=0.8)

        should_refuse, reasons = evaluate_refusal(report, verification)
        assert should_refuse, "Should refuse a report containing hallucination markers"
