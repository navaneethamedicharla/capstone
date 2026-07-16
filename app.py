"""
Competitive Intelligence Briefing Crew – Streamlit Application.

Professional dashboard providing:
- Sidebar controls (topic, sources, steps, RAG upload)
- Live workflow progress and agent status
- Execution trace viewer
- Report viewer with section tabs
- Citation viewer
- Human approval gate
- Markdown and PDF download
- Audit log viewer
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="Competitive Intelligence Crew",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Lazy imports (defer heavy imports until after page config) ────────────────
@st.cache_resource
def _import_workflow():
    from graph.workflow import run_workflow
    return run_workflow


@st.cache_resource
def _import_exporters():
    from tools.markdown_export import export_markdown
    from tools.pdf_export import export_pdf
    return export_markdown, export_pdf


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background & base text ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e8eaf6;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] label {
    color: #e8eaf6;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    color: #ffffff;
}

/* ── Main content block ── */
[data-testid="block-container"] {
    background: #0f1117;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a237e;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #e8eaf6 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stTextArea textarea {
    background: #283593 !important;
    color: #ffffff !important;
    border: 1px solid #5c6bc0 !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #283593 !important;
    color: #e8eaf6 !important;
    border: 1px solid #5c6bc0 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #3949ab !important;
    color: #ffffff !important;
}

/* ── Primary run button ── */
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: #3949ab !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 10px;
    border-left: 4px solid #5c6bc0;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #e8eaf6 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    color: #9fa8da !important;
    font-weight: 600;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #5c6bc0;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1e2130;
    border: 1px solid #2c3155;
    border-radius: 8px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #c5cae9 !important;
    font-weight: 600;
}

/* ── Info / warning / error / success boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px;
}
[data-testid="stAlert"] p {
    color: #1a1a2e !important;
}

/* ── Code blocks ── */
[data-testid="stCode"] {
    background: #1e2130 !important;
}
[data-testid="stCode"] code {
    color: #a5d6a7 !important;
}

/* ── Divider ── */
hr {
    border-color: #2c3155;
}

/* ── Agent status badges ── */
.agent-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 3px;
    text-align: center;
    min-width: 100px;
}
.badge-pending   { background: #1a237e; color: #9fa8da; border: 1px solid #3949ab; }
.badge-running   { background: #e65100; color: #ffffff; border: 1px solid #ff6d00; }
.badge-completed { background: #1b5e20; color: #a5d6a7; border: 1px solid #2e7d32; }
.badge-failed    { background: #b71c1c; color: #ffcdd2; border: 1px solid #c62828; }
.badge-skipped   { background: #311b92; color: #d1c4e9; border: 1px solid #512da8; }

/* ── Report container ── */
.report-container {
    background: #1e2130;
    color: #e8eaf6;
    border-radius: 10px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    line-height: 1.8;
    border: 1px solid #2c3155;
}
.report-container h1,
.report-container h2,
.report-container h3 {
    color: #90caf9;
}
.report-container a {
    color: #64b5f6;
}

/* ── Trace rows ── */
.trace-row {
    font-family: monospace;
    font-size: 12px;
    padding: 4px 0;
    border-bottom: 1px solid #1e2130;
    color: #c5cae9;
}

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] p {
    color: #9fa8da !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #1a237e !important;
    color: #e8eaf6 !important;
    border: 1px solid #3949ab !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #283593 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all Streamlit session state keys."""
    defaults = {
        "workflow_state": None,
        "running": False,
        "run_complete": False,
        "approval_pending": False,
        "approved": None,
        "error": None,
        "progress": 0.0,
        "status_text": "Ready",
        "log_queue": queue.Queue(),
        "uploaded_kb": False,
        "last_run_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> Dict[str, Any]:
    """Render sidebar controls and return configuration dict."""
    with st.sidebar:
        st.markdown("## 🔍 Intel Crew")
        st.markdown("_Competitive Intelligence_")
        st.divider()

        # Topic input
        st.markdown("### 📋 Research Topic")
        topic = st.text_area(
            "Enter topic",
            placeholder="e.g. Artificial Intelligence CRM market 2025",
            height=100,
            label_visibility="collapsed",
        )

        # Quick topic buttons
        st.markdown("**Sample Topics**")
        sample_topics = [
            "AI CRM software market 2025",
            "Electric Vehicles competitors",
            "Cloud Computing AWS Azure GCP",
            "Cybersecurity vendors 2025",
            "Healthcare AI diagnostics",
        ]
        for sample in sample_topics:
            if st.button(sample, key=f"sample_{sample[:15]}", use_container_width=True):
                st.session_state["_sample_topic"] = sample
                st.rerun()

        # Apply sample topic if set
        if "_sample_topic" in st.session_state:
            topic = st.session_state.pop("_sample_topic")

        st.divider()
        st.markdown("### ⚙️ Configuration")

        max_sources = st.slider("Max Sources", min_value=3, max_value=20, value=8, step=1)
        max_steps = st.slider("Max Workflow Steps", min_value=5, max_value=30, value=20, step=5)

        rag_enabled = st.toggle("Enable RAG Knowledge Base", value=False)

        if rag_enabled:
            uploaded_files = st.file_uploader(
                "Upload documents (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt", "csv"],
                accept_multiple_files=True,
            )
            if uploaded_files and st.button("🗄️ Index Documents", use_container_width=True):
                _index_uploaded_documents(uploaded_files)
        else:
            uploaded_files = []

        st.divider()
        st.markdown("### 🔑 API Status")
        import os
        has_key = bool(
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
        )
        if has_key:
            st.success("✅ API Key configured")
        else:
            st.error("❌ No API key found")
            st.caption("Set OPENROUTER_API_KEY in .env")

        st.divider()
        # Run button
        run_clicked = st.button(
            "🚀 Run Intelligence Briefing",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["running"],
        )

    return {
        "topic": topic,
        "max_sources": max_sources,
        "max_steps": max_steps,
        "rag_enabled": rag_enabled,
        "run_clicked": run_clicked,
    }


def _index_uploaded_documents(files: list) -> None:
    """Index uploaded documents into the FAISS knowledge base."""
    from pathlib import Path
    import tempfile
    from rag.knowledge_base import build_knowledge_base

    with st.spinner("Indexing documents..."):
        tmp_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in files:
                dest = Path(tmpdir) / f.name
                dest.write_bytes(f.read())
                tmp_paths.append(dest)
            try:
                vs = build_knowledge_base(file_paths=tmp_paths, persist=True)
                if vs:
                    st.session_state["uploaded_kb"] = True
                    st.success(f"✅ Indexed {len(files)} document(s) into knowledge base")
                else:
                    st.warning("No content extracted from documents.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")


# ── Agent status display ───────────────────────────────────────────────────────
def render_agent_status(workflow_state: Optional[Dict]) -> None:
    """Render agent pipeline status badges."""
    agents = ["supervisor", "research", "analyst", "fact_verification", "writer", "governance"]
    agent_labels = {
        "supervisor": "🎯 Supervisor",
        "research": "🔎 Research",
        "analyst": "📊 Analyst",
        "fact_verification": "✅ Fact Check",
        "writer": "✍️ Writer",
        "governance": "🛡️ Governance",
    }

    tracker = None
    if workflow_state and "agent_status" in workflow_state:
        tracker = workflow_state["agent_status"]

    cols = st.columns(len(agents))
    for col, agent_key in zip(cols, agents):
        with col:
            status = "pending"
            if tracker and hasattr(tracker, agent_key):
                status = getattr(tracker, agent_key).value
            badge_class = f"badge-{status}"
            label = agent_labels[agent_key]
            st.markdown(
                f'<div class="agent-badge {badge_class}">{label}<br><small>{status.upper()}</small></div>',
                unsafe_allow_html=True,
            )


# ── Progress bar & status ──────────────────────────────────────────────────────
def render_progress(state: Optional[Dict], running: bool) -> None:
    """Render the workflow progress bar."""
    if running:
        progress = st.session_state.get("progress", 0.0)
        st.progress(progress, text=st.session_state.get("status_text", "Running..."))
    elif state:
        phase = state.get("workflow_phase", "")
        if "completed" in phase:
            st.progress(1.0, text="✅ Workflow completed")
        else:
            st.progress(0.5, text=f"Phase: {phase}")


# ── Execution trace ────────────────────────────────────────────────────────────
def render_execution_trace(state: Dict) -> None:
    """Render the live execution trace."""
    trace = state.get("execution_trace", [])
    if not trace:
        st.info("No trace entries yet.")
        return

    st.markdown(f"**{len(trace)} trace entries**")
    rows_html = []
    for entry in reversed(trace[-50:]):  # Show latest 50
        ts = entry.timestamp[:19] if hasattr(entry, "timestamp") else ""
        agent = entry.agent if hasattr(entry, "agent") else ""
        phase = entry.phase if hasattr(entry, "phase") else ""
        msg = entry.message if hasattr(entry, "message") else ""
        dur = f"{entry.duration_ms:.0f}ms" if (hasattr(entry, "duration_ms") and entry.duration_ms) else ""
        rows_html.append(
            f'<div class="trace-row">'
            f'<span style="color:#888">{ts}</span> '
            f'<span style="color:#1565c0;font-weight:600">[{agent}]</span> '
            f'<span style="color:#6a1b9a">{phase}</span> '
            f'<span>{msg[:100]}</span> '
            f'<span style="color:#999">{dur}</span>'
            f'</div>'
        )
    st.markdown("\n".join(rows_html), unsafe_allow_html=True)


# ── Report viewer ──────────────────────────────────────────────────────────────
def render_report(state: Dict) -> None:
    """Render the final report in tabbed sections."""
    report = state.get("final_report")
    if not report:
        st.warning("No report generated yet.")
        return

    # Metrics header
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Word Count", f"{report.word_count:,}")
    c2.metric("Citations", str(len(report.references)))
    c3.metric("Confidence", f"{report.overall_confidence:.0%}")
    c4.metric("Coverage", f"{report.citation_coverage:.0%}")

    # Section tabs
    tabs = st.tabs([
        "📋 Executive Summary",
        "💰 Pricing",
        "🚀 Products",
        "📈 Market Signals",
        "⚠️ Risks",
        "💡 Recommendations",
        "🌟 Opportunities",
        "📄 Full Report",
    ])

    with tabs[0]:
        st.markdown(f'<div class="report-container">{report.executive_summary}</div>', unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(f'<div class="report-container">{report.competitor_pricing}</div>', unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(f'<div class="report-container">{report.product_updates}</div>', unsafe_allow_html=True)
    with tabs[3]:
        st.markdown(f'<div class="report-container">{report.market_signals}</div>', unsafe_allow_html=True)
    with tabs[4]:
        st.markdown(f'<div class="report-container">{report.business_risks}</div>', unsafe_allow_html=True)
    with tabs[5]:
        st.markdown(f'<div class="report-container">{report.strategic_recommendations}</div>', unsafe_allow_html=True)
    with tabs[6]:
        opportunities = getattr(report, "opportunities", "") or "_No opportunities section generated._"
        st.markdown(f'<div class="report-container">{opportunities}</div>', unsafe_allow_html=True)
    with tabs[7]:
        st.markdown(report.markdown_content, unsafe_allow_html=False)


# ── Source cards ───────────────────────────────────────────────────────────────
def render_source_cards(state: Dict) -> None:
    """Render source cards showing all collected research sources."""
    research = state.get("research_result")
    if not research or not research.sources:
        st.info("No sources collected.")
        return

    sources = research.sources
    st.markdown(f"**{len(sources)} source(s) collected**")

    cols = st.columns(2)
    for i, src in enumerate(sources):
        with cols[i % 2]:
            trusted_badge = "✅ Trusted" if src.is_trusted_domain else "🌐 General"
            relevance_pct = f"{src.relevance_score:.0%}" if src.relevance_score else "N/A"
            date_str = f" · {src.publication_date}" if src.publication_date else ""
            st.markdown(
                f"""<div style="background:#1e2130;border:1px solid #2c3155;border-radius:8px;
                padding:12px;margin:6px 0;">
                <div style="font-weight:700;color:#90caf9;font-size:14px">{src.title[:70]}</div>
                <div style="font-size:11px;color:#9fa8da;margin:4px 0">{src.domain or 'unknown'}{date_str}
                &nbsp;·&nbsp;{trusted_badge}&nbsp;·&nbsp;Relevance: {relevance_pct}</div>
                <div style="font-size:12px;color:#c5cae9;margin-top:6px">{(src.summary or '')[:160]}</div>
                <div style="margin-top:8px"><a href="{src.url}" target="_blank"
                style="color:#64b5f6;font-size:11px">🔗 {src.url[:60]}...</a></div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Evaluation dashboard ───────────────────────────────────────────────────────
def render_evaluation_dashboard(state: Dict) -> None:
    """Render evaluation metrics dashboard."""
    report = state.get("final_report")
    verification = state.get("verification_result")
    research = state.get("research_result")
    meta = state.get("run_metadata")
    gov = state.get("governance_result")

    st.markdown("### 📊 Evaluation Dashboard")

    # Row 1 – workflow stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Workflow Steps", meta.total_steps if meta else "–")
    c2.metric("Sources Collected", meta.total_sources if meta else "–")
    c3.metric("Search Queries", meta.search_queries if meta else "–")
    c4.metric("Tool Calls", meta.tool_calls if meta else "–")
    c5.metric("Execution Time", f"{meta.duration_seconds:.1f}s" if meta and meta.duration_seconds else "–")

    st.markdown("---")

    # Row 2 – quality metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_claims = meta.total_claims if meta else 0
    verified = meta.verified_claims if meta else 0
    rejected = meta.rejected_claims if meta else 0
    unverified = total_claims - verified - rejected if total_claims else 0

    c1.metric("Total Claims", total_claims)
    c2.metric("Verified", verified)
    c3.metric("Unverified", unverified)
    c4.metric("Rejected", rejected)
    c5.metric("Citation Coverage",
              f"{verification.citation_coverage:.0%}" if verification else "–")
    c6.metric("Confidence Score",
              f"{report.overall_confidence:.0%}" if report else "–")

    st.markdown("---")

    # Row 3 – governance
    if gov:
        c1, c2, c3 = st.columns(3)
        c1.metric("Governance", "✅ Passed" if not gov.refusal_triggered else "❌ Refused")
        c2.metric("Governance Confidence", f"{gov.confidence_score:.0%}")
        c3.metric("Articles Parsed",
                  len([s for s in research.sources if s.raw_content]) if research else "–")

    # Errors count
    errors = state.get("errors", [])
    if errors:
        st.warning(f"⚠️ {len(errors)} error(s) recorded during execution.")


# ── Citation viewer ────────────────────────────────────────────────────────────
def render_citations(state: Dict) -> None:
    """Render the citations table."""
    report = state.get("final_report")
    if not report or not report.references:
        st.info("No citations available.")
        return

    st.markdown(f"**{len(report.references)} source(s) cited**")
    for cit in report.references:
        with st.expander(f"[{cit.number}] {cit.title[:70]}", expanded=False):
            st.markdown(f"**URL:** [{cit.url}]({cit.url})")
            st.markdown(f"**Domain:** `{cit.domain or 'unknown'}`")
            st.markdown(f"**Accessed:** {cit.accessed_date}")


# ── Download buttons ───────────────────────────────────────────────────────────
def render_downloads(state: Dict) -> None:
    """Render Markdown and PDF download buttons."""
    report = state.get("final_report")
    if not report:
        return

    topic = report.topic
    run_id = state.get("run_id", "")
    md_content = report.markdown_content

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Download Markdown",
            data=md_content.encode("utf-8"),
            file_name=f"intel_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col2:
        if st.button("⬇️ Download PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    export_md, export_pdf = _import_exporters()
                    pdf_path = export_pdf(md_content, topic, run_id=run_id)
                    if pdf_path and Path(pdf_path).exists():
                        pdf_bytes = Path(pdf_path).read_bytes()
                        st.download_button(
                            label="📥 Click to save PDF",
                            data=pdf_bytes,
                            file_name=Path(pdf_path).name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.warning("PDF generation failed – download Markdown instead.")
                except Exception as exc:
                    st.error(f"PDF export error: {exc}")


# ── Audit log viewer ───────────────────────────────────────────────────────────
def render_audit_log(state: Dict) -> None:
    """Render the audit log table."""
    audit_log = state.get("audit_log", [])
    if not audit_log:
        st.info("No audit entries.")
        return

    st.markdown(f"**{len(audit_log)} audit entries**")

    sev_colors = {"info": "#e8f5e9", "warning": "#fff9c4", "error": "#ffebee"}
    for entry in audit_log:
        sev = entry.severity if hasattr(entry, "severity") else "info"
        bg = sev_colors.get(sev, "#e8f5e9")
        ts = entry.timestamp[:19] if hasattr(entry, "timestamp") else ""
        agent = entry.agent if hasattr(entry, "agent") else ""
        ev = entry.event_type if hasattr(entry, "event_type") else ""
        desc = entry.description if hasattr(entry, "description") else ""
        st.markdown(
            f'<div style="background:{bg};padding:6px 10px;border-radius:4px;margin:2px 0;'
            f'font-size:13px"><b>{ts}</b> | <b>{agent}</b> | {ev} | {desc}</div>',
            unsafe_allow_html=True,
        )


# ── Human approval gate ────────────────────────────────────────────────────────
def render_approval_gate(state: Dict) -> Optional[bool]:
    """Render the human approval UI. Returns True/False/None."""
    report = state.get("final_report")
    if not report:
        return None

    st.markdown("---")
    st.markdown("## 👤 Human Review & Approval")
    st.info(
        f"The report is ready for review. "
        f"Confidence: **{report.overall_confidence:.0%}** | "
        f"Citations: **{len(report.references)}** | "
        f"Words: **{report.word_count:,}**"
    )

    # Show governance issues if any
    gov = state.get("governance_result")
    if gov and gov.issues:
        with st.expander("⚠️ Governance Issues", expanded=True):
            for issue in gov.issues:
                st.warning(issue)
    if gov and gov.warnings:
        with st.expander("ℹ️ Governance Warnings"):
            for w in gov.warnings:
                st.info(w)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Publish Report", type="primary", use_container_width=True):
            return True
    with col2:
        if st.button("🔄 Request Revision", type="secondary", use_container_width=True):
            return False
    return None


# ── Workflow runner (background thread) ───────────────────────────────────────
def _run_workflow_thread(
    topic: str,
    max_sources: int,
    max_steps: int,
    rag_enabled: bool,
    result_holder: dict,
    progress_queue: queue.Queue,
) -> None:
    """Execute the workflow in a background thread."""
    try:
        run_workflow = _import_workflow()
        final_state = run_workflow(
            topic=topic,
            max_steps=max_steps,
            max_sources=max_sources,
            rag_enabled=rag_enabled,
            human_approved=None,  # UI handles approval
        )
        result_holder["state"] = final_state
        result_holder["error"] = None
    except Exception as exc:
        result_holder["state"] = None
        result_holder["error"] = str(exc)
    finally:
        progress_queue.put("done")


# ── Main application ───────────────────────────────────────────────────────────
def main() -> None:
    """Main Streamlit application entry point."""

    # ── Header ────────────────────────────────────────────────────────────────
    col_logo, col_title = st.columns([1, 9])
    with col_logo:
        st.markdown("# 🔍")
    with col_title:
        st.title("Competitive Intelligence Briefing Crew")
        st.caption(
            "Powered by LangGraph · Multi-Agent AI · Automated Competitive Research"
        )

    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    config = render_sidebar()

    # ── Trigger workflow run ──────────────────────────────────────────────────
    if config["run_clicked"] and config["topic"].strip():
        topic = config["topic"].strip()
        st.session_state["running"] = True
        st.session_state["run_complete"] = False
        st.session_state["approval_pending"] = False
        st.session_state["approved"] = None
        st.session_state["workflow_state"] = None
        st.session_state["error"] = None
        st.session_state["progress"] = 0.05
        st.session_state["status_text"] = "Starting workflow..."

        # Run synchronously (Streamlit doesn't support true async here)
        with st.spinner(f"🔄 Running intelligence briefing for: **{topic}**"):
            progress_bar = st.progress(0.05, text="Initialising agents...")
            status_placeholder = st.empty()

            phase_progress = {
                "initializing": 0.05,
                "researching": 0.20,
                "analyzing": 0.45,
                "verifying": 0.65,
                "writing": 0.80,
                "governance": 0.92,
                "awaiting_approval": 0.97,
                "completed": 1.0,
            }

            try:
                run_workflow = _import_workflow()
                from agents.state import WorkflowPhase

                # Use streaming to show progress
                from graph.workflow import stream_workflow

                final_state = None
                for snapshot in stream_workflow(
                    topic=topic,
                    max_steps=config["max_steps"],
                    max_sources=config["max_sources"],
                    rag_enabled=config["rag_enabled"],
                    human_approved=None,
                ):
                    # snapshot is a dict of {node_name: state}
                    for node_name, node_state in snapshot.items():
                        phase = node_state.get("workflow_phase", "initializing")
                        prog = phase_progress.get(phase, 0.5)
                        progress_bar.progress(prog, text=f"⚙️ Agent: {node_name} | Phase: {phase}")
                        final_state = node_state

                if final_state is None:
                    final_state = run_workflow(
                        topic=topic,
                        max_steps=config["max_steps"],
                        max_sources=config["max_sources"],
                        rag_enabled=config["rag_enabled"],
                        human_approved=None,
                    )

                progress_bar.progress(0.97, text="🛡️ Governance check complete – awaiting approval")
                st.session_state["workflow_state"] = final_state
                st.session_state["running"] = False
                st.session_state["approval_pending"] = True
                st.session_state["last_run_id"] = final_state.get("run_id", "")

            except Exception as exc:
                st.session_state["error"] = str(exc)
                st.session_state["running"] = False
                progress_bar.progress(1.0, text="❌ Error")

    elif config["run_clicked"] and not config["topic"].strip():
        st.error("Please enter a research topic before running.")

    # ── Display error if any ──────────────────────────────────────────────────
    if st.session_state.get("error"):
        st.error(f"❌ Workflow Error: {st.session_state['error']}")
        with st.expander("Error Details"):
            st.code(st.session_state["error"])

    # ── Main content area ──────────────────────────────────────────────────────
    state = st.session_state.get("workflow_state")

    if state is None and not st.session_state.get("running"):
        # Welcome / instructions
        st.markdown("### 👋 Welcome")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**What this does:**
- 🔎 Researches your topic across the web
- 📊 Extracts competitor intelligence
- ✅ Verifies all claims with sources
- ✍️ Writes a professional report
""")
        with col2:
            st.markdown("""
**How to use:**
1. Enter a topic in the sidebar
2. Adjust max sources / steps
3. Click **Run Intelligence Briefing**
4. Review and approve the report
5. Download as Markdown or PDF
""")
        with col3:
            st.markdown("""
**Sample topics:**
- AI CRM market 2025
- Electric vehicle competitors
- Cloud computing AWS/Azure/GCP
- Cybersecurity vendors
- Healthcare AI diagnostics
""")
        return

    if state:
        # ── Progress ──────────────────────────────────────────────────────────
        render_progress(state, st.session_state.get("running", False))

        # ── Agent status ──────────────────────────────────────────────────────
        st.markdown("### 🤖 Agent Pipeline")
        render_agent_status(state)

        # ── Run metrics ───────────────────────────────────────────────────────
        meta = state.get("run_metadata")
        if meta:
            st.markdown("### 📊 Run Statistics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Steps", meta.total_steps)
            c2.metric("Sources", meta.total_sources)
            c3.metric("Claims", meta.total_claims)
            c4.metric("Verified", meta.verified_claims)
            c5.metric("Errors", meta.errors)

        # ── Human approval gate ───────────────────────────────────────────────
        if st.session_state.get("approval_pending") and not st.session_state.get("approved"):
            decision = render_approval_gate(state)
            if decision is True:
                st.session_state["approved"] = True
                st.session_state["approval_pending"] = False
                state["human_approved"] = True
                # Mark report as published
                report = state.get("final_report")
                if report:
                    report.run_metadata["published"] = True
                    report.run_metadata["approved_at"] = datetime.utcnow().isoformat()
                st.session_state["workflow_state"] = state
                st.session_state["run_complete"] = True
                st.rerun()
            elif decision is False:
                st.session_state["approval_pending"] = False
                st.warning("🔄 Revision requested. Please re-run with a refined topic or settings.")
                st.session_state["workflow_state"] = None

        # ── Tabs for report, trace, citations, audit ──────────────────────────
        if st.session_state.get("approved") or st.session_state.get("run_complete"):
            report = state.get("final_report")
            if report:
                st.success("✅ Report approved and ready for download")

            main_tabs = st.tabs([
                "📄 Report",
                "🔗 Citations",
                "🗂️ Sources",
                "⬇️ Download",
                "📊 Evaluation",
                "🔍 Execution Trace",
                "📋 Audit Log",
                "❌ Errors",
            ])

            with main_tabs[0]:
                render_report(state)

            with main_tabs[1]:
                render_citations(state)

            with main_tabs[2]:
                render_source_cards(state)

            with main_tabs[3]:
                render_downloads(state)

            with main_tabs[4]:
                render_evaluation_dashboard(state)

            with main_tabs[5]:
                render_execution_trace(state)

            with main_tabs[6]:
                render_audit_log(state)

            with main_tabs[7]:
                errors = state.get("errors", [])
                if not errors:
                    st.success("No errors recorded.")
                else:
                    st.markdown(f"**{len(errors)} error(s)**")
                    for e in errors:
                        st.error(e)

        elif st.session_state.get("approval_pending"):
            # Show report preview while awaiting approval
            with st.expander("👁️ Report Preview", expanded=True):
                render_report(state)

            main_tabs = st.tabs(["🔍 Execution Trace", "📋 Audit Log"])
            with main_tabs[0]:
                render_execution_trace(state)
            with main_tabs[1]:
                render_audit_log(state)

        else:
            # In-progress or failed state — show what went wrong
            errors = state.get("errors", [])
            report = state.get("final_report")
            research = state.get("research_result")

            if not report:
                st.error("❌ Workflow completed but no report was generated.")

                if not research or not getattr(research, "sources", None):
                    st.warning(
                        "🔎 Research returned 0 sources. Possible causes:\n"
                        "- DuckDuckGo rate-limited your IP (wait 1–2 minutes and retry)\n"
                        "- The LLM API key is invalid or over quota\n"
                        "- Network issue reaching search providers\n\n"
                        "**Tip:** Add a free Tavily API key at https://tavily.com and set "
                        "`TAVILY_API_KEY` in your `.env` for more reliable search results."
                    )
                elif errors:
                    st.warning(f"Research found {len(research.sources)} sources but the workflow failed during processing.")

            if errors:
                with st.expander("❌ Errors", expanded=True):
                    for e in errors:
                        st.error(e)

            with st.expander("🔍 Execution Trace", expanded=True):
                render_execution_trace(state)


if __name__ == "__main__":
    main()
