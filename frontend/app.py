import httpx
import streamlit as st

st.set_page_config(page_title="Scholar - Research Assistant", page_icon="📚", layout="wide")

st.title("📚 Scholar")
st.markdown("*Agentic RAG research assistant for academic papers*")

try:
    health = httpx.get("http://localhost:8000/health", timeout=3).json()
    overall = health.get("status", "unknown")
    icon = "🟢" if overall == "healthy" else "🟡"
    st.caption(f"{icon} Backend: {overall}")
except Exception:
    st.caption("🔴 Backend unreachable — is `uvicorn backend.main:app` running?")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("📥 Library")
    st.markdown(
        "Upload a PDF or paste an arXiv link. Scholar parses it, splits it into "
        "section-aware chunks, and indexes it for retrieval."
    )
with col2:
    st.subheader("💬 Query")
    st.markdown(
        "Ask a research question in a chat interface. Scholar decomposes it, "
        "retrieves across your library, reasons with citations, checks for "
        "contradictions, and writes a report — with the conversation kept "
        "in view as you ask follow-ups."
    )
with col3:
    st.subheader("🕘 History")
    st.markdown(
        "Browse every past research run, or look one up by trace ID, and "
        "revisit its full report and audit trail."
    )

st.divider()
st.caption("Use the sidebar to navigate between pages.")
