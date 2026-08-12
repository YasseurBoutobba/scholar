import json

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

STATUS_ICONS = {"ready": "✅", "processing": "⏳", "failed": "❌", "completed": "✅"}


def iter_sse_events(response: httpx.Response):
    event_type = "message"
    data_lines: list[str] = []

    for line in response.iter_lines():
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    yield event_type, json.loads(raw)
                except json.JSONDecodeError:
                    pass
            event_type = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())


def render_report(report: dict) -> None:
    st.markdown("#### 💡 Answer")
    st.markdown(report["executive_summary"])

    sections = report.get("sections", [])
    scores = report.get("confidence_scores")
    contradictions = report.get("contradictions", [])
    sources = report.get("source_list", [])
    if not (sections or scores or contradictions or sources):
        return

    st.divider()
    st.markdown("#### 📋 Final Report")

    if scores:
        cols = st.columns(len(scores))
        for col, (key, value) in zip(cols, scores.items(), strict=False):
            col.metric(key.replace("_", " ").title(), round(value, 2))

    for c in contradictions:
        st.warning(
            f"⚠️ {c['description']}\n\n"
            f"Papers in conflict: {', '.join(c.get('papers_in_conflict', []))}"
        )

    if sections:
        st.caption(
            "Findings — one per sub-question this was broken into, each cited independently."
        )
        for section in sections:
            with st.expander(section["title"], expanded=False):
                st.markdown(section["content"])

    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for src in sources:
                st.markdown(f"- {src.get('title', 'Unknown')} (p. {src.get('page', '?')})")
