import httpx
import streamlit as st
from common import API_BASE, STATUS_ICONS, render_report

st.header("History")
st.caption("Browse past research runs, or look one up directly by trace ID.")

if "selected_trace_id" not in st.session_state:
    st.session_state.selected_trace_id = None

with st.expander("Look up by trace ID"):
    manual_id = st.text_input(
        "Trace ID", label_visibility="collapsed", placeholder="e.g. 88fcd145eed14a84"
    )
    if manual_id and st.button("Look up"):
        st.session_state.selected_trace_id = manual_id.strip()

st.subheader("Recent runs")
try:
    response = httpx.get(f"{API_BASE}/runs", timeout=10)
    response.raise_for_status()
    runs = response.json().get("runs", [])
except Exception as e:
    runs = []
    st.warning(f"Backend not reachable: {e}")

if not runs:
    st.info("No research runs yet — ask a question on the Query page.")
else:
    for run in runs:
        icon = STATUS_ICONS.get(run["status"], "❓")
        label = f"{icon} {run['question'] or '(untitled)'}"
        cols = st.columns([5, 2, 2])
        cols[0].markdown(label)
        cols[1].caption(run["status"])
        cols[2].caption((run.get("created_at") or "")[:19].replace("T", " "))
        if st.button("View", key=f"view_{run['trace_id']}"):
            st.session_state.selected_trace_id = run["trace_id"]

trace_id = st.session_state.selected_trace_id
if trace_id:
    st.divider()
    st.subheader(f"Run detail — `{trace_id}`")
    try:
        response = httpx.get(f"{API_BASE}/runs/{trace_id}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            run = data["run"]

            st.markdown(f"**Question:** {run['question']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", run["status"])
            c2.metric("Classification", run["classification"])
            c3.metric("Created", (run.get("created_at") or "")[:19].replace("T", " "))

            if data.get("report"):
                st.divider()
                st.subheader("Report")
                render_report(data["report"])
            elif run["status"] == "running":
                st.info("This run is still in progress.")
            elif run["status"] == "failed":
                st.error("This run failed before producing a report.")

            st.subheader("Audit log")
            for entry in data.get("audit_log", []):
                with st.expander(f"{entry['phase']} — {entry['event']}", expanded=False):
                    st.json(entry.get("payload", {}))
                    st.caption(f"Time: {entry.get('timestamp', 'N/A')}")
        else:
            st.error(f"Run not found: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")
