import httpx
import streamlit as st
from common import API_BASE, iter_sse_events, render_report

st.header("Research Chat")
st.caption(
    "Ask a question across your paper library. Each question is answered in "
    "context of this conversation's history — use **New chat** to start over."
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
    st.caption(f"{len(st.session_state.chat_messages) // 2} question(s) this session")


def _run_research(question: str) -> dict:
    phase_status = st.status("Starting research...", expanded=True)
    st.subheader("Sub-questions")
    sub_questions_container = st.container()
    contradiction_container = st.container()
    report_container = st.container()
    sub_answer_expanders: dict[int, object] = {}

    message = {"role": "assistant", "content": question, "report": None, "error": None}

    try:
        with httpx.stream(
            "POST", f"{API_BASE}/research", json={"question": question}, timeout=300
        ) as response:
            if response.status_code != 200:
                err = f"Request failed ({response.status_code}): {response.read().decode()}"
                phase_status.update(label="Research failed", state="error")
                st.error(err)
                message["error"] = err
                return message

            for event_type, data in iter_sse_events(response):
                if event_type == "phase":
                    phase, status = data.get("phase", ""), data.get("status", "")
                    if status == "running":
                        phase_status.update(label=f"Running: {phase}...", state="running")
                    elif status == "complete":
                        phase_status.update(label=f"{phase.capitalize()} complete", state="running")

                elif event_type == "sub_question":
                    idx = data["index"]
                    with sub_questions_container:
                        with st.expander(f"Sub-question {idx + 1}: {data['text']}", expanded=False):
                            sub_answer_expanders[idx] = st.container()

                elif event_type == "sub_answer":
                    idx = data["index"]
                    box = sub_answer_expanders.get(idx)
                    if box is not None:
                        with box:
                            st.markdown(data["answer"])
                            if data.get("sources"):
                                st.caption("Sources (paper IDs): " + ", ".join(data["sources"]))

                elif event_type == "confidence":
                    box = sub_answer_expanders.get(data.get("section"))
                    with box if box is not None else st:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Faithfulness", data["faithfulness"])
                        c2.metric("Relevance", data["relevance"])
                        c3.metric("Context precision", data["context_precision"])

                elif event_type == "contradiction":
                    with contradiction_container:
                        st.warning(
                            f"⚠️ Contradiction: {data['description']}\n\n"
                            f"Papers in conflict: {', '.join(data.get('papers', []))}"
                        )

                elif event_type == "error":
                    if data.get("scope") == "sub_question":
                        box = sub_answer_expanders.get(data.get("index"))
                        with box if box is not None else st:
                            st.error(f"Failed: {data.get('error')}")
                    else:
                        phase_status.update(label="Research failed", state="error")
                        st.error(f"Research failed: {data.get('error')}")
                        message["error"] = data.get("error")

                elif event_type == "report":
                    with report_container:
                        st.divider()
                        render_report(data)
                    message["report"] = data

                elif event_type == "done":
                    phase_status.update(label="Research complete!", state="complete")
                    st.caption(
                        f"Trace ID: `{data['trace_id']}` — findable later on the History page"
                    )

    except Exception as e:
        phase_status.update(label=f"Error: {e}", state="error")
        st.error(f"Connection error: {e}")
        message["error"] = str(e)

    return message


for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg.get("report"):
            render_report(msg["report"])
        elif msg.get("error"):
            st.error(msg["error"])
        else:
            st.markdown(msg["content"])

if question := st.chat_input("Ask a research question..."):
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        assistant_message = _run_research(question)
    st.session_state.chat_messages.append(assistant_message)
