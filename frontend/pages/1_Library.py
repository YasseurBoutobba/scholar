import httpx
import pandas as pd
import streamlit as st
from common import API_BASE, STATUS_ICONS

st.header("Paper Library")
st.caption("Upload PDFs or paste arXiv links to build your research library.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_upload")
    if uploaded_file and st.button("Ingest PDF", use_container_width=True):
        with st.spinner("Parsing, chunking, and indexing..."):
            try:
                response = httpx.post(
                    f"{API_BASE}/papers",
                    files={
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                    },
                    timeout=180,
                )
                if response.status_code == 200:
                    st.success(f"Ingested: {response.json().get('title', 'Unknown')}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

with col2:
    st.subheader("arXiv Link")
    arxiv_url = st.text_input("Paste arXiv URL or ID", placeholder="e.g. 1706.03762")
    if arxiv_url and st.button("Ingest arXiv", use_container_width=True):
        with st.spinner("Fetching, parsing, and indexing..."):
            try:
                response = httpx.post(
                    f"{API_BASE}/papers/arxiv",
                    json={"url": arxiv_url},
                    timeout=180,
                )
                if response.status_code == 200:
                    st.success(f"Ingested: {response.json().get('title', 'Unknown')}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.divider()

header_col, refresh_col = st.columns([5, 1])
header_col.subheader("Your Papers")
if refresh_col.button("↻ Refresh", use_container_width=True):
    st.rerun()

try:
    response = httpx.get(f"{API_BASE}/papers", timeout=10)
    if response.status_code == 200:
        papers = response.json().get("papers", [])
        if papers:
            df = pd.DataFrame(
                [
                    {
                        "Status": STATUS_ICONS.get(p.get("ingestion_status", ""), "❓"),
                        "Title": p.get("title") or "Untitled",
                        "Type": "arXiv" if p.get("source") == "arxiv_url" else "Uploaded PDF",
                        "Link": (
                            f"https://arxiv.org/abs/{p['arxiv_id']}" if p.get("arxiv_id") else None
                        ),
                        "Year": p.get("year"),
                        "Chunks": p.get("chunk_count", 0),
                        "Error": p.get("error") or "",
                    }
                    for p in papers
                ]
            )
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Link": st.column_config.LinkColumn(
                        "Paper Link", display_text="Open ↗", width="small"
                    ),
                    "Year": st.column_config.NumberColumn("Year", format="%d"),
                    "Chunks": st.column_config.NumberColumn("Chunks"),
                    "Error": st.column_config.TextColumn("Error", width="medium"),
                },
            )
        else:
            st.info("No papers yet. Upload one above!")
    else:
        st.warning(f"Failed to load papers: {response.text}")
except Exception:
    st.warning("Backend not reachable at localhost:8000")
