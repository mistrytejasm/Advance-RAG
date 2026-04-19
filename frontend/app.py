"""
Advanced RAG Engine — Streamlit Frontend

Full pipeline:
  Upload PDF → Parse & Chunk (POST /upload)
             → Embed to Pinecone  (POST /embed/{doc_id})
             → List documents     (GET  /documents)

Query pipeline:
  User question → POST /answer → display answer + developer inspector
"""

import time
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced RAG Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.pipeline-step {
    display: flex; align-items: center;
    padding: 0.45rem 0.75rem; border-radius: 8px; margin: 3px 0; font-size: 0.88rem;
}
.step-done { background:#d4edda; color:#155724; border-left:4px solid #28a745; }
.step-run  { background:#fff3cd; color:#856404; border-left:4px solid #ffc107; }
.step-wait { background:#f8f9fa; color:#6c757d; border-left:4px solid #dee2e6; }
.step-err  { background:#f8d7da; color:#721c24; border-left:4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


# ── Helper: coloured pipeline step ───────────────────────────────────────────
def pipeline_step(label: str, status: str, detail: str = "") -> None:
    icons = {"wait": "⬜", "run": "🔄", "done": "✅", "err": "❌"}
    css   = {"wait": "step-wait", "run": "step-run", "done": "step-done", "err": "step-err"}
    detail_html = f'&nbsp;—&nbsp;<span style="font-weight:400">{detail}</span>' if detail else ""
    st.markdown(
        f'<div class="pipeline-step {css[status]}">'
        f'{icons[status]}&nbsp;&nbsp;<b>{label}</b>{detail_html}</div>',
        unsafe_allow_html=True,
    )


# ── Helper: Pipeline Inspector (MUST be defined before chat history loop) ─────
def render_inspector(data: dict) -> None:
    """Render the 'Under the Hood' expander for an answer payload."""
    with st.expander("🛠️ Pipeline Inspector — Under the Hood"):
        llm_meta  = data.get("llm_metadata", {})
        retrieval = data.get("retrieval", {})
        qu        = retrieval.get("query_understanding", {})

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⏱️ Total",      f"{data.get('total_latency_ms', 0)} ms")
        c2.metric("🔍 Retrieval",  f"{retrieval.get('latency_ms', 0)} ms")
        c3.metric("📥 Tokens In",  llm_meta.get("input_tokens", 0))
        c4.metric("📤 Tokens Out", llm_meta.get("output_tokens", 0))
        c5.metric("🎯 Confidence", round(data.get("confidence", 0), 3))

        st.divider()
        col_a, col_b = st.columns(2)

        # Query Understanding
        with col_a:
            st.markdown("**🧠 Query Understanding**")
            st.markdown(f"- **Type:** `{qu.get('query_type', 'n/a')}`")
            st.markdown(f"- **Route:** `{qu.get('search_route', 'n/a')}`")
            st.markdown(f"- **Rewrite applied:** `{qu.get('rewrite_applied', False)}`")
            if qu.get("rewrite_applied"):
                st.markdown(f"- **Rewritten query:** _{qu.get('rewritten_query', '')}_")
            st.markdown(
                f"- **Vector weight:** `{qu.get('vector_weight', 0)}`  "
                f"| **BM25:** `{qu.get('bm25_weight', 0)}`"
            )

        # Citations
        with col_b:
            st.markdown("**🔗 Citations**")
            citations = data.get("citations", [])
            if citations:
                for i, c in enumerate(citations):
                    st.markdown(
                        f"**[{i+1}]** `{c.get('source','?')}` "
                        f"· Page {c.get('page', '?')} "
                        f"· Score `{round(c.get('rerank_score', 0), 4)}`"
                    )
            else:
                st.caption("No citations — answer is ungrounded.")

        st.divider()

        # Retrieved chunks
        results = retrieval.get("results", [])
        if results:
            st.markdown(f"**📦 Retrieved Chunks ({len(results)} total)**")
            for r in results:
                short_id = r.get("chunk_id", "?")[:12]
                score    = round(r.get("rerank_score", 0), 4)
                with st.expander(f"Chunk `{short_id}…` — rerank score `{score}`"):
                    st.caption(
                        f"Page {r.get('page','?')} · "
                        f"Section: {r.get('section','?')} · "
                        f"Source: {r.get('source','?')}"
                    )
                    st.markdown(r.get("content", ""))
        else:
            st.warning("No chunks retrieved — document may not be indexed yet or vectors are in wrong namespace.")

        st.divider()
        with st.expander("📄 Raw API Response (JSON)"):
            st.json(data)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Advanced RAG")
    st.caption("Upload → Chunk → Embed → Query")
    st.divider()

    # Upload section
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if st.button("🚀 Process & Index", use_container_width=True, type="primary"):
        if uploaded_file is None:
            st.warning("Please select a PDF first.")
        else:
            st.markdown("**⚙️ Pipeline Progress**")

            # Step 1 — Upload & chunk
            pipeline_step("Saving & parsing document", "run")
            files  = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            doc_id = None

            try:
                res = requests.post(f"{API_URL}/upload", files=files, timeout=1000)
                if res.status_code == 200:
                    d        = res.json()
                    doc_id   = d["document_id"]
                    n_raw    = d.get("raw_elements", 0)
                    n_chunks = d.get("semantic_chunks", 0)
                    pipeline_step("Saving & parsing document", "done",
                                  f"{n_raw} elements extracted")
                    pipeline_step("Semantic chunking → MongoDB", "done",
                                  f"{n_chunks} chunks stored")
                elif res.status_code == 409:
                    pipeline_step("Saving & parsing document", "err", "Duplicate detected")
                    st.warning(
                        f"⚠️ **'{uploaded_file.name}'** is already indexed. "
                        "Delete it from **Indexed Documents** first, then re-upload."
                    )
                    st.stop()
                else:
                    pipeline_step("Saving & parsing document", "err", res.text[:80])
                    st.stop()
            except Exception as e:
                pipeline_step("Saving & parsing document", "err", str(e)[:80])
                st.stop()

            # Step 2 — Embed
            pipeline_step("Generating embeddings → Pinecone", "run")
            try:
                emb_res = requests.post(f"{API_URL}/embed/{doc_id}", timeout=1000)
                if emb_res.status_code == 200:
                    ed       = emb_res.json()
                    embedded = ed.get("embedded", 0)
                    secs     = ed.get("duration_seconds", 0)
                    pipeline_step("Generating embeddings → Pinecone", "done",
                                  f"{embedded} vectors in {secs}s")
                else:
                    pipeline_step("Generating embeddings → Pinecone", "err",
                                  emb_res.text[:80])
                    st.stop()
            except Exception as e:
                pipeline_step("Generating embeddings → Pinecone", "err", str(e)[:80])
                st.stop()

            st.success(f"✅ **'{uploaded_file.name}'** is fully indexed!")
            st.rerun()

    # Document library
    st.divider()
    st.subheader("📚 Indexed Documents")
    try:
        docs_res  = requests.get(f"{API_URL}/documents", timeout=5)
        documents = docs_res.json().get("documents", []) if docs_res.status_code == 200 else []
    except Exception:
        documents = []
        st.warning("⚠️ Backend unreachable. Is FastAPI running?")

    if not documents:
        st.caption("No documents uploaded yet.")
    else:
        for doc in documents:
            doc_id   = doc["document_id"]
            filename = doc["filename"]
            c1, c2   = st.columns([5, 1])
            with c1:
                st.caption(f"📄 {filename}")
                st.caption(f"`{doc_id[:16]}…`")
            with c2:
                if st.button("🗑️", key=f"del_{doc_id}", help="Delete from DB & Pinecone"):
                    with st.spinner("Deleting…"):
                        d_res = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=30)
                        if d_res.status_code == 200:
                            st.success("Deleted!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Delete failed")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🧠 Advanced RAG — Document Q&A")
st.markdown(
    "Ask any question. The system retrieves relevant chunks via **Hybrid Search** "
    "(Semantic Vector + BM25), reranks with a Cross-Encoder, and generates a grounded answer."
)
st.divider()

# Render historical messages — render_inspector is defined above
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            render_inspector(msg["metadata"])

# Chat input
prompt = st.chat_input("Ask a question about your documents…")

if prompt:
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 Retrieving & generating answer…")

        try:
            response = requests.post(
                f"{API_URL}/answer",
                json={"query": prompt},
                timeout=120,
            )

            if response.status_code == 200:
                data   = response.json()
                answer = data.get("answer", "No answer generated.")

                placeholder.markdown(answer)
                render_inspector(data)

                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  answer,
                    "metadata": data,
                })
            else:
                err = f"❌ API Error {response.status_code}: {response.text}"
                placeholder.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

        except Exception as e:
            err = f"❌ Connection error: {e}"
            placeholder.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
