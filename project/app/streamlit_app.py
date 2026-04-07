"""
app/streamlit_app.py

Streamlit UI for the Permission-Aware GitLab Handbook QA system.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

# Allow imports from the project root when running via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.generation.claude_client import ClaudeClient
from src.permissions.access_control import AccessControl
from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import FaissVectorStore
from src.utils.cache import QueryCache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data" / "index"

_ROLE_OPTIONS: dict[str, str | None] = {
    "All content (admin)": None,
    "Public": "public",
    "Engineering": "engineering",
    "Finance": "finance",
}

_MODEL_OPTIONS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

# ---------------------------------------------------------------------------
# Cached pipeline loader  (runs once; reloads only when arguments change)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading index and embedding model…")
def _load_pipeline(index_dir: str, model: str, top_k: int, use_reranker: bool) -> RAGPipeline:
    index_path = Path(index_dir)
    faiss_store = FaissVectorStore.load(index_path)
    bm25_store = BM25Store.load(index_path)
    embedder = Embedder()
    reranker = Reranker() if use_reranker else None
    retriever = HybridRetriever(
        embedder=embedder,
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        access_control=AccessControl(),
        reranker=reranker,
        top_k=top_k,
    )
    claude = ClaudeClient(model=model)
    cache = QueryCache(cache_file=Path(index_dir) / "query_cache.pkl")
    return RAGPipeline(retriever=retriever, claude_client=claude, cache=cache)


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def _stream_rag(pipeline: RAGPipeline, query: str, role: str | None):
    """Run the pipeline in a background thread and yield tokens as they arrive.

    Yields str tokens for `st.write_stream`, then stores the final RAGResponse
    in st.session_state["last_result"] once the thread finishes.
    """
    token_queue: queue.Queue[str | None] = queue.Queue()
    result_box: dict = {}

    def _worker() -> None:
        def on_token(t: str) -> None:
            token_queue.put(t)

        result_box["result"] = pipeline.run_stream(
            query, user_role=role, on_token=on_token
        )
        token_queue.put(None)  # sentinel — generation complete

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        token = token_queue.get()
        if token is None:
            break
        yield token

    thread.join()
    st.session_state["last_result"] = result_box.get("result")


# ---------------------------------------------------------------------------
# Page layout helpers
# ---------------------------------------------------------------------------

_CONF_COLOR = {"low": "#e53935", "medium": "#f9a825", "high": "#43a047"}


def _render_confidence(result: RAGResponse) -> None:
    color = _CONF_COLOR.get(result.confidence_label, "grey")
    cache_badge = (
        " &nbsp;<span style='background:#1565c0; color:white; "
        "font-size:0.78em; padding:2px 7px; border-radius:4px;'>"
        "⚡ Cache hit</span>"
        if result.cache_hit else ""
    )
    st.markdown(
        f"**Confidence:** "
        f"<span style='color:{color}; font-weight:600;'>"
        f"{result.confidence:.2f} ({result.confidence_label.upper()})"
        f"</span>{cache_badge}",
        unsafe_allow_html=True,
    )


def _render_metrics(result: RAGResponse) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TTFT", f"{result.ttft_seconds:.2f}s",
                help="Time to first token — how quickly Claude started responding.")
    col2.metric("Total latency", f"{result.total_seconds:.2f}s",
                help="Total wall-clock time for the full response.")
    col3.metric("Input tokens", f"{result.input_tokens:,}",
                help="Tokens consumed by the prompt (context + question).")
    col4.metric("Output tokens", f"{result.output_tokens:,}",
                help="Tokens generated in the answer.")


def _render_citations(result: RAGResponse) -> None:
    if not result.citations:
        st.caption("No citations extracted.")
        return

    for c in result.citations:
        section_str = f" — {c.section}" if c.section else ""
        st.markdown(
            f"**[{c.index}]** `{c.filename}`{section_str} "
            f"<span style='color:grey; font-size:0.85em;'>(score: {c.score:.3f})</span>",
            unsafe_allow_html=True,
        )


def _render_chunks(result: RAGResponse) -> None:
    if not result.chunks:
        st.caption("No chunks retrieved.")
        return

    for i, chunk in enumerate(result.chunks, start=1):
        filename = chunk.metadata.get("filename", "?")
        section = chunk.metadata.get("section", "")
        roles = ", ".join(chunk.metadata.get("roles", []))
        tokens = chunk.metadata.get("token_count", "?")
        label = f"[{i}]  {filename}"
        if section:
            label += f"  ·  {section}"
        label += f"  ·  score {chunk.score:.3f}"

        with st.expander(label):
            st.caption(f"Roles: {roles}  |  Tokens: {tokens}")
            st.text(chunk.text)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def _render_about() -> None:
    st.title("About — GitLab Handbook QA")
    st.caption("A permission-aware RAG system built on the GitLab handbook.")

    st.divider()

    st.subheader("What it does")
    st.markdown(
        """
        This system lets you ask natural-language questions about the GitLab handbook
        and get grounded, cited answers generated by Claude.

        Every answer is built exclusively from retrieved handbook chunks — System never
        invents facts outside the provided context.  Chunks are filtered by your user
        role before generation, so restricted content (finance, engineering) is never
        exposed to unauthorised roles.
        """
    )

    st.divider()

    st.subheader("Features")
    st.markdown(
        """
        | Feature | Detail |
        |---|---|
        | **Hybrid retrieval** | BM25 sparse + FAISS dense, fused with Reciprocal Rank Fusion |
        | **Neural reranker** | Optional BAAI/bge-reranker-base cross-encoder for higher precision |
        | **Permission filtering** | Role-based access control at chunk level (finance / engineering / public) |
        | **Retrieval confidence** | Multi-signal score (avg · consistency · dominance) with low/medium/high labels |
        | **Fail-safe abstention** | Pipeline abstains instead of hallucinating when confidence is too low |
        | **Query cache** | In-memory + disk cache (24 h TTL) — repeated queries return instantly |
        | **Streaming answers** | Tokens streamed from model with TTFT measurement |
        | **Citations** | Every claim linked back to the source file and section |
        """
    )

    st.divider()

    st.subheader("Architecture")
    st.code(
        """
User Query
    │
    ├─► Embedder (all-MiniLM-L6-v2) ──► FaissVectorStore   [dense]
    │                                          │
    └─► BM25Store                ─────────────┤ [sparse]
                                              │
                                     RRF Fusion (k=60)
                                              │
                                     AccessControl (role filter)
                                              │
                                     Reranker (optional)
                                     BAAI/bge-reranker-base
                                              │
                                     Confidence Scoring
                                     avg · consistency · dominance
                                              │
                                     QueryCache  ◄── hit? return early
                                              │
                                     RAGPipeline
                                              │
                                     Claude API  (streaming)
                                              │
                                     Answer + Citations + Latency
        """,
        language="text",
    )

    st.divider()

    st.subheader("Ingestion pipeline")
    st.markdown(
        """
        ```
        Handbook .md files
            │
            ▼
        loader.py          Strip YAML front matter, HTML comments
            │
            ▼
        chunker.py         Tiktoken-based splitting
                           target 600 · max 800 · overlap 100 tokens
            │
            ▼
        metadata.py        Role inference from file path
                           finance/ → finance
                           engineering/ → engineering
                           else → public
            │
            ├──► FaissVectorStore  (index.faiss + metadata.json)
            └──► BM25Store         (bm25.pkl)
        ```
        """
    )

    st.divider()
    st.caption("Built with sentence-transformers · FAISS · rank-bm25 · Anthropic Claude · Streamlit")


def main() -> None:
    st.set_page_config(
        page_title="GitLab Handbook QA",
        page_icon="📚",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("GitLab Handbook QA")
        st.divider()

        page = st.radio("Navigate", ["Ask", "About"], horizontal=True)
        st.divider()

        model = st.selectbox("Model", _MODEL_OPTIONS, index=0)
        top_k = st.slider("Retrieved chunks (k)", min_value=1, max_value=10, value=5)
        use_reranker = st.toggle(
            "Neural reranker",
            value=False,
            help="Run BAAI/bge-reranker-base cross-encoder after RRF fusion. "
                 "Scores become [0, 1] probabilities. Slower but more accurate.",
        )

        st.divider()
        st.caption("Answers grounded in the GitLab handbook.")

    if page == "About":
        _render_about()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("Ask the Handbook")
    st.caption("Answers use only retrieved context")

    st.divider()

    # ── Input form ────────────────────────────────────────────────────────────
    query = st.text_area(
        "Your question",
        placeholder="e.g. How do I request time off?",
        height=90,
        label_visibility="collapsed",
    )

    col_role, col_btn = st.columns([3, 1])
    with col_role:
        role_label = st.selectbox(
            "User role",
            options=list(_ROLE_OPTIONS.keys()),
            index=0,
            help=(
                "Controls which chunks are visible. "
                "'All content' disables filtering (admin mode)."
            ),
        )
    with col_btn:
        st.write("")  # nudge button down to align with selectbox
        submit = st.button("Ask", type="primary", use_container_width=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if submit:
        if not query.strip():
            st.warning("Please enter a question before submitting.")
            st.stop()

        user_role = _ROLE_OPTIONS[role_label]
        pipeline = _load_pipeline(str(_DEFAULT_INDEX), model, top_k, use_reranker)

        st.divider()

        # Streaming answer
        st.subheader("Answer")
        st.write_stream(_stream_rag(pipeline, query.strip(), user_role))

        result: RAGResponse | None = st.session_state.get("last_result")
        if result is None:
            st.error("Pipeline returned no result. Check the index path and API key.")
            st.stop()

        _render_confidence(result)

        st.divider()

        # Latency metrics
        st.subheader("Performance")
        _render_metrics(result)

        st.divider()

        # Citations
        st.subheader("Citations")
        _render_citations(result)

        st.divider()

        # Retrieved chunks (expandable)
        st.subheader(f"Retrieved Context  ({result.chunks_used} chunks)")
        _render_chunks(result)


if __name__ == "__main__":
    main()
