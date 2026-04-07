# Permission-Aware Vector QA System

A production-style Retrieval-Augmented Generation (RAG) system built on the GitLab handbook. Every retrieved chunk is filtered through a role-based access-control layer before being passed to Claude. Retrieval combines dense (FAISS) and sparse (BM25) signals via Reciprocal Rank Fusion, with an optional neural cross-encoder reranker.

---

## Architecture

```
User Query
    │
    ├─► Embedder ──────────────────► FaissVectorStore (dense)
    │                                       │
    └─► BM25Store (sparse) ─────────────────┤
                                            │
                                    RRF Fusion (k=60)
                                            │
                                    AccessControl (role filter)
                                            │
                                    Reranker (optional, cross-encoder)
                                            │
                                    RAGPipeline
                                            │
                                    Claude API (streaming)
                                            │
                                    Answer + Citations + Latency
```

---

## Project Structure

```
project/
├── data/
│   └── index/                  # Built artifacts (gitignored — see Build Index)
│       ├── index.faiss         # FAISS dense index
│       ├── metadata.json       # Chunk records
│       └── bm25.pkl            # BM25 sparse index
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # Load .md/.txt, strip YAML front matter
│   │   ├── chunker.py          # Tiktoken-based chunking (target/max/overlap)
│   │   └── metadata.py         # Role inference, section extraction
│   ├── retrieval/
│   │   ├── embedder.py         # sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── vector_store.py     # FAISS IndexFlatIP (cosine similarity)
│   │   ├── bm25_store.py       # BM25Okapi sparse index (rank_bm25)
│   │   ├── retriever.py        # Dense-only retriever
│   │   ├── hybrid_retriever.py # RRF fusion of dense + sparse (default)
│   │   └── reranker.py         # Cross-encoder reranker (BAAI/bge-reranker-base)
│   ├── permissions/
│   │   └── access_control.py   # can_access(), filter_by_role(), AccessControl
│   ├── generation/
│   │   └── claude_client.py    # Claude API wrapper (streaming, TTFT measurement)
│   └── pipeline/
│       └── rag_pipeline.py     # End-to-end orchestrator, citation extraction
├── app/
│   └── streamlit_app.py        # Streamlit UI (streaming answer, citations, metrics)
├── scripts/
│   ├── build_index.py          # Build FAISS + BM25 index from handbook files
│   └── run_query.py            # CLI query runner / interactive REPL
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Install dependencies

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the `project/` directory:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Build the index

Point `--source` at the root of your handbook `.md` files:

```bash
python scripts/build_index.py \
    --source /path/to/handbook \
    --index-dir ./data/index
```

This runs 5 steps: load → chunk → embed → build FAISS → build BM25.
Output: `data/index/index.faiss`, `metadata.json`, `bm25.pkl` (~160 MB total, gitignored).

### 4. Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). The sidebar lets you choose model, top-k, and toggle the neural reranker.

### 5. Or run a CLI query

```bash
# Single query
python scripts/run_query.py --index-dir ./data/index --query "how do I request time off?"

# With role filtering
python scripts/run_query.py --index-dir ./data/index --role engineering --query "code review process"

# Interactive REPL
python scripts/run_query.py --index-dir ./data/index
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Hybrid retrieval (BM25 + FAISS) | Dense retrieval misses exact keywords; BM25 misses paraphrasing. RRF covers both blind spots. |
| RRF with k=60 | Standard constant that dampens top-rank dominance without losing signal from either retriever. |
| Permission filtering after fusion | Fuse first over maximum candidates, then gate — avoids role-leaking into ranking. |
| Cross-encoder reranker (optional) | BAAI/bge-reranker-base scores query–chunk pairs jointly; more accurate than bi-encoder but slower. |
| Streaming Claude with TTFT | `ClaudeClient` always uses streaming internally so time-to-first-token is measured accurately. |
| Role inference from file path | `finance/` → `["finance"]`, `engineering/` → `["engineering"]`, else → `["public"]`. Extend `_ROLE_RULES` in `metadata.py` to add more roles. |
| Chunk-level (not doc-level) permissions | A single document can have sections with different roles if split across paths. |
| Index artifacts gitignored | 161 MB total — too large for git. Rebuild with `build_index.py` or share via external storage. |

---

## Retrieval Pipeline Detail

```
build_index.py
  1. load_directory()        → Document[]         (strips YAML front matter)
  2. Chunker.chunk_document() → Chunk[]            (600 target / 800 max / 100 overlap tokens)
  3. attach_metadata()       → {text, metadata}[] (roles, section, chunk_id, doc_id)
  4. Embedder.embed_texts()  → float32[N, 384]    (all-MiniLM-L6-v2)
  5. FaissVectorStore.save() → index.faiss + metadata.json
  6. BM25Store.save()        → bm25.pkl

retrieve(query, user_role)
  1. embed_query()           → float32[384]
  2. faiss_store.search()    → top 3k dense candidates
  3. bm25_store.search()     → top 3k sparse candidates
  4. _rrf_fuse()             → unified ranked list  (score = Σ 1/(60+rank))
  5. filter_search_results() → role-permitted subset
  6. reranker.rerank()       → cross-encoder scores [0,1]  (if enabled)
  7. return top-k
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (required) |

All other parameters (model, top-k, index directory, reranker on/off) are configurable at runtime via the Streamlit sidebar or CLI flags.

---

## Running Tests

```bash
pytest tests/ -v
```
