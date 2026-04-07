# Permission-Aware Vector QA System

A production-style Retrieval-Augmented Generation (RAG) system built on the GitLab handbook. Every retrieved chunk is filtered through a role-based access-control layer before being passed to Claude. Retrieval combines dense (FAISS) and sparse (BM25) signals via Reciprocal Rank Fusion, with an optional neural cross-encoder reranker, retrieval confidence scoring, and a query cache.

---

## Architecture

```
User Query
    │
    ├─► QueryCache ──── hit? ──────────────────────────────► Return instantly
    │
    ├─► Embedder ──────────────────► FaissVectorStore (dense)
    │                                       │
    └─► BM25Store (sparse) ─────────────────┤
                                            │
                                    RRF Fusion (k=60)
                                    Scores normalised to [0, 1]
                                            │
                                    AccessControl (role filter)
                                            │
                                    Reranker (optional)
                                    BAAI/bge-reranker-base
                                            │
                                    Confidence Scoring
                                    avg · consistency · dominance
                                            │
                              conf < 0.15 → abstain
                              conf < 0.30 → answer with ⚠️ warning
                                            │
                                    Claude API (streaming)
                                            │
                                    Answer + Citations + Latency
                                            │
                                    QueryCache.set()
```

---

## Project Structure

```
project/
├── data/
│   └── index/                    # Built artifacts (gitignored — see Build Index)
│       ├── index.faiss           # FAISS dense index
│       ├── metadata.json         # Chunk records
│       ├── bm25.pkl              # BM25 sparse index
│       └── query_cache.pkl       # Persistent query cache (24 h TTL)
├── src/
│   ├── ingestion/
│   │   ├── loader.py             # Load .md/.txt, strip YAML front matter; defines Document
│   │   ├── chunker.py            # Tiktoken-based chunking (target/max/overlap)
│   │   └── metadata.py           # Role inference from path, section extraction
│   ├── retrieval/
│   │   ├── embedder.py           # sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── vector_store.py       # FAISS IndexFlatIP (cosine similarity)
│   │   ├── bm25_store.py         # BM25Okapi sparse index (rank_bm25)
│   │   ├── retriever.py          # Dense-only retriever
│   │   ├── hybrid_retriever.py   # RRF fusion of dense + sparse (default)
│   │   └── reranker.py           # Cross-encoder reranker (BAAI/bge-reranker-base)
│   ├── permissions/
│   │   └── access_control.py     # can_access(), filter_by_role(), AccessControl
│   ├── generation/
│   │   └── claude_client.py      # Claude API wrapper (streaming, TTFT measurement)
│   ├── pipeline/
│   │   ├── rag_pipeline.py       # End-to-end orchestrator, citation extraction
│   │   └── confidence.py         # Multi-signal retrieval confidence scoring
│   └── utils/
│       └── cache.py              # In-memory + disk query cache (pickle, 24 h TTL)
├── app/
│   └── streamlit_app.py          # Streamlit UI — Ask page + About page
├── scripts/
│   ├── build_index.py            # Build FAISS + BM25 index from handbook files
│   └── run_query.py              # CLI query runner / interactive REPL
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

### 3. Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). The sidebar has model selector, top-k slider, and neural reranker toggle. Use the **Ask / About** toggle to switch pages.

### 4. Or run a CLI query

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
| RRF scores normalised to [0, 1] | Raw RRF scores (~0.016–0.032) are unreadable. Dividing by the theoretical max makes 1.0 = top of both retrievers, 0.5 = top of one only. |
| Permission filtering after fusion | Fuse over maximum candidates first, then gate — prevents role from skewing the ranking. |
| Cross-encoder reranker (optional) | BAAI/bge-reranker-base scores query–chunk pairs jointly; more accurate than bi-encoder but slower. Toggle in sidebar. |
| Confidence scoring before generation | Three signals (avg score · consistency · dominance) decide whether to answer, warn, or abstain — avoids hallucination on weak retrievals. |
| Query cache (memory + disk) | Exact repeated queries (same text + role) skip retrieval and generation entirely. Cache survives server restarts via pickle. TTL = 24 h. |
| Streaming Claude with TTFT | `ClaudeClient` always uses streaming internally so time-to-first-token is accurately measured even for blocking calls. |
| Role inference from file path | `finance/` → `["finance"]`, `engineering/` → `["engineering"]`, else → `["public"]`. Extend `_ROLE_RULES` in `metadata.py` to add more roles. |
| Chunk-level permissions | Filtering happens per chunk, not per document — fine-grained control even within a single file. |

---

## Retrieval Pipeline Detail

```
build_index.py
  1. load_directory()         → Document[]          (strips YAML front matter)
  2. Chunker.chunk_document() → Chunk[]             (600 target / 800 max / 100 overlap tokens)
  3. attach_metadata()        → {text, metadata}[]  (roles, section, chunk_id, doc_id)
  4. Embedder.embed_texts()   → float32[N, 384]     (all-MiniLM-L6-v2)
  5. FaissVectorStore.save()  → index.faiss + metadata.json
  6. BM25Store.save()         → bm25.pkl

retrieve(query, user_role)
  1. QueryCache.get()         → return immediately on hit
  2. embed_query()            → float32[384]
  3. faiss_store.search()     → top 3×k dense candidates
  4. bm25_store.search()      → top 3×k sparse candidates
  5. _rrf_fuse()              → unified list, scores normalised to [0, 1]
  6. filter_search_results()  → role-permitted subset
  7. reranker.rerank()        → cross-encoder scores [0, 1]  (if enabled)
  8. compute_confidence()     → single confidence value + label
  9. Claude API (streaming)   → answer with citations
 10. QueryCache.set()         → persist result for next identical query
```

---

## Confidence Scoring

```
scores = [chunk.score for chunk in top_k_results]

avg_score   = mean(scores)
consistency = 1 / (1 + std(scores))   # tighter cluster → higher consistency
dominance   = scores[0] - scores[1]   # gap between top and second result

confidence  = 0.5 × avg_score + 0.3 × consistency + 0.2 × dominance
            clamped to [0, 1]

< 0.15  → abstain  ("not enough high-confidence information")
< 0.30  → warn     ("⚠️ This answer may be incomplete or uncertain")
≥ 0.30  → answer normally
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (required) |

All other parameters (model, top-k, reranker on/off) are set at runtime via the Streamlit sidebar or CLI flags.

---

## Running Tests

```bash
pytest tests/ -v
```
