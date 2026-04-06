# Permission-Aware Vector QA System

A production-style Retrieval-Augmented Generation (RAG) system where every retrieved chunk is filtered through a fine-grained access-control layer before being passed to the LLM.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     embed query     ┌─────────────┐
│  QAPipeline │ ──────────────────► │   Embedder  │
└─────────────┘                     └─────────────┘
    │                                      │ query vector
    │                                      ▼
    │                             ┌──────────────────┐
    │                             │   VectorStore    │  ◄── upsert (ingestion)
    │                             │  (similarity     │
    │                             │   search)        │
    │                             └──────────────────┘
    │                                      │ top-N candidates
    │                                      ▼
    │                             ┌──────────────────┐
    │                             │  AccessControl   │
    │                             │  (PolicyEngine)  │
    │                             └──────────────────┘
    │                                      │ permitted chunks only
    │                                      ▼
    │                             ┌──────────────────┐
    └──────────────────────────── │ AnswerGenerator  │
                                  │  (LLMClient)     │
                                  └──────────────────┘
                                          │
                                          ▼
                                     QAResponse
                                  (answer + citations)
```

---

## Project Structure

```
project/
├── data/                   # Raw and processed documents (gitignored)
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py   # Load docs from filesystem / S3
│   │   ├── chunker.py           # Split docs into overlapping chunks
│   │   └── embedder.py          # Generate dense embeddings
│   ├── retrieval/
│   │   ├── vector_store.py      # Vector DB abstraction (upsert / search)
│   │   └── retriever.py         # Permission-filtered retrieval
│   ├── permissions/
│   │   ├── access_control.py    # Per-chunk access decisions
│   │   └── policy_engine.py     # Policy definition & evaluation
│   ├── generation/
│   │   ├── llm_client.py        # LLM API wrapper (Claude)
│   │   └── answer_generator.py  # Prompt construction & answer generation
│   ├── pipeline/
│   │   └── qa_pipeline.py       # End-to-end orchestrator
│   ├── evaluation/
│   │   └── evaluator.py         # Retrieval & answer quality metrics
│   └── utils/
│       ├── config.py            # Typed settings via pydantic-settings
│       └── logger.py            # Structured JSON logging via structlog
├── app/
│   ├── main.py                  # FastAPI app factory + health endpoint
│   └── routes.py                # REST API endpoints (/ask, /ingest, ...)
├── scripts/
│   ├── ingest.py                # CLI: ingest documents into vector store
│   └── evaluate.py              # CLI: run evaluation suite
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY, vector store credentials, etc.
```

### 3. Ingest documents

```bash
python scripts/ingest.py --source ./data/raw --recursive \
  --permission-tags department=engineering access_level=internal
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload
# API docs available at http://localhost:8000/docs
```

### 5. Run the evaluation suite

```bash
python scripts/evaluate.py --dataset ./data/eval_set.jsonl --top-k 5
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Permission tags stored on chunks | Enables chunk-level access control, not just document-level |
| Default-deny policy model | All policies must permit; a single deny blocks access |
| Permission filtering before generation | LLM never sees unauthorised content |
| Structured logging (structlog) | Machine-parseable JSON logs for observability pipelines |
| Pydantic Settings | Type-safe config with environment variable overrides |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (required) |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model ID for answer generation |
| `EMBEDDING_MODEL` | `voyage-3` | Embedding model identifier |
| `VECTOR_STORE_BACKEND` | `chromadb` | Vector DB backend |
| `RETRIEVAL_TOP_K` | `10` | Candidates fetched before permission filtering |
| `CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Running Tests

```bash
pytest tests/ -v
```
