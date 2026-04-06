"""
rag_pipeline.py

End-to-end Retrieval-Augmented Generation pipeline.

Flow
----
  query + user_role
      │
      ▼
  Retriever.retrieve()       ← embed query, FAISS search, permission filter
      │  list[RetrievalResult]
      ▼
  build_prompt()             ← number context chunks, format user message
      │  system + messages
      ▼
  ClaudeClient.complete()    ← generate grounded answer (TTFT measured)
      │  ClaudeResponse
      ▼
  extract_citations()        ← map [N] references → source metadata
      │
      ▼
  RAGResponse                ← answer + citations + latency

Prompt contract
---------------
Chunks are numbered [1], [2], … in the user message.  Claude is instructed
to cite by number.  extract_citations() scans the response for [N] patterns
and maps them back to the chunk metadata, so every citation carries a
filename and section title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.retrieval.retriever import Retriever, RetrievalResult
from src.generation.claude_client import ClaudeClient, ClaudeResponse

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a helpful assistant for the GitLab handbook.

Rules:
- Answer using ONLY the information provided in the context below.
- If the context does not contain enough information, say exactly:
  "I don't have enough information in the provided context to answer this question."
- Cite your sources inline using the reference numbers [1], [2], etc.
- Be concise, accurate, and professional.
- Do not invent facts or reference knowledge outside the context.\
"""

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A source reference extracted from the generated answer."""

    index: int        # The [N] number used in the answer text
    filename: str
    section: str
    chunk_id: str
    score: float      # Retrieval similarity score of the cited chunk


@dataclass
class RAGResponse:
    """Complete result of a single RAG pipeline run."""

    query: str
    answer: str
    citations: list[Citation]
    chunks: list[RetrievalResult]   # The actual retrieved chunks sent as context
    chunks_used: int                # len(chunks) — kept for quick access
    model: str
    input_tokens: int
    output_tokens: int
    ttft_seconds: float
    total_seconds: float
    user_role: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"RAGResponse("
            f"chunks={self.chunks_used}, "
            f"citations={len(self.citations)}, "
            f"tokens={self.input_tokens}→{self.output_tokens}, "
            f"ttft={self.ttft_seconds:.2f}s, "
            f"total={self.total_seconds:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_context_block(chunks: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is presented as:
        [N] Source: filename | Section: title
        ---
        ...chunk text...

    Args:
        chunks: Permission-filtered retrieved chunks, ranked by similarity.

    Returns:
        Multi-line string ready to embed in the user message.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", "unknown")
        section = chunk.metadata.get("section", "")
        header = f"[{i}] Source: {filename}"
        if section:
            header += f" | Section: {section}"
        parts.append(f"{header}\n---\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def build_messages(query: str, context_block: str) -> list[dict[str, str]]:
    """Build the messages list to send to Claude.

    Args:
        query:         The user's raw question.
        context_block: Pre-formatted numbered context from build_context_block().

    Returns:
        Single-turn messages list in Anthropic format.
    """
    user_content = (
        f"Here is the relevant context from the GitLab handbook:\n\n"
        f"{context_block}\n\n"
        f"---\n"
        f"Question: {query}\n\n"
        f"Answer the question using ONLY the context above. "
        f"Cite sources with [1], [2], etc."
    )
    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citations(
    answer: str,
    chunks: list[RetrievalResult],
) -> list[Citation]:
    """Map [N] references in the answer text back to chunk metadata.

    Only indices that actually appear in the answer AND correspond to a chunk
    are included.  Duplicate references to the same chunk are deduplicated.

    Args:
        answer: Generated answer text from Claude.
        chunks: Context chunks in the same numbered order as the prompt.

    Returns:
        Ordered, deduplicated list of Citation objects.
    """
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1))
        if idx in seen:
            continue
        seen.add(idx)

        # idx is 1-based; chunks list is 0-based
        if 1 <= idx <= len(chunks):
            chunk = chunks[idx - 1]
            citations.append(
                Citation(
                    index=idx,
                    filename=chunk.metadata.get("filename", "unknown"),
                    section=chunk.metadata.get("section", ""),
                    chunk_id=chunk.metadata.get("chunk_id", ""),
                    score=chunk.score,
                )
            )

    return citations


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Permission-aware RAG pipeline: retrieve → prompt → generate."""

    def __init__(
        self,
        retriever: Retriever,
        claude_client: ClaudeClient,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        """
        Args:
            retriever:      Handles query embedding, FAISS search, and access
                            control filtering in one call.
            claude_client:  Configured Claude API client.
            system_prompt:  System instructions prepended to every request.
                            Override to customise tone, language, or domain.
        """
        self.retriever = retriever
        self.claude = claude_client
        self.system_prompt = system_prompt

    def run(
        self,
        query: str,
        user_role: str | None = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline for one query.

        Args:
            query:     The user's natural-language question.
            user_role: Role used for permission filtering.  Pass None to
                       return all chunks regardless of role.

        Returns:
            RAGResponse with the answer, citations, and latency data.
            If no chunks are retrieved, the answer explains that.
        """
        # ── 1. Retrieve ───────────────────────────────────────────────────────
        chunks: list[RetrievalResult] = self.retriever.retrieve(
            query, user_role=user_role
        )

        if not chunks:
            return RAGResponse(
                query=query,
                answer=(
                    "I don't have enough information in the provided context "
                    "to answer this question."
                ),
                citations=[],
                chunks=[],
                chunks_used=0,
                model=self.claude.model,
                input_tokens=0,
                output_tokens=0,
                ttft_seconds=0.0,
                total_seconds=0.0,
                user_role=user_role,
            )

        # ── 2. Build prompt ───────────────────────────────────────────────────
        context_block = build_context_block(chunks)
        messages = build_messages(query, context_block)

        # ── 3. Generate ───────────────────────────────────────────────────────
        response: ClaudeResponse = self.claude.complete(
            messages=messages,
            system=self.system_prompt,
        )

        # ── 4. Extract citations ──────────────────────────────────────────────
        citations = extract_citations(response.content, chunks)

        return RAGResponse(
            query=query,
            answer=response.content,
            citations=citations,
            chunks=chunks,
            chunks_used=len(chunks),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            ttft_seconds=response.ttft_seconds,
            total_seconds=response.total_seconds,
            user_role=user_role,
        )

    def run_stream(
        self,
        query: str,
        user_role: str | None = None,
        on_token: callable = None,
    ) -> RAGResponse:
        """Like run(), but streams tokens via the on_token callback.

        Useful for API endpoints that want to forward tokens to the client
        as they arrive while still returning the full RAGResponse at the end.

        Args:
            query:     The user's question.
            user_role: Role for permission filtering.
            on_token:  Called with each text token as it is generated.

        Returns:
            RAGResponse with the complete assembled answer.
        """
        chunks = self.retriever.retrieve(query, user_role=user_role)

        if not chunks:
            no_context_answer = (
                "I don't have enough information in the provided context "
                "to answer this question."
            )
            if on_token:
                on_token(no_context_answer)
            return RAGResponse(
                query=query,
                answer=no_context_answer,
                citations=[],
                chunks=[],
                chunks_used=0,
                model=self.claude.model,
                input_tokens=0,
                output_tokens=0,
                ttft_seconds=0.0,
                total_seconds=0.0,
                user_role=user_role,
            )

        context_block = build_context_block(chunks)
        messages = build_messages(query, context_block)

        response = self.claude.stream(
            messages=messages,
            system=self.system_prompt,
            on_token=on_token,
        )
        citations = extract_citations(response.content, chunks)

        return RAGResponse(
            query=query,
            answer=response.content,
            citations=citations,
            chunks=chunks,
            chunks_used=len(chunks),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            ttft_seconds=response.ttft_seconds,
            total_seconds=response.total_seconds,
            user_role=user_role,
        )
