"""
qa_pipeline.py

Orchestrates the end-to-end Permission-Aware QA flow:
  1. Embed the incoming query.
  2. Retrieve permission-filtered candidate chunks.
  3. Generate a grounded answer.
  4. Return a structured response with citations and audit metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ingestion.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.answer_generator import AnswerGenerator, GeneratedAnswer


@dataclass
class QARequest:
    """Encapsulates a single QA request from a user."""

    query: str
    user_context: dict[str, Any]   # Identity + attributes used for access control
    session_id: str = ""


@dataclass
class QAResponse:
    """Full response returned to the caller after the pipeline completes."""

    query: str
    answer: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    permitted_chunk_count: int
    total_candidate_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class QAPipeline:
    """End-to-end orchestrator for permission-aware question answering."""

    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
    ) -> None:
        """
        Args:
            embedder:          Converts the query string into a dense vector.
            retriever:         Returns permission-filtered relevant chunks.
            answer_generator:  Generates a grounded answer from those chunks.
        """
        self.embedder = embedder
        self.retriever = retriever
        self.answer_generator = answer_generator

    def run(self, request: QARequest) -> QAResponse:
        """Execute the full QA pipeline for a single request.

        Args:
            request: Incoming QA request including query and user context.

        Returns:
            QAResponse with the generated answer, citations, and audit fields.
        """
        raise NotImplementedError

    def run_batch(self, requests: list[QARequest]) -> list[QAResponse]:
        """Process multiple QA requests, useful for evaluation runs.

        Args:
            requests: List of QARequest objects.

        Returns:
            List of QAResponse objects in the same order.
        """
        raise NotImplementedError
