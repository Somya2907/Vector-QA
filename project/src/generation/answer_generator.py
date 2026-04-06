"""
answer_generator.py

Constructs a grounded answer from retrieved context chunks using an LLM.
Manages prompt construction, context window budgeting, and citation tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieval.vector_store import SearchResult
from .llm_client import LLMClient, LLMResponse


@dataclass
class GeneratedAnswer:
    """The final answer produced for a user query."""

    query: str
    answer: str
    citations: list[str]          # chunk_ids used as grounding evidence
    model: str
    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class AnswerGenerator:
    """Generates a cited, grounded answer from retrieved chunks."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_context_chunks: int = 5,
        system_prompt: str = "",
    ) -> None:
        """
        Args:
            llm_client:          Configured LLM client to use for generation.
            max_context_chunks:  Maximum number of retrieved chunks to include
                                 in the prompt context window.
            system_prompt:       Base system prompt injected before the context.
        """
        self.llm_client = llm_client
        self.max_context_chunks = max_context_chunks
        self.system_prompt = system_prompt

    def generate(
        self,
        query: str,
        context_chunks: list[SearchResult],
    ) -> GeneratedAnswer:
        """Generate a grounded answer for the query given the context chunks.

        Args:
            query:          The user's original question.
            context_chunks: Permission-filtered retrieved chunks to use as context.

        Returns:
            GeneratedAnswer with the response text and citation metadata.
        """
        raise NotImplementedError

    def build_prompt(
        self,
        query: str,
        context_chunks: list[SearchResult],
    ) -> list[dict[str, str]]:
        """Construct the messages list to send to the LLM.

        Args:
            query:          The user's question.
            context_chunks: Chunks to include as grounding context.

        Returns:
            Formatted messages list ready for LLMClient.complete().
        """
        raise NotImplementedError
