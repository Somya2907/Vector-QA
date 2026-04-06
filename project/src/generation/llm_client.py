"""
llm_client.py

Thin wrapper around the LLM API (Anthropic Claude by default).
Decouples the rest of the system from provider-specific SDK details and
provides a uniform interface for chat completions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Structured response from an LLM completion call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    raw_response: Any = None   # Provider-specific response object for debugging


class LLMClient:
    """Sends prompts to an LLM and returns structured LLMResponse objects."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        """
        Args:
            model:   Model identifier (e.g. "claude-sonnet-4-6").
            api_key: API key; falls back to the ANTHROPIC_API_KEY env var if omitted.
            kwargs:  Additional provider-specific parameters (temperature, max_tokens, etc.).
        """
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Send a chat-formatted prompt and return the model's response.

        Args:
            messages: List of role/content dicts following the OpenAI/Anthropic
                      messages format, e.g. [{"role": "user", "content": "..."}].

        Returns:
            LLMResponse with the generated text and usage metadata.
        """
        raise NotImplementedError

    def complete_with_system(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        """Convenience wrapper that prepends a system prompt.

        Args:
            system_prompt: Instructions for the model's behaviour and persona.
            messages:      Conversation history in role/content format.

        Returns:
            LLMResponse with the generated text and usage metadata.
        """
        raise NotImplementedError
