"""
claude_client.py

Anthropic Claude API client with streaming support and latency instrumentation.

Latency metrics
---------------
- TTFT (time-to-first-token): wall-clock seconds from request dispatch until
  the first text token arrives.  Measured via the streaming API even for the
  blocking complete() path, so the number is always meaningful.
- Total latency: wall-clock seconds from request dispatch until the full
  response is received and assembled.

Both are included in every ClaudeResponse.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env if present

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0   # Deterministic by default for RAG


@dataclass
class ClaudeResponse:
    """Structured response from a Claude API call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    ttft_seconds: float    # Seconds until the first token arrived
    total_seconds: float   # Total wall-clock latency for the full response


class ClaudeClient:
    """Thin, reusable wrapper around the Anthropic messages API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: str | None = None,
    ) -> None:
        """
        Args:
            model:       Claude model ID (e.g. "claude-sonnet-4-6").
            max_tokens:  Maximum tokens to generate in a single response.
            temperature: Sampling temperature.  0.0 = deterministic.
            api_key:     Anthropic API key.  Falls back to ANTHROPIC_API_KEY env var.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    # ------------------------------------------------------------------
    # Blocking call (uses streaming internally to capture TTFT)
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> ClaudeResponse:
        """Send a prompt and return the full response with latency metrics.

        Uses the streaming API internally so that TTFT is accurately measured
        even though this method blocks until the entire response is ready.

        Args:
            messages: Conversation in [{"role": "user"|"assistant", "content": "..."}]
                      format.
            system:   Optional system prompt.

        Returns:
            ClaudeResponse with the generated text and timing data.
        """
        kwargs = self._build_kwargs(messages, system)

        t_start = time.perf_counter()
        ttft: float | None = None
        tokens: list[str] = []

        with self._client.messages.stream(**kwargs) as stream:
            for token in stream.text_stream:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                tokens.append(token)
            final_msg = stream.get_final_message()

        total = time.perf_counter() - t_start

        return ClaudeResponse(
            content="".join(tokens),
            model=final_msg.model,
            input_tokens=final_msg.usage.input_tokens,
            output_tokens=final_msg.usage.output_tokens,
            stop_reason=str(final_msg.stop_reason or "end_turn"),
            ttft_seconds=ttft if ttft is not None else total,
            total_seconds=total,
        )

    # ------------------------------------------------------------------
    # Streaming call — yields tokens, returns ClaudeResponse at the end
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        on_token: Callable[[str], None] | None = None,
    ) -> ClaudeResponse:
        """Stream a response token-by-token and return the final ClaudeResponse.

        Useful for APIs and UIs that want to forward tokens to the client as
        they arrive.

        Args:
            messages:  Conversation in role/content format.
            system:    Optional system prompt.
            on_token:  Callback invoked with each text token as it arrives.
                       Use this to forward tokens to a client or print them.

        Returns:
            ClaudeResponse with the complete assembled text and timing data.
        """
        kwargs = self._build_kwargs(messages, system)

        t_start = time.perf_counter()
        ttft: float | None = None
        tokens: list[str] = []

        with self._client.messages.stream(**kwargs) as stream:
            for token in stream.text_stream:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                tokens.append(token)
                if on_token is not None:
                    on_token(token)
            final_msg = stream.get_final_message()

        total = time.perf_counter() - t_start

        return ClaudeResponse(
            content="".join(tokens),
            model=final_msg.model,
            input_tokens=final_msg.usage.input_tokens,
            output_tokens=final_msg.usage.output_tokens,
            stop_reason=str(final_msg.stop_reason or "end_turn"),
            ttft_seconds=ttft if ttft is not None else total,
            total_seconds=total,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kwargs(
        self, messages: list[dict[str, str]], system: str
    ) -> dict:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        return kwargs
