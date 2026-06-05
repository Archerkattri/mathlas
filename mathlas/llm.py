"""Pluggable LLM backend — the reasoning "brain" the map/verify steps use.

math_engine is provider-agnostic: it does NOT hardwire any model vendor and does
NOT depend on any third-party math system. A caller supplies an LLM (Anthropic,
OpenAI, a local model, or the EchoLLM stub) implementing `complete()`. The
optional Anthropic backend is loaded only if explicitly constructed.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLM(ABC):
    """Minimal text-in / text-out interface for the reasoning steps."""

    @abstractmethod
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 2048, temperature: float = 0.0) -> str:
        ...


class EchoLLM(LLM):
    """No-op stub so the pipeline runs end-to-end offline / in tests. Returns an
    empty JSON object, which the map step reads as 'no applicable mapping'."""

    def complete(self, prompt, *, system=None, max_tokens=2048, temperature=0.0):
        return "{}"


class AnthropicLLM(LLM):
    """Optional Anthropic-backed brain. Requires `anthropic` + an API key."""

    def __init__(self, model: str = "claude-opus-4-8", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'mathlas[llm]' for the Anthropic backend") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self._model = model

    def complete(self, prompt, *, system=None, max_tokens=2048, temperature=0.0):
        msg = self._client.messages.create(
            model=self._model, max_tokens=max_tokens, temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", None) == "text")
