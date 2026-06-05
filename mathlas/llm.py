"""Optional bring-your-own-LLM interface — mathlas itself NEVER calls an LLM.

mathlas is a tool that an AI *uses*, not a tool that uses an AI. The reasoning
"brain" is always the CALLER (Claude Code, Cursor, any agent / MCP client). The
core of mathlas — numeric identify/verify, retrieval, provenance, the MCP
server, the scaffolds and checklists — runs with **NO LLM and NO API key**.

This module exists only for the SECONDARY, standalone ``solve()`` convenience
(see solve.py): if a user wants a one-shot library call that also does the
needs<->guarantees reasoning, they may *bring their own* LLM implementing
``complete()``. mathlas ships **no vendor SDK and no default model** — only the
``LLM`` interface and the ``EchoLLM`` no-op stub. Wire in your own provider
(Anthropic, OpenAI, a local model) by subclassing ``LLM`` yourself.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLM(ABC):
    """Minimal text-in / text-out interface for the OPTIONAL standalone path.

    mathlas never constructs one of these on its own. A caller supplies it to
    ``solve()`` only if they want the bring-your-own-LLM convenience. The MCP
    server and the plain library functions do not use it at all.
    """

    @abstractmethod
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 2048, temperature: float = 0.0) -> str:
        ...


class EchoLLM(LLM):
    """No-op stub so the OPTIONAL pipeline runs end-to-end offline / in tests.
    Returns an empty JSON object, which the map step reads as 'no applicable
    mapping'. This is the default for ``solve()`` so the package needs no LLM."""

    def complete(self, prompt, *, system=None, max_tokens=2048, temperature=0.0):
        return "{}"


__all__ = ["LLM", "EchoLLM"]
