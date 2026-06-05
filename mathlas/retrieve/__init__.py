"""Retrieval — OUR OWN sources of candidate existing math.

math_engine builds its OWN retrieval and does NOT depend on any third-party
running service at runtime. In particular we do NOT call TheoremSearch's API or
use their code — that would make the tool a wrapper, not an independent system.
TheoremSearch is REFERENCE ONLY. The single thing we reuse is their openly
licensed (CC-BY/CC0) released *dataset*, as raw input data, to build our own
index — applying the lesson we learned (embed a natural-language description,
not raw LaTeX) with our own implementation and our own models.

A `Retriever` maps a query to candidate existing results. Concrete
implementations live alongside this module:
  * `hybrid.HybridRetriever` — our own dense (Qwen3) + BM25 index, fused by RRF,
    over `Document`s built from the open dataset (`corpus.load_documents`).
  * `manual.ManualRetriever` — caller-provided candidates (for tests / when the
    candidates are already in hand).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Candidate:
    """A candidate existing result surfaced by retrieval (source-agnostic)."""
    statement: str                  # the result, as text
    name: Optional[str] = None
    source: Optional[str] = None    # citation / URL / identifier
    score: Optional[float] = None
    meta: Optional[dict] = None


class Retriever(ABC):
    """Maps a query to candidate existing results. Our own implementations only."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Candidate]:
        ...


__all__ = ["Candidate", "Retriever"]
