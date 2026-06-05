"""ManualRetriever — drive the pipeline with caller-provided candidates.

Lets the map/verify pipeline run end-to-end before our own large-scale index
exists, and is useful when the caller already holds candidate results. Our own
embedding index over open corpora will implement the same `Retriever` interface.
"""
from __future__ import annotations

from typing import List

from . import Candidate, Retriever


class ManualRetriever(Retriever):
    def __init__(self, candidates: List[Candidate]):
        self._candidates = list(candidates)

    def retrieve(self, query: str, k: int = 10) -> List[Candidate]:
        return self._candidates[:k]
