"""HybridRetriever -- OUR OWN semantic index: dense + BM25 fused by RRF.

This is math_engine's retrieval, built and served by us. Two channels:
  * DENSE  -- an ``Embedder`` (Qwen3-Embedding in production) over each
              document's NL slogan: captures meaning / paraphrase / cross-naming.
  * SPARSE -- BM25 over name+slogan+statement: captures exact symbol/term hits a
              dense model drifts past.

They are combined by **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher,
SIGIR 2009): ``score(d) = sum_channels 1/(rrf_k + rank_d)``. RRF is unsupervised,
needs no score normalisation, and is the consistently-best simple fusion -- the
right choice when one channel (here the default HashingEmbedder) may be weak.

For the validation subset the dense matrix is a plain NumPy array and search is
an exact dot-product top-k (no ANN/faiss needed at this scale). The full-corpus
build (a documented offline GPU script) is where ANN + quantisation come in.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from . import Candidate, Retriever
from .bm25 import BM25Index
from .corpus import Document
from ..embed import Embedder, HashingEmbedder


def rrf_fuse(rankings: Sequence[Sequence[int]], rrf_k: int = 60) -> dict:
    """Reciprocal Rank Fusion. Each ``ranking`` is doc-ids best-first. Returns
    ``{doc_id: fused_score}``."""
    fused: dict = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused


class HybridRetriever(Retriever):
    """Dense + BM25 + RRF over a ``Document`` corpus we built ourselves."""

    def __init__(self, documents: Sequence[Document],
                 embedder: Optional[Embedder] = None,
                 rrf_k: int = 60, channel_depth: int = 50) -> None:
        self.docs: List[Document] = list(documents)
        self.embedder = embedder or HashingEmbedder()
        self.rrf_k = int(rrf_k)
        self.channel_depth = int(channel_depth)

        self._bm25 = BM25Index([d.sparse_text for d in self.docs])
        if self.docs:
            self._emb = self.embedder.encode([d.embed_text for d in self.docs],
                                             is_query=False)
        else:
            self._emb = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def _dense_rank(self, query: str, k: int) -> List[int]:
        if not self.docs:
            return []
        q = self.embedder.encode([query], is_query=True)[0]
        sims = self._emb @ q                      # rows are unit-norm -> cosine
        return [int(i) for i in np.argsort(-sims)[:k]]

    def retrieve(self, query: str, k: int = 10) -> List[Candidate]:
        depth = max(k, self.channel_depth)
        dense = self._dense_rank(query, depth)
        sparse = [i for i, _ in self._bm25.search(query, depth)]
        fused = rrf_fuse([dense, sparse], rrf_k=self.rrf_k)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: List[Candidate] = []
        for doc_id, score in top:
            d = self.docs[doc_id]
            out.append(Candidate(
                statement=d.statement,
                name=d.name,
                source=d.source,
                score=float(score),
                meta={"slogan": d.slogan, "title": d.title, "doc_id": d.doc_id},
            ))
        return out


__all__ = ["HybridRetriever", "rrf_fuse"]
