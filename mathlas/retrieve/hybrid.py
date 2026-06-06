"""HybridRetriever -- OUR OWN semantic index: dense + BM25 fused by RRF.

This is math_engine's retrieval, built and served by us. Two channels:
  * DENSE  -- an ``Embedder`` (Qwen3-Embedding in production) over each
              document's NL slogan: captures meaning / paraphrase / cross-naming.
  * SPARSE -- BM25 over name+slogan+statement+label: captures exact symbol/term
              hits a dense model drifts past (the edge over slogan-only dense).

They are combined by **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher,
SIGIR 2009): ``score(d) = sum_channels 1/(rrf_k + rank_d)``. RRF is unsupervised,
needs no score normalisation, and is the consistently-best simple fusion -- the
right choice when one channel (here the default HashingEmbedder) may be weak.

OPTIONAL citation-weighted rerank: ``score = fused + lambda*log(max(cites,1))``
adds a light "the field already leans on this result" prior. Default ``lambda=0``
(pure fusion, no behaviour change) -- it is opt-in because the published target
is a pure-retrieval metric and a citation prior must never silently regress it.

The index can be either built in-process (embed the corpus now) or LOADED from a
prebuilt ``index.npz`` via :meth:`HybridRetriever.from_index` -- the dense matrix
is precomputed (e.g. the offline Qwen3-Embedding-8B pass), BM25 is rebuilt from
the stored meta at load, and only the *query* is embedded at query time. For the
validation subset the dense matrix is a plain NumPy array and search is an exact
dot-product top-k (no ANN/faiss needed at this scale); the full-corpus build is
where ANN + quantisation come in.
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Sequence

import numpy as np

from . import Candidate, Retriever
from .bm25 import BM25Index
from .corpus import Document, doc_from_meta
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
    """Dense + BM25 + RRF over a ``Document`` corpus we built ourselves.

    ``citation_lambda`` (default 0.0) optionally adds ``lambda*log(max(cites,1))``
    to each fused score before the final top-k cut -- a small, opt-in prior. With
    the default 0.0 the ranking is pure dense+BM25 RRF (no regression risk).
    """

    def __init__(self, documents: Sequence[Document],
                 embedder: Optional[Embedder] = None,
                 rrf_k: int = 60, channel_depth: int = 50,
                 citation_lambda: float = 0.0,
                 _precomputed_emb: Optional[np.ndarray] = None,
                 _faiss_index=None, build_bm25: bool = True) -> None:
        self.docs: List[Document] = list(documents)
        self.embedder = embedder or HashingEmbedder()
        self.rrf_k = int(rrf_k)
        self.channel_depth = int(channel_depth)
        self.citation_lambda = float(citation_lambda)

        # SPARSE channel: BM25 over sparse_text. Optional -- off for very large faiss-served
        # indexes where an in-memory inverted index over every doc is impractical (dense-only).
        self._bm25 = BM25Index([d.sparse_text for d in self.docs]) if build_bm25 else None

        # DENSE channel, exactly one mode: a prebuilt FAISS ANN index (the 10M+ scale path, query
        # searched against it), a precomputed flat matrix (from_index, exact dot-product), or
        # embed-the-corpus-now.
        self._faiss = _faiss_index
        if _faiss_index is not None:
            self._emb = None
        elif _precomputed_emb is not None:
            emb = np.asarray(_precomputed_emb)
            if emb.shape[0] != len(self.docs):
                raise ValueError(
                    f"precomputed matrix rows ({emb.shape[0]}) != docs "
                    f"({len(self.docs)})")
            # keep float32 unit-norm rows for an exact cosine dot-product.
            self._emb = _ensure_unit_rows(emb.astype(np.float32))
        elif self.docs:
            self._emb = self.embedder.encode([d.embed_text for d in self.docs],
                                             is_query=False).astype(np.float32)
        else:
            self._emb = np.zeros((0, self.embedder.dim), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Load a prebuilt index (precomputed dense matrix + meta) -> serve.
    # ------------------------------------------------------------------ #
    @classmethod
    def from_index(cls, path: str, embedder: Optional[Embedder] = None,
                   rrf_k: int = 60, channel_depth: int = 50,
                   citation_lambda: float = 0.0,
                   label_in_embed: bool = False) -> "HybridRetriever":
        """Build a retriever from a prebuilt ``index.npz``.

        The npz (written by ``scripts/build_index_mp.py finalize``) holds
        ``{matrix, dim, model, embedder, meta_file}``: ``matrix`` is the (N, dim)
        dense document matrix (e.g. Qwen3-Embedding-8B over the slogans). The
        per-doc meta lives in a sidecar ``<index>.meta.jsonl`` (one JSON record
        per line, row-aligned to the matrix) referenced by ``meta_file`` — the
        full-corpus meta is far too large to store as an npz string scalar. Small
        dev indices may instead carry an in-npz ``meta`` JSON list; both are
        handled. We attach ``matrix`` directly as the dense index (NO re-embedding
        of documents), rebuild BM25 from the meta's ``sparse_text``, and embed
        *queries* with ``embedder`` at query time.

        ``embedder`` MUST be the same model the matrix was built with (so query
        and document vectors live in the same space). On CPU that means the
        Qwen3 query encode runs at query time; pass a matching ``Qwen3Embedder``.
        For a CPU dev-smoke a ``HashingEmbedder`` matrix + the same embedder works.
        """
        data = np.load(path, allow_pickle=True)
        matrix = np.asarray(data["matrix"])
        # Meta source: the full corpus stores a sidecar JSONL (referenced by
        # ``meta_file``, streamed line by line so we never hold a giant string);
        # small dev indices store an in-npz ``meta`` JSON list. Prefer the sidecar.
        sidecar = None
        if "meta_file" in data:
            cand = os.path.join(os.path.dirname(os.fspath(path)), str(data["meta_file"]))
            sidecar = cand if os.path.exists(cand) else None
        if sidecar is None:  # default sibling name, even if meta_file wasn't recorded
            cand = os.path.splitext(os.fspath(path))[0] + ".meta.jsonl"
            sidecar = cand if os.path.exists(cand) else None
        if sidecar is not None:
            docs = []
            with open(sidecar) as mf:
                for line in mf:
                    docs.append(doc_from_meta(json.loads(line), label_in_embed=label_in_embed))
        elif "meta" in data:
            docs = [doc_from_meta(m, label_in_embed=label_in_embed)
                    for m in json.loads(str(data["meta"]))]
        else:
            docs = []
        if matrix.shape[0] != len(docs):
            raise ValueError(
                f"index matrix rows ({matrix.shape[0]}) != meta docs "
                f"({len(docs)}) in {path}")
        retr = cls(docs, embedder=embedder, rrf_k=rrf_k,
                   channel_depth=channel_depth, citation_lambda=citation_lambda,
                   _precomputed_emb=matrix)
        # surface a couple of index facts for introspection / logging.
        retr.index_path = os.fspath(path)
        retr.index_dim = int(data["dim"]) if "dim" in data else matrix.shape[1]
        retr.index_model = str(data["model"]) if "model" in data else None
        if embedder is not None and getattr(embedder, "dim", None) and \
                int(embedder.dim) != int(matrix.shape[1]):
            raise ValueError(
                f"embedder dim ({embedder.dim}) != index dim "
                f"({matrix.shape[1]}); query and doc vectors must match. Use the "
                f"same model the index was built with (index model={retr.index_model}).")
        return retr

    @classmethod
    def from_faiss(cls, faiss_path: str, meta_path: Optional[str] = None,
                   embedder: Optional[Embedder] = None, rrf_k: int = 60,
                   channel_depth: int = 50, citation_lambda: float = 0.0,
                   with_bm25: bool = False, label_in_embed: bool = False) -> "HybridRetriever":
        """Serve a large corpus from a prebuilt FAISS ANN index (the 10M+ scale path).

        ``scripts/index_dolma_corpus.py merge`` writes ``<out>.faiss`` (an IVF+PQ index over the
        union of the 1.34M + dolma vectors; inner-product == cosine on the unit-normalised
        vectors) and a row-aligned ``<out>.meta.jsonl`` sidecar. We load the faiss index for the
        dense channel and the meta for the doc records; a query is embedded, L2-normalised, and
        ANN-searched at query time. BM25 over 10M+ docs is impractical in memory, so it is OFF by
        default (dense-only); pass ``with_bm25=True`` for smaller faiss indexes. ``embedder`` MUST
        be the model the index was built with (query and doc vectors must share a space).
        """
        import faiss
        index = faiss.read_index(os.fspath(faiss_path))
        if meta_path is None:
            meta_path = os.path.splitext(os.fspath(faiss_path))[0] + ".meta.jsonl"
        docs: List[Document] = []
        with open(meta_path) as mf:
            for line in mf:
                docs.append(doc_from_meta(json.loads(line), label_in_embed=label_in_embed))
        if index.ntotal != len(docs):
            raise ValueError(
                f"faiss index ntotal ({index.ntotal}) != meta docs ({len(docs)}) in {faiss_path}")
        retr = cls(docs, embedder=embedder, rrf_k=rrf_k, channel_depth=channel_depth,
                   citation_lambda=citation_lambda, _faiss_index=index, build_bm25=with_bm25)
        retr.index_path = os.fspath(faiss_path)
        retr.index_dim = int(index.d)
        if embedder is not None and getattr(embedder, "dim", None) and \
                int(embedder.dim) != int(index.d):
            raise ValueError(
                f"embedder dim ({embedder.dim}) != faiss index dim ({index.d}); use the same "
                f"model the index was built with.")
        return retr

    def _dense_rank(self, query: str, k: int) -> List[int]:
        if not self.docs:
            return []
        q = self.embedder.encode([query], is_query=True)[0].astype(np.float32)
        if self._faiss is not None:               # ANN path (10M+ scale)
            import faiss
            qn = np.ascontiguousarray(q[None, :])
            faiss.normalize_L2(qn)                 # inner product == cosine on unit vectors
            _, idx = self._faiss.search(qn, k)
            return [int(i) for i in idx[0] if i >= 0]
        sims = self._emb @ q                      # rows are unit-norm -> cosine
        return [int(i) for i in np.argsort(-sims)[:k]]

    def retrieve(self, query: str, k: int = 10, mode: str = "hybrid") -> List[Candidate]:
        """``mode``: ``hybrid`` (dense+BM25+RRF, the default and production path),
        or ``dense``/``sparse`` for single-channel ablation. Single-channel modes
        rank by that channel's RRF-shaped score so the top-k cut is comparable."""
        depth = max(k, self.channel_depth)
        dense = self._dense_rank(query, depth) if mode in ("hybrid", "dense") else []
        sparse = ([i for i, _ in self._bm25.search(query, depth)]
                  if (mode in ("hybrid", "sparse") and self._bm25 is not None) else [])
        if mode == "dense":
            fused = {d: 1.0 / (self.rrf_k + r + 1) for r, d in enumerate(dense)}
        elif mode == "sparse":
            fused = {d: 1.0 / (self.rrf_k + r + 1) for r, d in enumerate(sparse)}
        else:
            fused = rrf_fuse([dense, sparse], rrf_k=self.rrf_k)
        if self.citation_lambda > 0.0:
            lam = self.citation_lambda
            for doc_id in fused:
                c = self.docs[doc_id].citations
                if c and c > 0:
                    fused[doc_id] += lam * math.log(c)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: List[Candidate] = []
        for doc_id, score in top:
            d = self.docs[doc_id]
            out.append(Candidate(
                statement=d.statement,
                name=d.name,
                source=d.source,
                score=float(score),
                meta={"slogan": d.slogan, "title": d.title, "doc_id": d.doc_id,
                      "label": d.label, "citations": d.citations,
                      "category": d.category, "source": d.source},
            ))
        return out


def _ensure_unit_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot-product is cosine. The build writes unit-norm
    rows already, but a fp16->fp32 round-trip can leave tiny drift; renormalise
    defensively (cheap, and makes a hand-built dev matrix safe too)."""
    if x.size == 0:
        return x.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


__all__ = ["HybridRetriever", "rrf_fuse"]
