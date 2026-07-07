"""Pluggable dense-embedding backend for semantic retrieval.

Mirrors the ``llm.py`` design: the embedder is an interface, not a hardwired
vendor. The PRODUCTION path is an open-weights instruction-tuned text embedder
(Qwen3-Embedding, the current open MTEB SOTA — Zhang et al., arXiv:2506.05176);
the DEFAULT path needs no model download so validation runs CPU-only/offline.

Why Qwen3-Embedding for the production index
--------------------------------------------
Qwen3-Embedding (0.6B/4B/8B; 1024/2560/4096-dim, Matryoshka-truncatable) tops
MTEB as of mid-2026 and is exactly what TheoremSearch independently converged on
for the same corpus — but we build OUR OWN index with it, over the open dataset.
It is *instruction-aware*: prefixing the query with a task instruction gives a
documented +1-5% on retrieval, so ``Qwen3Embedder`` adds that prefix to queries
(never to documents), per the model card.

The corpus unit we embed is the natural-language *slogan/denotation* of a
theorem, NOT raw LaTeX (the load-bearing lesson: symbol-heavy LaTeX drowns dense
embedders; mathematicians query in prose). See ``retrieve/corpus.py``.

References
----------
- Qwen3-Embedding: Zhang et al., "Qwen3 Embedding," arXiv:2506.05176 (2026).
- MTEB: Muennighoff et al., "MTEB: Massive Text Embedding Benchmark," 2023.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np


def _stable_hash(token: str, salt: str) -> int:
    """Deterministic, process-independent string hash.

    Python's builtin ``hash()`` on str is salted per process (PYTHONHASHSEED),
    so the same token maps to different buckets/signs after a restart. A
    persisted HashingEmbedder index (stored doc vectors built in one process,
    query vectors in another) would then land documents and queries in
    different feature spaces, silently breaking retrieval. We hash with
    ``blake2b`` (stdlib, no new dep) over a fixed salt to get a stable int.
    """
    digest = hashlib.blake2b(f"{salt}\x00{token}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big")


class Embedder(ABC):
    """Text -> unit-norm dense vectors. Documents and queries embed the same
    way unless a backend opts to instruction-prefix queries (Qwen3 does)."""

    #: dimensionality of the returned vectors
    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of L2-normalised rows."""
        ...


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class HashingEmbedder(Embedder):
    """Zero-dependency, zero-download fallback embedder (the DEFAULT).

    A deterministic word-hashing bag-of-words projected to ``dim`` and
    L2-normalised. It is intentionally weak -- it exists so the retrieval +
    fusion + map + verify pipeline can be exercised end-to-end on CPU with no
    model weights. For any real corpus, plug in ``Qwen3Embedder`` (production)
    or another open embedder. We deliberately keep the *dense channel* honest
    by pairing it with BM25 in the hybrid retriever, so a weak dense backend
    still yields a usable system (sparse carries exact-term matches).
    """

    def __init__(self, dim: int = 256, ngram: int = 1) -> None:
        self.dim = int(dim)
        self.ngram = int(ngram)

    def _toks(self, text: str) -> List[str]:
        words = "".join(c.lower() if (c.isalnum() or c.isspace()) else " "
                        for c in text).split()
        if self.ngram <= 1:
            return words
        grams = words[:]
        for n in range(2, self.ngram + 1):
            grams += ["_".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        return grams

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._toks(t):
                h = _stable_hash(tok, "mathlas-embed") % self.dim
                sign = 1.0 if (_stable_hash(tok, "sign") & 1) else -1.0
                out[i, h] += sign
        return _l2norm(out)


class Qwen3Embedder(Embedder):
    """Open-weights production embedder (Qwen3-Embedding via sentence-transformers).

    Lazy-loaded so importing mathlas never pulls torch/transformers unless a
    caller actually constructs this. Heavy on CPU; run on GPU for the full
    index. Validation in this repo uses ``HashingEmbedder`` to stay light.
    """

    # Query-side instruction (documents are embedded bare). Per the Qwen3 card,
    # the instruction goes ONLY on the query and lifts retrieval 1-5%.
    DEFAULT_INSTRUCT = (
        "Given a mathematical problem or a description of a result, retrieve the "
        "statement of the existing theorem or lemma that applies to it."
    )

    def __init__(self, model: str = "Qwen/Qwen3-Embedding-0.6B",
                 dim: Optional[int] = None, device: Optional[str] = None,
                 instruct: Optional[str] = DEFAULT_INSTRUCT) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional heavy dep
            raise ImportError(
                "pip install 'mathlas-mcp[embed]' (sentence-transformers + torch) "
                "for the Qwen3 production embedder; the default HashingEmbedder "
                "needs no extra deps."
            ) from e
        self._st = SentenceTransformer(model, device=device,
                                       truncate_dim=dim)  # MRL truncation if set
        self.dim = int(dim or self._st.get_sentence_embedding_dimension())
        self._instruct = instruct

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        kw = {}
        if is_query and self._instruct:
            kw["prompt"] = f"Instruct: {self._instruct}\nQuery: "
        vecs = self._st.encode(list(texts), normalize_embeddings=True,
                               convert_to_numpy=True, **kw)
        return vecs.astype(np.float32)


__all__ = ["Embedder", "HashingEmbedder", "Qwen3Embedder"]
