"""BM25 sparse retrieval (pure-Python/NumPy, Okapi BM25).

The sparse half of our hybrid retriever. BM25 is exact-term-aware, which matters
for math: a query naming a specific operator/space/constant ("etale", "PSLQ",
"Golod-Shafarevich") must hit documents containing that token, where a dense
embedder can semantically drift. Dense + sparse fused by Reciprocal Rank Fusion
is the robust 2025-2026 retrieval pattern (Cormack et al. RRF; widely confirmed
to beat either channel alone, no score normalisation needed).

No external IR engine: a CSR term-document matrix + the standard Okapi formula,
which is plenty for the per-domain subset indexes math_engine builds.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens. LaTeX punctuation/backslashes split words,
    so ``\\mathbb{Z}`` -> ``mathbb``, ``z`` -- crude but effective for term hits."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over an in-memory corpus."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = float(k1), float(b)
        toks = [tokenize(d) for d in docs]
        self.n = len(toks)
        self.doc_len = np.array([len(t) for t in toks], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0

        vocab: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        vals: List[int] = []
        for di, t in enumerate(toks):
            counts: Dict[int, int] = {}
            for w in t:
                j = vocab.setdefault(w, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(di)
                cols.append(j)
                vals.append(c)
        self.vocab = vocab
        V = max(len(vocab), 1)
        # term-frequency matrix (docs x vocab), CSC for fast column slicing.
        self.tf = csr_matrix((vals, (rows, cols)), shape=(self.n, V),
                             dtype=np.float32).tocsc()
        # document frequency per term -> idf (BM25 '+1' variant, always >0).
        df = np.diff(self.tf.indptr)
        self.idf = np.log(1.0 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)

    def search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """Return up to ``k`` ``(doc_index, bm25_score)`` pairs, best first."""
        if self.n == 0:
            return []
        scores = np.zeros(self.n, dtype=np.float32)
        denom_len = self.k1 * (1.0 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for w in set(tokenize(query)):
            j = self.vocab.get(w)
            if j is None:
                continue
            col = self.tf.getcol(j)
            idx = col.indices
            tf = col.data
            scores[idx] += self.idf[j] * (tf * (self.k1 + 1.0)) / (tf + denom_len[idx])
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0.0]


__all__ = ["BM25Index", "tokenize"]
