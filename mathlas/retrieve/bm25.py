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

import json
import math
import os
import re
import shutil
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

_TOKEN = re.compile(r"[A-Za-z0-9]+")

#: On-disk cache format tag + version, written into the bundle's ``meta.json``
#: and validated by :meth:`BM25Index.load` so a stale/foreign bundle is refused
#: rather than silently mis-loaded.
_CACHE_FORMAT = "mathlas-bm25"
_CACHE_VERSION = 1


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
        # Top-k via argpartition: a full argsort over a multi-million-doc score
        # vector is O(n log n) per query; partition + boundary-tie resolution +
        # sort-the-k is O(n + k log k). Ties (within the set and at the k-th
        # score boundary) are broken deterministically by ascending doc index
        # --- the previous full argsort used quicksort, whose tie order was
        # arbitrary, so this is both faster and reproducible.
        if k < scores.shape[0]:
            kth = scores[np.argpartition(-scores, k - 1)[:k]].min()  # boundary score
            above = np.flatnonzero(scores > kth)
            ties = np.flatnonzero(scores == kth)[: k - above.size]   # lowest-index ties
            cand = np.concatenate([above, ties])
            order = cand[np.lexsort((cand, -scores[cand]))]
        else:
            order = np.arange(scores.shape[0])
            order = order[np.lexsort((order, -scores))][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0.0]

    # ------------------------------------------------------------------ #
    # Persistence: serialize the built index so it can be reloaded WITHOUT
    # re-tokenizing the corpus. The build (tokenize every doc + assemble the
    # CSR term-document matrix) is the slow part; over a multi-million-doc
    # corpus it costs minutes + several GB of RAM. The cache turns every
    # subsequent load into a few fast array reads.
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> str:
        """Serialize this index to ``path`` (a self-contained directory bundle).

        Four artifacts are written, exactly the pieces :meth:`load` needs to
        rebuild the object with no tokenization:

          * ``tf.npz``     -- the term-frequency matrix (``scipy.sparse``);
          * ``arrays.npz`` -- the ``idf`` and ``doc_len`` dense vectors;
          * ``vocab.json`` -- the ``{token: column-index}`` map;
          * ``meta.json``  -- the ``k1``/``b``/``n``/``avgdl`` scalars plus a
                               format+version tag.

        The bundle is written atomically (a temp dir is fully populated, then
        ``os.replace``d onto ``path``) so a reader never observes a partially
        written cache. Returns the bundle path.
        """
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        try:
            # tf is the big artifact; store it uncompressed so LOADS stay fast
            # (load speed is the whole point of the cache). scipy preserves the
            # CSC layout, which ``search`` relies on for fast column slicing.
            save_npz(os.path.join(tmp, "tf.npz"), self.tf, compressed=False)
            np.savez(os.path.join(tmp, "arrays.npz"),
                     idf=self.idf, doc_len=self.doc_len)
            with open(os.path.join(tmp, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False)
            meta = {
                "format": _CACHE_FORMAT,
                "version": _CACHE_VERSION,
                "k1": float(self.k1),
                "b": float(self.b),
                "n": int(self.n),
                "avgdl": float(self.avgdl),
                "vocab_size": len(self.vocab),
                "nnz": int(self.tf.nnz),
            }
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            shutil.rmtree(path, ignore_errors=True)
            os.replace(tmp, path)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return path

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Reconstruct a :class:`BM25Index` from a :meth:`save` bundle.

        The inverse of :meth:`save`: the tokenized corpus is NOT touched -- the
        precomputed ``tf``/``idf``/``doc_len``/``vocab`` are read straight back,
        and ``__init__`` is bypassed so no document is re-tokenized. A foreign or
        version-mismatched bundle is refused with a clear error.
        """
        path = os.fspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"BM25 cache bundle not found (expected a directory): {path}")
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != _CACHE_FORMAT or meta.get("version") != _CACHE_VERSION:
            raise ValueError(
                f"incompatible BM25 cache at {path}: "
                f"format={meta.get('format')!r} version={meta.get('version')!r} "
                f"(expected {_CACHE_FORMAT!r} v{_CACHE_VERSION})")
        tf = load_npz(os.path.join(path, "tf.npz")).tocsc()
        arr = np.load(os.path.join(path, "arrays.npz"))
        with open(os.path.join(path, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)

        obj = cls.__new__(cls)            # bypass __init__ -> NO re-tokenization
        obj.k1 = float(meta["k1"])
        obj.b = float(meta["b"])
        obj.n = int(meta["n"])
        obj.avgdl = float(meta["avgdl"])
        obj.vocab = vocab
        obj.tf = tf
        obj.idf = np.asarray(arr["idf"], dtype=np.float32)
        obj.doc_len = np.asarray(arr["doc_len"], dtype=np.float32)
        # cheap structural sanity: the serialized shapes must agree, else the
        # bundle is corrupt and ``search`` would index out of bounds.
        v = max(len(vocab), 1)
        if obj.tf.shape != (obj.n, v) or obj.idf.shape[0] != v \
                or obj.doc_len.shape[0] != obj.n:
            raise ValueError(
                f"corrupt BM25 cache at {path}: array shapes disagree "
                f"(tf={obj.tf.shape}, n={obj.n}, vocab={len(vocab)}, "
                f"idf={obj.idf.shape}, doc_len={obj.doc_len.shape})")
        return obj


__all__ = ["BM25Index", "tokenize"]
