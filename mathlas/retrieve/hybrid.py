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

``rrf_k`` defaults to 10 (not the literature-traditional 60): tuned on the full
3.68M production index, n=3000 body-query self-recall (scripts/
eval_retrieval_upgrades.py rrf, 2026-06-09) -- k=10 beat k=60 on every metric
(R@1 0.771 vs 0.764, R@5 0.991 vs 0.963, R@10 0.999 vs 0.991, MRR 0.869 vs
0.855); k=20 sat between. A smaller k weights each channel's top hits harder,
which suits two channels of very different precision profiles.

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
import sys
from typing import List, Optional, Sequence

import numpy as np

from . import Candidate, Retriever
from .bm25 import BM25Index
from .corpus import Document, KNOWN_SOURCE_KEYS, doc_from_meta, source_key
from .rerank import Reranker, doc_rerank_text
from ..embed import Embedder, HashingEmbedder


def rrf_fuse(rankings: Sequence[Sequence[int]], rrf_k: int = 10) -> dict:
    """Reciprocal Rank Fusion. Each ``ranking`` is doc-ids best-first. Returns
    ``{doc_id: fused_score}``."""
    fused: dict = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused


# --------------------------------------------------------------------- #
# Source-aware retrieval plumbing (opt-in; the default path is untouched).
# --------------------------------------------------------------------- #
def _normalize_source_filter(source_filter) -> Optional[tuple]:
    """Validate/normalise a ``source_filter`` into ``(include, exclude)``
    frozensets of canonical source keys, or ``None`` when it is a no-op.

    Accepted form: ``{"include": [...]}, {"exclude": [...]}`` or both (a doc
    passes iff it is in ``include`` -- when given -- AND not in ``exclude``).
    Keys must come from :data:`KNOWN_SOURCE_KEYS`; anything else raises
    ``ValueError`` so a typo ("dolmaa") can never silently no-op."""
    if not source_filter:
        return None
    if not isinstance(source_filter, dict) or \
            not set(source_filter) <= {"include", "exclude"}:
        raise ValueError(
            "source_filter must be a dict with 'include' and/or 'exclude' "
            f"lists of source keys, got {source_filter!r}")
    include = frozenset(source_filter.get("include") or ())
    exclude = frozenset(source_filter.get("exclude") or ())
    bad = (include | exclude) - set(KNOWN_SOURCE_KEYS)
    if bad:
        raise ValueError(
            f"unknown source key(s) {sorted(bad)}; valid keys: "
            f"{list(KNOWN_SOURCE_KEYS)}")
    if not include and not exclude:
        return None
    return (include or None, exclude)


def _normalize_source_weights(source_weights) -> Optional[dict]:
    """Validate/normalise ``source_weights`` into ``{key: float}`` with the
    1.0 (no-op) entries dropped, or ``None`` when nothing remains. Weights must
    be finite and >= 0; a weight of 0 removes the source entirely (it behaves
    like an exclude). Unknown keys raise ``ValueError`` (typo safety)."""
    if not source_weights:
        return None
    if not isinstance(source_weights, dict):
        raise ValueError(
            f"source_weights must be a dict of source-key -> weight, got "
            f"{source_weights!r}")
    bad = set(source_weights) - set(KNOWN_SOURCE_KEYS)
    if bad:
        raise ValueError(
            f"unknown source key(s) {sorted(bad)}; valid keys: "
            f"{list(KNOWN_SOURCE_KEYS)}")
    out: dict = {}
    for key, w in source_weights.items():
        w = float(w)
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(
                f"source_weights[{key!r}] must be a finite weight >= 0, "
                f"got {w!r}")
        if w != 1.0:
            out[key] = w
    return out or None


class HybridRetriever(Retriever):
    """Dense + BM25 + RRF over a ``Document`` corpus we built ourselves.

    ``citation_lambda`` (default 0.0) optionally adds ``lambda*log(max(cites,1))``
    to each fused score before the final top-k cut -- a small, opt-in prior. With
    the default 0.0 the ranking is pure dense+BM25 RRF (no regression risk).

    SOURCE-AWARE RETRIEVAL (opt-in; both default to off => byte-identical
    ranking, pinned by tests/test_source_aware.py). Motivated by the measured
    index-growth regression (RESULTS.md §3b, 2026-06-10): adding 2.34M
    web-mined Dolma docs to the served 3.68M index crowded the canonical
    targets out of the paper-level top-20 on the TheoremSearch-110
    (corpus-only 13.6% -> 11.8%). Two knobs, settable per-instance here (and
    in ``from_index``) or per-call in :meth:`retrieve`:

    ``source_filter={"include": [...], "exclude": [...]}``
        Hard include/exclude of canonical source keys
        (:data:`KNOWN_SOURCE_KEYS`: arxiv / dolma / stacks / proofwiki /
        other -- derived from each doc's provenance by
        :func:`mathlas.retrieve.corpus.source_key`). Applied IN-CHANNEL,
        before RRF: on the exact dense path the disallowed rows are masked
        out of the similarity argsort (exactly equivalent to retrieving over
        the filtered corpus); BM25 / quantized / faiss channel lists are
        over-fetched (``5*depth + 100``) then filtered, so the channel depth
        is preserved (approximation: a doc beyond the over-fetched head of a
        non-maskable channel stays unreachable; honest and documented).

    ``source_weights={"dolma": 0.5, ...}``
        Per-source multiplicative weight on the FUSED RRF score::

            score(d) = w_src(d) * sum_c 1/(rrf_k + rank_c(d))

        Because the weight multiplies the *sum* over channels, this is
        identical to scaling every channel's reciprocal-rank contribution --
        the formulation that composes cleanly with the rrf_k=10 fusion AND
        with the dual-channel max-sim path (max-sim happens inside the dense
        ranking, upstream of the weight). A weight of 0 drops the source from
        the fused pool entirely. Honesty note: a weight in (0, 1) re-orders
        the fused candidate pool but cannot resurrect a doc that never entered
        any channel's top-``channel_depth``; use ``source_filter`` (or weight
        0) when a source should genuinely not compete for channel slots.
    """

    #: Set when the retriever was built from a prebuilt index (``from_index`` /
    #: ``from_faiss``); ``None`` for a freshly embedded corpus.
    index_path: Optional[str] = None
    index_dim: Optional[int] = None
    index_model: Optional[str] = None

    def __init__(self, documents: Sequence[Document],
                 embedder: Optional[Embedder] = None,
                 rrf_k: int = 10, channel_depth: int = 50,
                 citation_lambda: float = 0.0,
                 _precomputed_emb: Optional[np.ndarray] = None,
                 _faiss_index=None, _quant_index=None, build_bm25: bool = True,
                 _prebuilt_bm25: Optional[BM25Index] = None,
                 statement_matrix: Optional[np.ndarray] = None,
                 reranker: Optional[Reranker] = None,
                 rerank_depth: int = 100,
                 source_filter: Optional[dict] = None,
                 source_weights: Optional[dict] = None) -> None:
        self.docs: List[Document] = list(documents)
        self.embedder = embedder or HashingEmbedder()
        self.rrf_k = int(rrf_k)
        self.channel_depth = int(channel_depth)
        self.citation_lambda = float(citation_lambda)
        self.reranker = reranker
        self.rerank_depth = int(rerank_depth)
        # instance-default source knobs (see class docstring); validated now so
        # a bad config fails at construction, not on the first query.
        self.source_filter = _normalize_source_filter(source_filter)
        self.source_weights = _normalize_source_weights(source_weights)
        self._source_codes_cache: Optional[np.ndarray] = None  # lazy int8/doc

        # OPT-IN second dense channel: the same docs embedded by their CLEANED
        # STATEMENT text (vs the default slogan channel). Attacks the
        # cross-representation gap directly -- a LaTeX-shaped query matches the
        # statement channel; a prose query matches the slogan channel. The two
        # dense channels are combined per-doc by MAX-SIM into ONE dense ranking
        # (then fused with BM25 as usual): on the 200k-prefix mechanism check
        # (docs/RETRIEVAL_UPGRADE_NOTES.md, 2026-06-09) max-sim hit R@1 0.981
        # vs 0.872 for treating the statement channel as a third RRF channel
        # and 0.833 for slogan-only. Row-aligned with ``documents``; default
        # None keeps behaviour exactly as before.
        if statement_matrix is not None:
            sm = np.asarray(statement_matrix)
            if sm.shape[0] != len(self.docs):
                raise ValueError(
                    f"statement matrix rows ({sm.shape[0]}) != docs "
                    f"({len(self.docs)})")
            # a float32 matrix is ADOPTED (normalised in place, not copied):
            # a second full copy is ~60 GB at the served scale. Non-fp32
            # inputs are converted (copied) as before.
            self._stmt_emb: Optional[np.ndarray] = _ensure_unit_rows(sm)
        else:
            self._stmt_emb = None

        # SPARSE channel: BM25 over sparse_text. Optional -- off for very large faiss-served
        # indexes where an in-memory inverted index over every doc is impractical (dense-only).
        # A ``_prebuilt_bm25`` (e.g. loaded from a cache by ``from_index``) is adopted as-is so
        # the corpus is never re-tokenized; otherwise it is built now iff ``build_bm25``.
        if _prebuilt_bm25 is not None:
            self._bm25 = _prebuilt_bm25
        else:
            self._bm25 = BM25Index([d.sparse_text for d in self.docs]) if build_bm25 else None

        # DENSE channel, exactly one mode: a prebuilt FAISS ANN index (the 10M+ scale path, query
        # searched against it), a memmap-served QUANTIZED index (the laptop tier — int8 exact dot
        # or binary-Hamming shortlist + exact rescore, see retrieve/quantized.py), a precomputed
        # flat matrix (from_index, exact dot-product), or embed-the-corpus-now.
        self._faiss = _faiss_index
        self._quant = _quant_index
        if _quant_index is not None:
            if _quant_index.n != len(self.docs):
                raise ValueError(
                    f"quantized index rows ({_quant_index.n}) != docs "
                    f"({len(self.docs)})")
            self._emb = None
        elif _faiss_index is not None:
            self._emb = None
        elif _precomputed_emb is not None:
            emb = np.asarray(_precomputed_emb)
            if emb.shape[0] != len(self.docs):
                raise ValueError(
                    f"precomputed matrix rows ({emb.shape[0]}) != docs "
                    f"({len(self.docs)})")
            # keep float32 unit-norm rows for an exact cosine dot-product.
            # fp32 input is ADOPTED in place (see _ensure_unit_rows): the
            # served 3.68M matrix is ~60 GB and must not be copied again.
            self._emb = _ensure_unit_rows(emb)
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
                   rrf_k: int = 10, channel_depth: int = 50,
                   citation_lambda: float = 0.0,
                   label_in_embed: bool = False,
                   with_bm25: bool = True,
                   bm25_cache: Optional[str] = "auto",
                   statement_index: Optional[str] = None,
                   reranker: Optional[Reranker] = None,
                   rerank_depth: int = 100,
                   quantized: Optional[str] = None,
                   quantized_shortlist: int = 1000,
                   source_filter: Optional[dict] = None,
                   source_weights: Optional[dict] = None) -> "HybridRetriever":
        """Build a retriever from a prebuilt ``index.npz``.

        The npz (written by ``scripts/build_index_mp.py finalize``) holds
        ``{matrix, dim, model, embedder, meta_file}``: ``matrix`` is the (N, dim)
        dense document matrix (e.g. Qwen3-Embedding-8B over the slogans). The
        per-doc meta lives in a sidecar ``<index>.meta.jsonl`` (one JSON record
        per line, row-aligned to the matrix) referenced by ``meta_file`` — the
        full-corpus meta is far too large to store as an npz string scalar. Small
        dev indices may instead carry an in-npz ``meta`` JSON list; both are
        handled. We attach ``matrix`` directly as the dense index (NO re-embedding
        of documents), obtain BM25 over the meta's ``sparse_text``, and embed
        *queries* with ``embedder`` at query time.

        ``embedder`` MUST be the same model the matrix was built with (so query
        and document vectors live in the same space). On CPU that means the
        Qwen3 query encode runs at query time; pass a matching ``Qwen3Embedder``.
        For a CPU dev-smoke a ``HashingEmbedder`` matrix + the same embedder works.

        ``with_bm25`` (default ``True``) turns the sparse channel on -> the served
        retriever is the full dense+BM25+RRF. ``bm25_cache`` controls where the
        sparse index is persisted so it is built ONCE rather than re-tokenized at
        every load (which is slow on a multi-million-doc corpus):

          * ``"auto"`` (default) -> a ``<index-stem>.bm25`` bundle beside the index;
            loaded if present and fresh, else built once and saved there;
          * an explicit path -> that bundle location;
          * ``None`` -> no caching (build in memory each load; the old behaviour,
            handy for throwaway/temp indexes).

        The cache is invalidated automatically if the corpus changes (its doc
        count or the meta file's size/mtime no longer match), so a stale bundle is
        rebuilt rather than served. ``with_bm25=False`` serves dense-only.

        ``statement_index`` (opt-in) names a SECOND npz whose ``matrix`` is the
        same corpus embedded by its cleaned STATEMENT text (built by
        ``scripts/build_statement_channel.py``), row-aligned with ``path``'s
        matrix; it becomes a third RRF channel (see ``__init__``). ``reranker``
        (opt-in) injects a cross-encoder used when ``retrieve(..., rerank=True)``.

        ``quantized`` (opt-in, the LAPTOP TIER): ``"int8"`` or ``"binary"``
        serves the dense channel from memmapped quantized sidecar artifacts
        beside the npz (built once by ``mathlas.retrieve.quantized.
        build_quantized_artifacts``) instead of loading the fp16 matrix to
        fp32 RAM. ``"int8"`` is an exact dequantized dot over ~15 GB on disk;
        ``"binary"`` is a packed-bit Hamming shortlist (~1.9 GB, size
        ``quantized_shortlist``) rescored exactly — measured recall deltas in
        docs/QUANTIZED_TIER.md. Queries must STILL be embedded by the index's
        own model (see retrieve/quantized.py honesty note). Incompatible with
        ``statement_index`` (the statement channel stays fp32-resident).

        ``source_filter`` / ``source_weights`` (opt-in, default off) become the
        served retriever's instance defaults — see the class docstring for the
        semantics and the score math.
        """
        data = np.load(path, allow_pickle=True)
        quant = None
        if quantized is not None:
            if statement_index is not None:
                raise ValueError(
                    "quantized=... does not support statement_index (the "
                    "statement channel is served from the fp32 matrix)")
            from .quantized import QuantizedDenseIndex
            quant = QuantizedDenseIndex(path, kind=quantized,
                                        shortlist=quantized_shortlist)
            matrix = None                 # never touch the 30 GB member
        else:
            # memory-lean: stream the member into ONE unit-norm fp32 array
            # (see _load_unit_fp32_matrix; np.load's transient copies OOM'd
            # the dual-channel load on a 251 GB box).
            matrix = _load_unit_fp32_matrix(path)
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
        n_rows = quant.n if quant is not None else matrix.shape[0]
        if n_rows != len(docs):
            raise ValueError(
                f"index matrix rows ({n_rows}) != meta docs "
                f"({len(docs)}) in {path}")
        # SPARSE channel: load a prebuilt BM25 from the cache beside the index, or
        # build it once and persist it there. Building BM25 over a multi-million-doc
        # corpus (tokenize every doc + assemble the CSR matrix) costs minutes; the
        # cache turns later loads into a few fast array reads. ``with_bm25=False``
        # serves dense-only (e.g. when the sparse channel is intentionally dropped).
        bm25 = (cls._load_or_build_bm25(docs, path, sidecar, bm25_cache)
                if with_bm25 else None)
        stmt_matrix = None
        if statement_index is not None:
            stmt_matrix = _load_unit_fp32_matrix(statement_index)
            if stmt_matrix.shape != matrix.shape:
                raise ValueError(
                    f"statement index shape {stmt_matrix.shape} != slogan index "
                    f"shape {matrix.shape}; the channels must be row-aligned "
                    f"embeddings of the SAME corpus ({statement_index} vs {path})")
        retr = cls(docs, embedder=embedder, rrf_k=rrf_k,
                   channel_depth=channel_depth, citation_lambda=citation_lambda,
                   _precomputed_emb=matrix, _quant_index=quant, build_bm25=False,
                   _prebuilt_bm25=bm25, statement_matrix=stmt_matrix,
                   reranker=reranker, rerank_depth=rerank_depth,
                   source_filter=source_filter, source_weights=source_weights)
        # surface a couple of index facts for introspection / logging.
        n_dim = quant.dim if quant is not None else matrix.shape[1]
        retr.index_path = os.fspath(path)
        retr.index_dim = int(data["dim"]) if "dim" in data else int(n_dim)
        retr.index_model = str(data["model"]) if "model" in data else None
        if embedder is not None and getattr(embedder, "dim", None) and \
                int(embedder.dim) != int(n_dim):
            raise ValueError(
                f"embedder dim ({embedder.dim}) != index dim "
                f"({n_dim}); query and doc vectors must match. Use the "
                f"same model the index was built with (index model={retr.index_model}).")
        return retr

    # ------------------------------------------------------------------ #
    # BM25 cache plumbing (used by ``from_index``): load a prebuilt sparse index
    # from disk when fresh, else build it once and persist it beside the dense
    # index, keeping the cache correctly invalidated when the corpus changes.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_bm25_cache_path(index_path: str,
                                 bm25_cache: Optional[str]) -> Optional[str]:
        """``None`` -> caching disabled; ``"auto"`` -> ``<index-stem>.bm25`` beside
        the index; any other value -> that explicit bundle path."""
        if bm25_cache is None:
            return None
        if bm25_cache == "auto":
            return os.path.splitext(os.fspath(index_path))[0] + ".bm25"
        return os.fspath(bm25_cache)

    @staticmethod
    def _bm25_cache_fresh(bm: BM25Index, cache: str, docs: Sequence[Document],
                          meta_path: Optional[str]) -> bool:
        """A cache is fresh iff its doc count matches the corpus AND -- when a
        ``source.json`` fingerprint and the meta file are both present -- the meta
        file's size+mtime are unchanged. The doc-count guard is load-bearing: a
        row-misaligned BM25 would return wrong/garbage doc ids; the size/mtime
        guard additionally catches a same-count corpus rebuild."""
        if bm.n != len(docs):
            return False
        src = os.path.join(cache, "source.json")
        if not meta_path or not os.path.exists(src):
            return True  # count matches and nothing finer to check -> trust it
        try:
            with open(src, encoding="utf-8") as f:
                info = json.load(f)
            st = os.stat(meta_path)
            return (int(info.get("n", -1)) == len(docs)
                    and int(info.get("meta_size", -1)) == st.st_size
                    and int(info.get("meta_mtime_ns", -1)) == st.st_mtime_ns)
        except Exception:
            return True  # fingerprint unreadable but count matched -> still serve

    @staticmethod
    def _write_bm25_source(cache: str, meta_path: Optional[str], n: int) -> None:
        """Record a small provenance fingerprint of the corpus the cache was built
        from, so a later ``from_index`` can detect a rebuilt/changed corpus."""
        info: dict = {"n": int(n)}
        if meta_path and os.path.exists(meta_path):
            st = os.stat(meta_path)
            info["meta_path"] = os.path.abspath(meta_path)
            info["meta_size"] = int(st.st_size)
            info["meta_mtime_ns"] = int(st.st_mtime_ns)
        with open(os.path.join(cache, "source.json"), "w", encoding="utf-8") as f:
            json.dump(info, f)

    @classmethod
    def _load_or_build_bm25(cls, docs: Sequence[Document], index_path: str,
                            meta_path: Optional[str],
                            bm25_cache: Optional[str]) -> BM25Index:
        cache = cls._resolve_bm25_cache_path(index_path, bm25_cache)
        if cache and os.path.isdir(cache):
            try:
                bm = BM25Index.load(cache)
            except Exception as e:            # corrupt / version-mismatched bundle
                _log(f"BM25 cache at {cache} unreadable ({e}); rebuilding.")
            else:
                if cls._bm25_cache_fresh(bm, cache, docs, meta_path):
                    return bm                 # FAST PATH: no tokenization
                _log(f"BM25 cache at {cache} is stale (corpus changed); rebuilding.")
        # build once (the slow path) and persist for next time.
        if cache:
            _log(f"building BM25 cache over {len(docs)} docs -> {cache} "
                 f"(one-time; minutes at multi-million scale)...")
        bm = BM25Index([d.sparse_text for d in docs])
        if cache:
            try:
                bm.save(cache)
                cls._write_bm25_source(cache, meta_path, len(docs))
                _log(f"wrote BM25 cache: {cache}")
            except Exception as e:            # caching is best-effort; never fatal
                _log(f"could not write BM25 cache to {cache} ({e}); "
                     f"serving without a persisted cache.")
        return bm

    @classmethod
    def from_faiss(cls, faiss_path: str, meta_path: Optional[str] = None,
                   embedder: Optional[Embedder] = None, rrf_k: int = 10,
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

    # ------------------------------------------------------------------ #
    # Source-aware helpers (lazy; nothing is computed until a filter/weight
    # is actually used, so the default path costs zero extra work).
    # ------------------------------------------------------------------ #
    def _source_codes(self) -> np.ndarray:
        """Per-doc canonical-source code (int8 index into KNOWN_SOURCE_KEYS),
        derived once from doc provenance and cached."""
        if self._source_codes_cache is None:
            code = {k: i for i, k in enumerate(KNOWN_SOURCE_KEYS)}
            self._source_codes_cache = np.fromiter(
                (code[source_key(d.doc_id, d.source)] for d in self.docs),
                dtype=np.int8, count=len(self.docs))
        return self._source_codes_cache

    def _allowed_mask(self, filt: Optional[tuple]) -> Optional[np.ndarray]:
        """Boolean allow-mask over doc rows for a normalised filter, or ``None``."""
        if filt is None:
            return None
        include, exclude = filt
        codes = self._source_codes()
        code = {k: i for i, k in enumerate(KNOWN_SOURCE_KEYS)}
        if include is not None:
            mask = np.isin(codes, [code[k] for k in include])
        else:
            mask = np.ones(len(self.docs), dtype=bool)
        if exclude:
            mask &= ~np.isin(codes, [code[k] for k in exclude])
        return mask

    @staticmethod
    def _overfetch(depth: int) -> int:
        """Channel fetch depth when a filter must be applied post-hoc to a
        channel we cannot mask in place (BM25 / quantized / faiss): fetch
        deeper, filter, truncate back to ``depth``. 5x+100 covers an excluded
        source that dominates ~2/3 of the corpus (dolma = 64% of the served
        3.68M index) with room to spare."""
        return 5 * depth + 100

    def _dense_rank(self, query: str, k: int,
                    allowed_mask: Optional[np.ndarray] = None) -> List[int]:
        if not self.docs:
            return []
        q = self.embedder.encode([query], is_query=True)[0].astype(np.float32)
        if self._quant is not None:               # quantized memmap path (laptop tier)
            n = float(np.linalg.norm(q))
            qd = q / n if n > 0 else q
            if allowed_mask is None:
                return self._quant.search(qd, k)
            got = self._quant.search(qd, min(len(self.docs), self._overfetch(k)))
            return [i for i in got if allowed_mask[i]][:k]
        if self._faiss is not None:               # ANN path (10M+ scale)
            import faiss
            qn = np.ascontiguousarray(q[None, :])
            faiss.normalize_L2(qn)                 # inner product == cosine on unit vectors
            fetch = k if allowed_mask is None else min(len(self.docs),
                                                       self._overfetch(k))
            _, idx = self._faiss.search(qn, fetch)
            got = [int(i) for i in idx[0] if i >= 0]
            if allowed_mask is None:
                return got
            return [i for i in got if allowed_mask[i]][:k]
        sims = self._emb @ q                      # rows are unit-norm -> cosine
        if self._stmt_emb is not None:
            # dual-channel max-sim: each doc scores by its best representation
            # (slogan or cleaned statement); see __init__ for the measurement.
            sims = np.maximum(sims, self._stmt_emb @ q)
        if allowed_mask is not None:
            # exact in-channel filter: disallowed rows can never take a slot,
            # so this equals retrieving over the filtered corpus.
            sims = np.where(allowed_mask, sims, -np.inf)
            order = np.argsort(-sims)[:k]
            return [int(i) for i in order if np.isfinite(sims[i])]
        return [int(i) for i in np.argsort(-sims)[:k]]

    def _sparse_rank(self, query: str, depth: int,
                     allowed_mask: Optional[np.ndarray] = None) -> List[int]:
        if self._bm25 is None:
            return []
        if allowed_mask is None:
            return [i for i, _ in self._bm25.search(query, depth)]
        got = self._bm25.search(query, min(len(self.docs),
                                           self._overfetch(depth)))
        return [i for i, _ in got if allowed_mask[i]][:depth]

    def _maybe_rerank(self, query: str, ranked: List[int]) -> List[int]:
        """Cross-encoder rerank of the top ``rerank_depth`` fused candidates,
        BLENDED with the first-stage order by RRF (not a replacement).

        Why blend: measured on the 3.68M-index self-recall proxy (n=1000,
        docs/RETRIEVAL_UPGRADE_NOTES.md, 2026-06-09), replacing the first-stage
        order with the raw cross-encoder order DROPS R@1 (a reranker orders by
        relevance, not exact identity, and this corpus holds many same-content
        theorem variants), while RRF-fusing the rerank ranking with the
        first-stage ranking lifted every metric in the honest cross-
        representation setting (dense channel: R@1 0.731 -> 0.748, R@10
        0.950 -> 0.989). The first-stage prior stays in the loop.

        Honest fallback: if the reranker cannot load (no torch/transformers, no
        weights, no GPU memory), we SAY so on stderr and return the fused order
        unchanged -- never a silent pretend-rerank, never a crash of retrieval.
        """
        if self.reranker is None or not ranked:
            return ranked
        head, tail = ranked[:self.rerank_depth], ranked[self.rerank_depth:]
        try:
            texts = [doc_rerank_text(self.docs[i].name, self.docs[i].slogan,
                                     self.docs[i].statement) for i in head]
            scores = self.reranker.score(query, texts)
        except Exception as e:  # ImportError / OSError / CUDA OOM ...
            _log(f"reranker unavailable ({type(e).__name__}: {e}); "
                 f"serving the un-reranked dense+BM25 fusion.")
            return ranked
        order = np.argsort(-np.asarray(scores), kind="stable")
        rerank_order = [head[int(i)] for i in order]
        blended = rrf_fuse([head, rerank_order], rrf_k=self.rrf_k)
        head_out = [d for d, _ in
                    sorted(blended.items(), key=lambda kv: kv[1], reverse=True)]
        return head_out + tail

    def retrieve(self, query: str, k: int = 10, mode: str = "hybrid",
                 rerank: Optional[bool] = None,
                 source_filter: Optional[dict] = None,
                 source_weights: Optional[dict] = None) -> List[Candidate]:
        """``mode``: ``hybrid`` (dense+BM25+RRF, the default and production path),
        or ``dense``/``sparse`` for single-channel ablation. Single-channel modes
        rank by that channel's RRF-shaped score so the top-k cut is comparable.
        The opt-in statement channel (if loaded) is folded into the dense
        ranking by per-doc max-sim (see ``_dense_rank``).

        ``rerank``: ``True`` -> cross-encoder rerank of the top ``rerank_depth``
        fused candidates (requires an injected ``reranker``); ``None`` (default)
        -> rerank exactly when a reranker was injected; ``False`` -> never.

        ``source_filter`` / ``source_weights``: per-call source-aware knobs
        (semantics + math in the class docstring). ``None`` (default) falls
        back to the instance defaults set at construction; pass ``{}`` to
        explicitly disable an instance default for this call. With both
        resolved to off, this method is byte-identical to the pre-source-aware
        ranking (pinned by tests/test_source_aware.py)."""
        if mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(
                f"unknown retrieve mode {mode!r}; "
                "expected one of 'hybrid', 'dense', 'sparse'")
        filt = (self.source_filter if source_filter is None
                else _normalize_source_filter(source_filter))
        weights = (self.source_weights if source_weights is None
                   else _normalize_source_weights(source_weights))
        mask = self._allowed_mask(filt)
        depth = max(k, self.channel_depth)
        dense = (self._dense_rank(query, depth, allowed_mask=mask)
                 if mode in ("hybrid", "dense") else [])
        sparse = (self._sparse_rank(query, depth, allowed_mask=mask)
                  if mode in ("hybrid", "sparse") else [])
        if mode == "dense":
            fused = {d: 1.0 / (self.rrf_k + r + 1) for r, d in enumerate(dense)}
        elif mode == "sparse":
            fused = {d: 1.0 / (self.rrf_k + r + 1) for r, d in enumerate(sparse)}
        else:
            fused = rrf_fuse([dense, sparse], rrf_k=self.rrf_k)
        if weights:
            # score(d) = w_src(d) * sum_c 1/(rrf_k + rank_c(d)); w=0 drops the
            # doc. Applied before the (also opt-in) citation prior so a
            # down-weighted source cannot buy its way back via citations.
            codes = self._source_codes()
            wvec = np.ones(len(KNOWN_SOURCE_KEYS), dtype=np.float64)
            for key, w in weights.items():
                wvec[KNOWN_SOURCE_KEYS.index(key)] = w
            for doc_id in list(fused):
                w = wvec[codes[doc_id]]
                if w <= 0.0:
                    del fused[doc_id]
                else:
                    fused[doc_id] *= w
        if self.citation_lambda > 0.0:
            lam = self.citation_lambda
            for doc_id in fused:
                c = self.docs[doc_id].citations
                if c and c > 0:
                    fused[doc_id] += lam * math.log(c)
        ranked = [d for d, _ in
                  sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
        do_rerank = (self.reranker is not None) if rerank is None else bool(rerank)
        if do_rerank:
            if self.reranker is None:
                _log("retrieve(rerank=True) but no reranker injected; "
                     "serving the un-reranked fusion.")
            else:
                ranked = self._maybe_rerank(
                    query, ranked[:max(self.rerank_depth, k)]) + \
                    ranked[max(self.rerank_depth, k):]
        top = [(d, fused[d]) for d in ranked[:k]]
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


def _log(msg: str) -> None:
    """Operational notice to STDERR only — never stdout, which the MCP server
    reserves for JSON-RPC. Emitted just on the slow BM25 cache build/rebuild/
    failure paths; the fast cached-load path stays silent."""
    print(f"[mathlas] {msg}", file=sys.stderr, flush=True)


def _ensure_unit_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot-product is cosine. The build writes unit-norm
    rows already, but a fp16->fp32 round-trip can leave tiny drift; renormalise
    defensively (cheap, and makes a hand-built dev matrix safe too).

    Memory contract: a writable float32 input is normalised IN PLACE, chunked,
    and returned as-is -- at the served 3.68M x 4096 scale the old
    ``(x / n).astype(np.float32)`` allocated a full ~60 GB temp per matrix,
    which (with the optional second statement matrix) OOM-killed a 251 GB box
    at load time (measured 2026-06-10). Non-float32 / read-only inputs are
    copied first, so callers that pass a fresh ``astype`` copy see the old
    behaviour exactly."""
    x = np.asarray(x)
    if x.size == 0:
        return x.astype(np.float32)
    x = x.astype(np.float32, copy=False)
    if not x.flags.writeable:
        x = x.copy()
    chunk = 200_000
    for i in range(0, x.shape[0], chunk):
        n = np.linalg.norm(x[i:i + chunk], axis=1, keepdims=True)
        n[n == 0] = 1.0
        x[i:i + chunk] /= n
    return x


def _load_unit_fp32_matrix(path: str, member: str = "matrix.npy") -> np.ndarray:
    """Stream an npz matrix member into ONE unit-norm fp32 array.

    ``np.load(path)["matrix"]`` materialises the raw member (30 GB fp16 at the
    served 3,683,428-doc scale), then the fp32 conversion and the
    normalisation temp pile on top -- with the opt-in second (statement)
    matrix that transient peak exceeded 250 GB and the loader was OOM-killed
    on the build box. ``np.savez`` members are ZIP_STORED, so instead memmap
    the fp16 bytes in place (zero RAM) and convert+renormalise chunkwise into
    one preallocated fp32 array: peak extra memory is a 200k-row chunk, not a
    matrix copy. Falls back to the plain ``np.load`` path for compressed /
    exotic npz files (small dev indexes)."""
    try:
        from .quantized import npz_member_memmap
        src = npz_member_memmap(path, member)
    except Exception:                      # compressed npz etc. -> old path
        return _ensure_unit_rows(
            np.asarray(np.load(path, allow_pickle=True)[member[:-4]])
            .astype(np.float32))
    out = np.empty(src.shape, dtype=np.float32)
    chunk = 200_000
    for i in range(0, src.shape[0], chunk):
        c = src[i:i + chunk].astype(np.float32)
        n = np.linalg.norm(c, axis=1, keepdims=True)
        n[n == 0] = 1.0
        np.divide(c, n, out=out[i:i + chunk])
    return out


__all__ = ["HybridRetriever", "rrf_fuse"]
