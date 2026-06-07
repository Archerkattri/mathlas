#!/usr/bin/env python3
"""Head-to-head eval vs TheoremSearch on THEIR OWN 110-query test set.

Their published numbers (dataset README, Qwen3-Embedding-8B on DeepSeek-V3.1
slogans, dense-only + citation rerank):

    theorem-level Hit@20 = 45.0%      paper-level Hit@20 = 56.8%

This script measures OUR retriever (dense + Okapi-BM25 + RRF, optional citation
rerank) on the SAME 110 queries and prints both numbers side by side.

Metric (matches the dataset README's definition):
  * paper-level Hit@20   -- some top-20 candidate's paper == the ground-truth
                            paper (matched by arXiv id from the link, or, failing
                            that, by normalised paper title).
  * theorem-level Hit@20 -- some top-20 candidate's paper matches AND its theorem
                            name matches the GT ``theorem number`` (normalised,
                            e.g. "Theorem 3.1" ~ "Thm. 3.1" ~ "3.1").

Two ways to run
---------------
1. REAL index (the point of the exercise) -- after you build the 8B index:

     python scripts/eval_vs_theoremsearch.py \
         --index reference/downloads/index.npz \
         --test  reference/theorem-search-dataset/theorems-test.parquet

   The index's meta carries each doc's paper link/title + theorem name, so NO
   parquet corpus is needed at eval time -- only the test parquet. Queries are
   embedded at query time by the SAME model the index was built with; pass
   ``--embedder qwen3`` (default when an index is given) so it loads Qwen3 for
   the query encode (CPU is fine for 110 short queries).

2. CPU dev-smoke (no GPU, no 8B) -- proves the pipeline + from_index end to end
   on the hashing embedder over a tiny in-process index of the *reachable* test
   papers + a few distractors:

     python scripts/eval_vs_theoremsearch.py --dev-smoke \
         --corpus reference/theorem-search-dataset --distractors 300

   The dev-smoke number is a hashing-embedder floor over only the ~9 reachable
   permissive-subset papers -- it validates the harness, NOT the real score. The
   real 45%+ comparison comes from mode 1 on the 8B index over the full corpus.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time

# allow running as a plain script (python scripts/eval_vs_theoremsearch.py)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_ARXIV = re.compile(r"(\d{4}\.\d{4,5})")


def _arxiv_id(s: str) -> str:
    """Bare arXiv id from a link/string ('.../abs/2310.15076v2' -> '2310.15076')."""
    m = _ARXIV.search(s or "")
    return m.group(1) if m else ""


def _norm_num(s: str) -> str:
    """Theorem-number key: 'Theorem 3.1' / 'Thm. 3.1' / 'Lemma 46.3.3.' -> '3.1'.
    The leading dotted-number run is the stable identifier across naming styles."""
    m = re.search(r"(\d+(?:\.\d+)*)", s or "")
    return m.group(1) if m else ""


def _norm_title(s: str) -> str:
    """Lowercased alphanumeric title key for robust title equality."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _paper_matches(cand_meta: dict, gt_arxiv: str, gt_title_key: str) -> bool:
    """Does this candidate come from the ground-truth paper? Match by arXiv id of
    the candidate's source link first; fall back to normalised paper title."""
    src = cand_meta.get("source") or ""
    if gt_arxiv and _arxiv_id(src) == gt_arxiv:
        return True
    ttl = cand_meta.get("title") or ""
    if gt_title_key and _norm_title(ttl) == gt_title_key:
        return True
    return False


def _load_test(test_path: str):
    """Return a list of ``(query, gt_arxiv_id, gt_title_key, gt_thm_key)``."""
    import pyarrow.parquet as pq
    rows = pq.read_table(test_path).to_pylist()
    out = []
    for r in rows:
        out.append((
            r["query"],
            _arxiv_id(r.get("link to paper on arxiv") or ""),
            _norm_title(r.get("paper title") or ""),
            _norm_num(r.get("theorem number") or ""),
        ))
    return out


def _evaluate(retriever, test, k: int, mode: str = "hybrid", verbose: bool = False):
    """Run the retriever over the test queries; return (paper_hits, thm_hits, n).
    ``mode`` selects the retrieval channel (hybrid/dense/sparse) for ablation."""
    paper_hits = thm_hits = 0
    n = len(test)
    for q, gt_arxiv, gt_title_key, gt_thm in test:
        cands = retriever.retrieve(q, k=k, mode=mode)
        ph = th = False
        for c in cands:
            meta = dict(c.meta or {})
            meta.setdefault("source", getattr(c, "source", None))
            if _paper_matches(meta, gt_arxiv, gt_title_key):
                ph = True
                if gt_thm and _norm_num(c.name or "") == gt_thm:
                    th = True
                    break
        paper_hits += ph
        thm_hits += th
        if verbose:
            print(f"  [{'P' if ph else '.'}{'T' if th else '.'}] {q[:72]}")
    return paper_hits, thm_hits, n


def _report(paper_hits: int, thm_hits: int, n: int, k: int, scope: str) -> None:
    nn = n or 1
    print()
    print(f"=== OUR Hit@{k} ({scope}, n={n} queries) ===")
    print(f"  theorem-level: {thm_hits}/{n} = {100*thm_hits/nn:5.1f}%   "
          f"(TheoremSearch: 45.0%)")
    print(f"  paper-level:   {paper_hits}/{n} = {100*paper_hits/nn:5.1f}%   "
          f"(TheoremSearch: 56.8%)")
    print(f"  delta vs TheoremSearch:  theorem {100*thm_hits/nn - 45.0:+.1f} pts   "
          f"paper {100*paper_hits/nn - 56.8:+.1f} pts")


def run_index(args) -> None:
    """Mode 1: serve a prebuilt index.npz and evaluate on the 110 queries.

    One index load + one BM25 build, then THREE honest views:
      * full-110         -- coverage-limited: most test targets live in the
                           non-permissive 85% of arXiv that is absent from the
                           permissive corpus we are legally allowed to index.
      * reachable-subset -- the FAIR retrieval-quality number: only the queries
                           whose ground-truth paper IS in our corpus (the regime
                           TheoremSearch's 45.0/56.8 was measured in -- every
                           target present in their full 9.2M corpus).
      * channel ablation -- hybrid vs dense-only vs sparse-only on that subset,
                           isolating the value of our BM25+RRF fusion.
    """
    import numpy as np
    from mathlas.retrieve.hybrid import HybridRetriever

    head = np.load(args.index, allow_pickle=True)
    model = str(head["model"]) if "model" in head else "Qwen/Qwen3-Embedding-8B"
    dim = int(head["dim"]) if "dim" in head else None
    embedder = None
    if args.embedder == "qwen3":
        from mathlas.embed import Qwen3Embedder
        print(f"# loading query embedder {model} (dim={dim}) on {args.device} ...", flush=True)
        embedder = Qwen3Embedder(model=model, dim=dim, device=args.device)
    elif args.embedder == "hashing":
        from mathlas.embed import HashingEmbedder
        embedder = HashingEmbedder(dim=dim or 256)

    print(f"# loading index {args.index} (matrix + sidecar meta + BM25 build) ...", flush=True)
    t0 = time.time()
    R = HybridRetriever.from_index(args.index, embedder=embedder,
                                   citation_lambda=args.citation_lambda)
    print(f"# index ready in {time.time()-t0:.0f}s: {len(R.docs)} docs, "
          f"dim={getattr(R, 'index_dim', '?')}, model={getattr(R, 'index_model', '?')}, "
          f"citation_lambda={args.citation_lambda}", flush=True)

    test = _load_test(args.test)

    # Coverage: which test targets are actually present in our permissive corpus?
    idx_arxiv, idx_titles = set(), set()
    for d in R.docs:
        aid = _arxiv_id(d.source or "")
        if aid:
            idx_arxiv.add(aid)
        tk = _norm_title(d.title or "")
        if tk:
            idx_titles.add(tk)
    reachable = [t for t in test
                 if (t[1] and t[1] in idx_arxiv) or (t[2] and t[2] in idx_titles)]
    print(f"# corpus coverage: {len(reachable)}/{len(test)} test targets are in the "
          f"permissive subset\n#   (the other {len(test)-len(reachable)} target "
          f"non-permissive arXiv papers we cannot redistribute/index)", flush=True)

    # full-110 (hybrid) -- coverage-limited
    ph, th, n = _evaluate(R, test, k=args.k, mode="hybrid", verbose=args.verbose)
    _report(ph, th, n, args.k, scope="full-110 (coverage-limited)")

    # reachable-subset, three channels (the fair number + ablation)
    if reachable:
        for mode in ("hybrid", "dense", "sparse"):
            ph, th, n = _evaluate(R, reachable, k=args.k, mode=mode)
            _report(ph, th, n, args.k, scope=f"reachable n={len(reachable)} [{mode}]")
    print("\n# READING: TheoremSearch's 45.0/56.8 was on the FULL 9.2M corpus (every "
          "target present).\n# Our comparable number is the reachable-subset HYBRID "
          "row; full-110 is bounded by corpus licensing, not retrieval quality.\n"
          "# The dense/sparse/hybrid rows isolate the value of our BM25+RRF fusion "
          "over a dense-only baseline (their design).", flush=True)


def run_dev_smoke(args) -> None:
    """Mode 2: build a tiny hashing index of the reachable test papers + a few
    distractors, SAVE it to a temp npz, reload via from_index, and evaluate. This
    exercises the real serving path (from_index + BM25 rebuild + query encode) on
    CPU with no model download. The number is a floor, not the real score."""
    import json
    import numpy as np
    import pyarrow.parquet as pq

    from mathlas.embed import HashingEmbedder
    from mathlas.retrieve.corpus import load_documents
    from mathlas.retrieve.hybrid import HybridRetriever

    base = args.corpus
    raw_test = pq.read_table(f"{base}/theorems-test.parquet").to_pylist()

    # map arXiv id -> paper_id (only papers present in the permissive subset).
    link2pid = {}
    for b in pq.ParquetFile(f"{base}/paper.parquet").iter_batches(batch_size=8192):
        for r in b.to_pylist():
            aid = _arxiv_id(r.get("link") or "")
            if aid:
                link2pid.setdefault(aid, str(r["paper_id"]))

    target_pids = set()
    for r in raw_test:
        aid = _arxiv_id(r.get("link to paper on arxiv") or "")
        if aid in link2pid:
            target_pids.add(link2pid[aid])
    print(f"# dev-smoke: {len(target_pids)} of {len(raw_test)} test papers are in "
          f"the permissive subset; {args.distractors} distractors")
    if not target_pids:
        print("# no reachable test papers — dev-smoke cannot score; harness only.")

    docs = load_documents(base, limit=args.distractors,
                          include_paper_ids=target_pids,
                          label_in_embed=args.label_in_embed)
    print(f"# built {len(docs)} docs; embedding with hashing + saving temp index ...")

    # Build the dense matrix exactly as the real build would, then persist an
    # index.npz with the SAME schema finalize() writes, and serve it via from_index.
    emb = HashingEmbedder()
    matrix = emb.encode([d.embed_text for d in docs], is_query=False).astype(np.float16)
    meta = [{"doc_id": d.doc_id, "name": d.name, "slogan": d.slogan,
             "statement": d.statement, "source": d.source, "title": d.title,
             "label": d.label, "citations": d.citations, "category": d.category}
            for d in docs]
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    np.savez(tmp.name, matrix=matrix, meta=json.dumps(meta), dim=matrix.shape[1],
             embedder="hashing", model="hashing-dev")
    try:
        R = HybridRetriever.from_index(tmp.name, embedder=HashingEmbedder(),
                                       label_in_embed=args.label_in_embed,
                                       citation_lambda=args.citation_lambda)
        print(f"# served temp index: {len(R.docs)} docs, dim={R.index_dim}")
        test = _load_test(f"{base}/theorems-test.parquet")
        # only score queries whose paper is reachable (others can't possibly hit).
        reachable = [t for t in test if t[1] in link2pid]
        print(f"# scoring {len(reachable)} queries whose target paper is reachable")
        ph, th, n = _evaluate(R, reachable, k=args.k, verbose=args.verbose)
        _report(ph, th, n, args.k, scope="DEV-SMOKE hashing floor")
        print("\n# dev-smoke validates from_index + BM25 + the metric harness on "
              "CPU.\n# The REAL 45%/56.8% comparison: build the 8B index, then:\n"
              "#   python scripts/eval_vs_theoremsearch.py --index "
              "reference/downloads/index.npz \\\n"
              "#       --test reference/theorem-search-dataset/theorems-test.parquet")
    finally:
        os.unlink(tmp.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", help="path to a prebuilt index.npz (mode 1)")
    ap.add_argument("--test",
                    default="reference/theorem-search-dataset/theorems-test.parquet",
                    help="path to theorems-test.parquet (110 queries)")
    ap.add_argument("--k", type=int, default=20, help="top-k (Hit@k); default 20")
    ap.add_argument("--embedder", choices=["qwen3", "hashing"], default=None,
                    help="query embedder for --index; default qwen3 (match the "
                         "index's model). Use hashing only for a hashing-built index.")
    ap.add_argument("--device", default="cpu", help="device for the query embedder")
    ap.add_argument("--citation-lambda", type=float, default=0.0,
                    help="optional citation rerank weight (score += lambda*log(cites)); "
                         "default 0 = pure dense+BM25 RRF")
    ap.add_argument("--label-in-embed", action="store_true",
                    help="append humanised meaningful labels to the dense text")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-query P/T hit markers")
    # dev-smoke (mode 2)
    ap.add_argument("--dev-smoke", action="store_true",
                    help="CPU hashing-embedder smoke over the reachable test papers")
    ap.add_argument("--corpus", default="reference/theorem-search-dataset",
                    help="dataset dir (for --dev-smoke)")
    ap.add_argument("--distractors", type=int, default=300,
                    help="distractor theorems for --dev-smoke")
    args = ap.parse_args()

    if args.dev_smoke:
        run_dev_smoke(args)
        return
    if not args.index:
        ap.error("provide --index PATH (mode 1) or --dev-smoke (mode 2).")
    if args.embedder is None:
        args.embedder = "qwen3"  # an index defaults to its build-time model
    run_index(args)


if __name__ == "__main__":
    main()
