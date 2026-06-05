"""Light retrieval validation: Hit@k of OUR hybrid retriever on the open test set.

Honest, CPU-only, small-subset validation (NOT the full 9.2M benchmark, which
needs a GPU index). The released test set (``theorems-test.parquet``) is 110 NL
queries, each naming a target theorem in an arXiv paper. Only some of those
papers' theorems are reachable in a small subset, so we:

  1. find the test queries whose target PAPER is in the dataset,
  2. force-load those papers' theorems + N distractor theorems,
  3. retrieve for each query and measure Hit@k at the PAPER level (did a
     top-k candidate come from the correct paper?) and the THEOREM level (did it
     match the target theorem's number/label?).

Paper-level Hit@k is the robust metric here (theorem-number parsing varies);
both are reported. With the DEFAULT HashingEmbedder this measures the BM25-led
floor; swap ``--embedder qwen3`` (GPU) for the production number.

Usage:
  python scripts/eval_retrieval.py --corpus reference/theorem-search-dataset \\
      --distractors 3000 --k 20
"""
from __future__ import annotations

import argparse
import re


def _norm_num(s: str) -> str:
    """Normalise 'Theorem 3.1' / 'Thm. 3.1' / 'Lemma 46.3.3.' -> the number '3.1'."""
    m = re.search(r"(\d+(?:\.\d+)*)", s or "")
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--distractors", type=int, default=3000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--embedder", choices=["hashing", "qwen3"], default="hashing")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from mathlas.retrieve.corpus import load_documents
    from mathlas.retrieve.hybrid import HybridRetriever

    base = args.corpus
    test = pq.read_table(f"{base}/theorems-test.parquet").to_pylist()

    # map arXiv id -> paper_id
    papers = {}
    for b in pq.ParquetFile(f"{base}/paper.parquet").iter_batches(batch_size=8192):
        for r in b.to_pylist():
            papers[str(r["paper_id"])] = r
    link2pid = {}
    for pid, p in papers.items():
        m = re.search(r"(\d{4}\.\d{4,5})", p.get("link") or "")
        if m:
            link2pid.setdefault(m.group(1), pid)

    queries = []  # (query, target_paper_id, target_number)
    target_pids = set()
    for r in test:
        m = re.search(r"(\d{4}\.\d{4,5})", r["link to paper on arxiv"] or "")
        if m and m.group(1) in link2pid:
            pid = link2pid[m.group(1)]
            target_pids.add(pid)
            queries.append((r["query"], pid, _norm_num(r["theorem number"])))
    print(f"# {len(queries)} test queries with target paper in dataset "
          f"({len(target_pids)} papers); {args.distractors} distractors")

    docs = load_documents(base, limit=args.distractors, include_paper_ids=target_pids)
    # doc_id -> (paper_id, number); we need paper_id per doc -> re-derive from theorem table
    tid2pid = {}
    for b in pq.ParquetFile(f"{base}/theorem.parquet").iter_batches(batch_size=8192):
        for r in b.to_pylist():
            tid = str(r["theorem_id"])
            if tid in {d.doc_id for d in docs}:
                tid2pid[tid] = str(r["paper_id"])
        if len(tid2pid) >= len(docs):
            break

    embedder = None
    if args.embedder == "qwen3":
        from mathlas.embed import Qwen3Embedder
        embedder = Qwen3Embedder()
    print(f"# indexing {len(docs)} docs (embedder={args.embedder}) ...")
    R = HybridRetriever(docs, embedder=embedder)

    paper_hits = thm_hits = 0
    for q, pid, num in queries:
        cands = R.retrieve(q, k=args.k)
        ph = any(tid2pid.get((c.meta or {}).get("doc_id")) == pid for c in cands)
        th = any(_norm_num(c.name or "") == num
                 and tid2pid.get((c.meta or {}).get("doc_id")) == pid
                 for c in cands)
        paper_hits += ph
        thm_hits += th
        print(f"  [{'P' if ph else '.'}{'T' if th else '.'}] {q[:70]}")

    n = len(queries) or 1
    print(f"\nHit@{args.k}  paper-level: {paper_hits}/{n} ({100*paper_hits/n:.0f}%)  "
          f"theorem-level: {thm_hits}/{n} ({100*thm_hits/n:.0f}%)")
    print("(small-subset, CPU, hashing-embedder floor; production = qwen3 + full index)")


if __name__ == "__main__":
    main()
