#!/usr/bin/env python3
"""Batch DENSE re-embedding of the live web-added corpus — run LATER, on a GPU box.

``add_finding`` (mathlas/webaug.py) appends web-found results to the live corpus
(``reference/downloads/findings.jsonl``) through the BM25/sparse channel with NO
embedding-model load — that is the whole point (growing the index must not load
the 8B per finding, so it works on any machine). Dense vectors are only attached
opportunistically, when a Qwen3 index already happens to be resident in-process.

This script is the SEPARATE BATCH step that embeds the backlog of findings that
still lack a dense vector, using the SAME model the main index was built with, so
the live corpus can later be folded into a dense index too. Run it when a GPU is
free — it is the only place the embedder is loaded for findings.

Usage (one model load for the whole backlog):
  HF_HUB_CACHE=reference/downloads/hf python scripts/reindex_findings.py \\
      --model Qwen/Qwen3-Embedding-8B --device cuda

  # dry-run (no model load): just report how many findings need dense vectors
  python scripts/reindex_findings.py --dry-run

It rewrites findings.jsonl in place (atomic), adding ``dense_vec`` to each record
that lacked one. Idempotent: already-embedded findings are skipped, so it resumes.
NO LLM. (Folding the dense findings into the served index.npz is a further step;
this script just makes the vectors available.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", default=None,
                    help="path to findings.jsonl (default: webaug.findings_path())")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: let sentence-transformers choose)")
    ap.add_argument("--dim", type=int, default=None,
                    help="optional Matryoshka truncation dim")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the backlog without loading any model")
    args = ap.parse_args()

    sys.path.insert(0, _REPO)
    from mathlas.webaug import findings_path, load_findings

    path = args.findings or findings_path()
    recs = load_findings(path)
    todo = [r for r in recs if not r.get("dense_vec")]
    print(f"[reindex] {len(recs)} findings total; {len(todo)} need a dense vector "
          f"-> {path}", flush=True)
    if not todo:
        print("[reindex] nothing to do (all findings already dense).", flush=True)
        return 0
    if args.dry_run:
        print("[reindex] dry-run: no model loaded. Re-run without --dry-run on a "
              "GPU box to embed the backlog.", flush=True)
        return 0

    # The ONLY model load for findings — done once for the whole backlog.
    from mathlas.embed import Qwen3Embedder
    emb = Qwen3Embedder(model=args.model, dim=args.dim, device=args.device)
    print(f"[reindex] model loaded (dim={emb.dim}); embedding {len(todo)} "
          f"findings...", flush=True)
    texts = [" -- ".join(p for p in (r.get("name"), r.get("slogan")) if p)
             or r.get("statement", "") for r in todo]
    vecs = emb.encode(texts, is_query=False)
    by_id = {r["doc_id"]: r for r in recs}
    for r, v in zip(todo, vecs):
        by_id[r["doc_id"]]["dense"] = True
        by_id[r["doc_id"]]["dense_vec"] = [float(x) for x in v]
        by_id[r["doc_id"]]["dense_model"] = args.model
        by_id[r["doc_id"]]["dense_dim"] = int(emb.dim)

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    print(f"[reindex] wrote {len(recs)} findings ({len(todo)} newly embedded) "
          f"with model {args.model}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
