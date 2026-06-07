#!/usr/bin/env python3
"""Large-n DENSE retrieval self-recall over the full index.

Complements the n=15 human-query head-to-head (docs/02_eval_vs_theoremsearch.md):
query each sampled theorem by its FORMAL STATEMENT (LaTeX body) and check whether
its OWN entry — embedded as its NL slogan — is returned in the top-k by the dense
Qwen3 channel, over all 1.34M documents. recall@k for thousands of queries.

This is a self-retrieval PROXY (query = body, target = the same theorem's
slogan-embedded doc), NOT human queries — but it is large-n and a genuine
cross-representation test: the body (LaTeX-heavy) and the slogan (symbol-stripped
NL) are different surface forms of the same theorem, so retrieving one from the
other measures the dense embedder's semantic robustness at scale. Only the DENSE
channel is evaluated (BM25 would trivially match the body against the stored
statement; that would not test semantics).

Run:
  CUDA_VISIBLE_DEVICES=1 HF_HUB_CACHE=reference/downloads/hf PYTHONPATH=. \
    python3 scripts/eval_dense_recall.py --index reference/downloads/index.npz \
      --n 3000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _meta_path(index_path: str, data) -> str:
    if "meta_file" in data:
        cand = os.path.join(os.path.dirname(os.path.abspath(index_path)), str(data["meta_file"]))
        if os.path.exists(cand):
            return cand
    return os.path.splitext(os.path.abspath(index_path))[0] + ".meta.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, default=3000, help="number of query theorems")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", default="1,5,10,20")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    maxk = max(ks)

    print(f"# loading index matrix {args.index} ...", flush=True)
    data = np.load(args.index, allow_pickle=True)
    matrix = np.asarray(data["matrix"])                       # (N, dim) fp16, unit rows
    model = str(data["model"]) if "model" in data else "Qwen/Qwen3-Embedding-8B"
    dim = int(data["dim"]) if "dim" in data else matrix.shape[1]

    sidecar = _meta_path(args.index, data)
    print(f"# loading statements from sidecar {sidecar} ...", flush=True)
    statements = []
    with open(sidecar) as f:
        for line in f:
            r = json.loads(line)
            statements.append(r.get("statement") or r.get("slogan") or "")
    N = len(statements)
    if N != matrix.shape[0]:
        raise SystemExit(f"meta rows ({N}) != matrix rows ({matrix.shape[0]})")
    print(f"# {N} docs, dim={dim}, model={model}", flush=True)

    # Sample query theorems with a non-trivial, bounded-length statement.
    rng = np.random.default_rng(args.seed)
    cand = np.array([i for i in range(N) if 20 <= len(statements[i]) <= 2000], dtype=np.int64)
    idx = np.sort(rng.choice(cand, size=min(args.n, len(cand)), replace=False))
    print(f"# sampled {len(idx)} query theorems (of {len(cand)} eligible)", flush=True)

    from mathlas.embed import Qwen3Embedder
    print(f"# loading {model} on {args.device} for query embedding ...", flush=True)
    emb = Qwen3Embedder(model=model, dim=dim, device=args.device)
    queries = [statements[i] for i in idx]
    t0 = time.time()
    Q = emb.encode(queries, is_query=True).astype(np.float32)        # (n, dim)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-12)
    print(f"# embedded {len(queries)} body-queries in {time.time()-t0:.0f}s", flush=True)

    # Dense retrieval on CPU (the 8B already used the GPU; matrix is 1.34M x 4096).
    M = matrix.astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True).clip(1e-12)
    hits = {k: 0 for k in ks}
    ranks = []
    t0 = time.time()
    B = 64
    for b in range(0, len(idx), B):
        Qb = Q[b:b + B]
        sims = M @ Qb.T                                              # (N, b)
        part = np.argpartition(-sims, maxk, axis=0)[:maxk]           # (maxk, b) unordered
        for j in range(Qb.shape[0]):
            col = part[:, j]
            order = col[np.argsort(-sims[col, j])]                   # top maxk, best first
            target = idx[b + j]
            where = np.where(order == target)[0]
            rank = int(where[0]) + 1 if where.size else maxk + 1
            ranks.append(rank)
            for k in ks:
                if rank <= k:
                    hits[k] += 1
        if (b // B) % 8 == 0:
            print(f"  retrieved {b + Qb.shape[0]}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)

    n = len(idx)
    print("\n=== DENSE self-recall (query = LaTeX statement, target = own slogan-doc) ===")
    print(f"  index: {N} docs, dim {dim}, model {model}; n={n} queries")
    for k in ks:
        print(f"  recall@{k:<3} = {hits[k]}/{n} = {100*hits[k]/n:5.1f}%")
    ranks_arr = np.array(ranks)
    found = ranks_arr[ranks_arr <= maxk]
    if found.size:
        print(f"  median rank (of those found in top-{maxk}) = {int(np.median(found))}")
    print(f"  (self-retrieval proxy; dense channel only; honest n={n} large-sample number)")


if __name__ == "__main__":
    main()
