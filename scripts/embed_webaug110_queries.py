#!/usr/bin/env python3
"""Embed the TheoremSearch-110 human queries with Qwen3-Embedding-8B on CPU.

One-time cache builder for the source-aware re-score (scripts/
eval_source_weights.py): the 110 query vectors land in
reference/downloads/retrieval_upgrades/queries_webaug110.npz so the eval never
needs a GPU (and never needs the 8B model again). Query-side encoding —
``is_query=True`` — so the vectors live in exactly the space the served index
was built in (same model, same instruction prefix as the MCP server /
benchmarks/webaug_110_bench.py at query time).

CPU-only by design: CUDA is masked OFF in-process so this can run while the
GPUs carry other jobs. ~2 threads; minutes-to-an-hour, run once.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # NEVER touch the GPUs
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

TEST = os.path.join(_REPO, "reference", "theorem-search-dataset",
                    "theorems-test.parquet")
OUT = os.path.join(_REPO, "reference", "downloads", "retrieval_upgrades",
                   "queries_webaug110.npz")
MODEL = "Qwen/Qwen3-Embedding-8B"
DIM = 4096


def main() -> None:
    import torch
    torch.set_num_threads(2)
    import pyarrow.parquet as pq

    queries = [r["query"] for r in pq.read_table(TEST).to_pylist()]
    print(f"# {len(queries)} queries from {TEST}", flush=True)

    from mathlas.embed import Qwen3Embedder
    t0 = time.time()
    print(f"# loading {MODEL} on CPU (this is the slow, one-time part) ...",
          flush=True)
    emb = Qwen3Embedder(model=MODEL, dim=DIM, device="cpu")
    print(f"# model ready in {time.time()-t0:.0f}s", flush=True)

    vecs = np.zeros((len(queries), DIM), dtype=np.float32)
    t0 = time.time()
    bs = 4
    for b in range(0, len(queries), bs):
        vecs[b:b + bs] = emb.encode(queries[b:b + bs], is_query=True)
        print(f"#  {min(b+bs, len(queries))}/{len(queries)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(OUT, queries=np.array(queries, dtype=object),
             Q=vecs.astype(np.float16), model=MODEL, dim=DIM)
    print(f"# wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
