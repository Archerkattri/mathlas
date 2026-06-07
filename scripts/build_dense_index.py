#!/usr/bin/env python3
"""Build the merged EXACT dense index (base 1.34M + dolma 294K), no PQ.

The IVF+PQ merge crushed recall (slogan R@1 0.98 -> 0.61). The base index was always served
as a dense fp16 matrix with brute-force cosine; we keep that for the union too. Row order
matches index_full.meta.jsonl (base first, then dolma emb shards in sorted order).
"""
import glob

import numpy as np

BASE = "reference/downloads/index.npz"
DOLMA_EMB = "reference/downloads/dolma_index_build/clean/emb/emb_*.npy"
OUT = "reference/downloads/index_full_dense.npz"

base = np.load(BASE, allow_pickle=True)
mats = [base["matrix"]] + [np.load(e) for e in sorted(glob.glob(DOLMA_EMB))]
M = np.vstack(mats).astype(np.float16)
print(f"merged dense matrix: {M.shape} ({M.nbytes/1e9:.1f} GB fp16)", flush=True)

np.savez(
    OUT,
    matrix=M,
    dim=M.shape[1],
    model=base["model"],
    embedder="qwen3",
    n_docs=M.shape[0],
    meta_file="index_full.meta.jsonl",
)
print(f"wrote {OUT}", flush=True)
