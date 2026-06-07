#!/usr/bin/env python3
"""Exact (PQ-free) retrieval ceiling: brute-force cosine over the full fp16 matrix on GPU.

Tells us the TRUE recall the embeddings support, independent of the IVF+PQ index. If this is
much higher than the faiss-PQ numbers, the index needs rebuilding without PQ. Uses both GPUs:
the 8B embedder on cuda:0, the 1.6M x 4096 matrix + search on cuda:1.
"""
import glob
import json
import random
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

META = "reference/downloads/index_full.meta.jsonl"
BASE = "reference/downloads/index.npz"
DOLMA_EMB = "reference/downloads/dolma_index_build/clean/emb/emb_*.npy"
TEST = "reference/downloads/splits/test.txt"
N, K = 2000, 10

sample = set(random.Random(0).sample([l.strip() for l in open(TEST)], N))
id2row, q_stmt, q_slog = {}, {}, {}
row = 0
for line in open(META):
    r = json.loads(line)
    did = str(r["doc_id"])
    id2row[did] = row
    if did in sample:
        q_stmt[did] = r.get("statement") or ""
        q_slog[did] = r.get("slogan") or ""
    row += 1
print(f"index rows: {row:,} | sample: {len(sample):,}", flush=True)

from mathlas.embed import Qwen3Embedder
emb = Qwen3Embedder(model="Qwen/Qwen3-Embedding-8B", device="cuda:0")


def embed(qmap):
    dids = [d for d in sample if qmap[d].strip()]
    texts = [qmap[d][:2000] for d in dids]
    vs = [np.array(emb.encode(texts[i:i + 64], is_query=True)) for i in range(0, len(texts), 64)]
    return dids, np.vstack(vs).astype("float32")


s_dids, s_qv = embed(q_stmt)
l_dids, l_qv = embed(q_slog)
del emb
torch.cuda.empty_cache()

print("loading full matrix onto cuda:1 ...", flush=True)
M = np.vstack([np.load(BASE)["matrix"]] + [np.load(e) for e in sorted(glob.glob(DOLMA_EMB))])
print(f"matrix {M.shape}", flush=True)
Mt = torch.nn.functional.normalize(torch.from_numpy(M).to("cuda:1").half(), dim=1)
del M


def run(name, dids, qv):
    q = torch.nn.functional.normalize(torch.from_numpy(qv).to("cuda:1").half(), dim=1)
    gold = torch.tensor([id2row[d] for d in dids], device="cuda:1")
    r1 = rk = mrr = 0
    for i in range(0, len(dids), 256):
        topk = (q[i:i + 256] @ Mt.T).topk(K, dim=1).indices
        match = topk == gold[i:i + 256].unsqueeze(1)
        r1 += match[:, 0].sum().item()
        any_ = match.any(1)
        rk += any_.sum().item()
        ranks = match.float().argmax(1) + 1
        mrr += (any_.float() / ranks).sum().item()
    n = len(dids)
    print(f"{name} EXACT:  R@1 {r1/n:.3f}  R@{K} {rk/n:.3f}  MRR {mrr/n:.3f}", flush=True)


run("STATEMENT", s_dids, s_qv)
run("SLOGAN", l_dids, l_qv)
