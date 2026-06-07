#!/usr/bin/env python3
"""Retrieval eval on the merged faiss index using the held-out test split.

For each sampled test doc, query with its STATEMENT (cross-representation: raw theorem -> its
slogan-indexed entry) and its SLOGAN (the realistic NL-query form). Sweep nprobe to separate
IVF-coarse-search loss from the embedding/task difficulty. Reports Recall@1, Recall@k, MRR.
"""
import argparse
import json
import random
import sys

import faiss
import numpy as np

sys.path.insert(0, ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", default="reference/downloads/index_full.faiss")
    ap.add_argument("--meta", default="reference/downloads/index_full.meta.jsonl")
    ap.add_argument("--test", default="reference/downloads/splits/test.txt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--nprobe", default="64,256,1024,1276")  # sweep
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    nprobes = [int(x) for x in str(a.nprobe).split(",")]

    test_ids = [l.strip() for l in open(a.test)]
    sample = set(random.Random(0).sample(test_ids, min(a.n, len(test_ids))))

    id2row, q_stmt, q_slog = {}, {}, {}
    row = 0
    for line in open(a.meta):
        r = json.loads(line)
        did = str(r["doc_id"])
        id2row[did] = row
        if did in sample:
            q_stmt[did] = r.get("statement") or ""
            q_slog[did] = r.get("slogan") or ""
        row += 1
    print(f"index rows: {row:,} | sampled test docs: {len(sample):,}", flush=True)

    from mathlas.embed import Qwen3Embedder
    emb = Qwen3Embedder(model="Qwen/Qwen3-Embedding-8B", device="cuda")
    idx = faiss.read_index(a.faiss)

    def run(field, qmap):
        dids = [d for d in sample if qmap[d].strip()]
        texts = [qmap[d][:2000] for d in dids]
        chunks = [np.array(emb.encode(texts[i:i + 64], is_query=True)) for i in range(0, len(texts), 64)]
        qv = np.ascontiguousarray(np.vstack(chunks).astype("float32"))
        faiss.normalize_L2(qv)
        n = len(dids)
        gold = [id2row[d] for d in dids]
        print(f"\nquery = {field}  (n={n}, k={a.k})", flush=True)
        for nprobe in nprobes:
            idx.nprobe = nprobe
            _, I = idx.search(qv, a.k)
            r1 = r_k = mrr = 0
            for j in range(n):
                hits = list(I[j])
                if gold[j] in hits:
                    rank = hits.index(gold[j]) + 1
                    r1 += rank == 1
                    r_k += 1
                    mrr += 1.0 / rank
            print(f"  nprobe={nprobe:>5}: R@1 {r1/n:.3f}  R@{a.k} {r_k/n:.3f}  MRR {mrr/n:.3f}", flush=True)

    run("STATEMENT", q_stmt)
    run("SLOGAN", q_slog)


if __name__ == "__main__":
    main()
