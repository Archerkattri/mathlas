"""Build the FULL retrieval index over the open theorem dataset (offline, GPU).

DEFERRED / DO-NOT-RUN-NOW: this is the heavy job (up to ~1.3M theorems x a Qwen3
embedding pass). Run it on an idle GPU box, never while another GPU job is live.
The repo's light validation uses a small ``--limit`` subset on CPU instead
(``scripts/eval_retrieval.py``).

What it does
------------
1. Loads ``Document`` records from the open dataset parquets (our own data
   plumbing -- no third-party service).
2. Embeds each document's NL slogan with an open-weights embedder
   (Qwen3-Embedding by default -- the open MTEB SOTA, arXiv:2506.05176), the
   load-bearing "embed the meaning not the notation" lesson.
3. Saves the dense matrix + the document metadata to disk as an .npz, so serving
   loads instantly. (BM25 is rebuilt at load -- it is cheap.)

Scale notes (knowledge from the TheoremSearch recipe, implemented ourselves)
----------------------------------------------------------------------------
At full scale, replace the exact NumPy dot-product with an ANN index
(faiss/hnswlib) and binary-quantise -> Hamming -> cosine-rerank. For <~1M docs an
exact GPU matmul is fine. Shard by hashing theorem_id across GPUs.

Usage
-----
  python scripts/build_index.py --corpus reference/theorem-search-dataset \\
      --out index.npz --embedder qwen3 --limit 0   # 0 = no cap (FULL build)
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the math_engine retrieval index (offline).")
    ap.add_argument("--corpus", required=True, help="dir with dataset parquets")
    ap.add_argument("--out", default="index.npz")
    ap.add_argument("--embedder", choices=["qwen3", "hashing"], default="qwen3")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--embed-dim", type=int, default=0, help="0 = model default")
    ap.add_argument("--limit", type=int, default=0, help="cap docs (0 = no cap)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from mathlas.retrieve.corpus import load_documents

    print(f"[1/3] loading documents (limit={args.limit or 'ALL'}) ...")
    docs = load_documents(args.corpus, limit=args.limit or None)
    print(f"      {len(docs)} documents")

    print(f"[2/3] embedding slogans with {args.embedder} ...")
    if args.embedder == "qwen3":
        from mathlas.embed import Qwen3Embedder
        emb = Qwen3Embedder(model=args.model, dim=args.embed_dim or None)
    else:
        from mathlas.embed import HashingEmbedder
        emb = HashingEmbedder()

    texts = [d.embed_text for d in docs]
    mats = []
    for i in range(0, len(texts), args.batch):
        mats.append(emb.encode(texts[i:i + args.batch], is_query=False))
        if (i // args.batch) % 20 == 0:
            print(f"      {i + len(mats[-1])}/{len(texts)}")
    matrix = np.vstack(mats) if mats else np.zeros((0, emb.dim), np.float32)

    print(f"[3/3] saving -> {args.out}")
    meta = [{"doc_id": d.doc_id, "name": d.name, "slogan": d.slogan,
             "statement": d.statement, "source": d.source, "title": d.title}
            for d in docs]
    np.savez_compressed(args.out, matrix=matrix, meta=json.dumps(meta),
                        dim=emb.dim, embedder=args.embedder, model=args.model)
    print("done.")


if __name__ == "__main__":
    main()
