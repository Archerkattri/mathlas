#!/usr/bin/env python3
"""Index the dolma ~10M theorem corpus, SYNCED with the 1.34M production index.

"Synced" = identical embedder (Qwen3-Embedding-8B, 4096-d, fp16, is_query=False) and identical
meta schema, so the dolma vectors live in the SAME space as `index.npz` and combine row-for-row
with it. The dolma records carry no NL slogan, so their embedded text is the CLEANED statement
(the best available unit); doc_ids are namespaced `dolma:<hash>` so they never collide with the
1.34M theorem_ids.

Pipeline (each step resumable; only `embed` needs a GPU):
  1. shard : dolma theorems_*.jsonl -> docs_*.jsonl in the build_index_mp schema, applying
             clean_statement + per-paper dedup + the field mapping.  CPU — run now.
  2. embed : REUSE `build_index_mp.py embed --workdir <wd>` — same Qwen3-8B, multi-GPU, resumable.
  3. merge : base 1.34M vectors + dolma vectors -> a faiss IVF+PQ index over the union (the
             scalable ~11.3M serving structure) + a row-aligned meta sidecar.  CPU.

WHY faiss at merge: 11.3M × 4096 fp16 ≈ 92 GB (≈185 GB as fp32) — exact numpy dot-product no
longer fits in RAM, exactly the ANN/quantisation follow-up `hybrid.py` already flags. faiss
IVF+PQ compresses the union to ~1 GB and serves ANN top-k in milliseconds. The 1.34M flat index
stays as-is for the small/eval path; the faiss index is the big-corpus serving path.

Usage:
  python scripts/index_dolma_corpus.py shard \
      --corpus reference/downloads/datasets/arxiv_fulltext_theorems \
      --workdir reference/downloads/dolma_index_build
  # GPU (when free): one worker per GPU
  HF_HUB_CACHE=reference/downloads/hf python scripts/build_index_mp.py embed \
      --workdir reference/downloads/dolma_index_build --procs 2 --proc-id 0   # +1 on GPU1
  python scripts/index_dolma_corpus.py merge \
      --base reference/downloads/index.npz --workdir reference/downloads/dolma_index_build \
      --out reference/downloads/index_full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ME not in sys.path:
    sys.path.insert(0, ME)
from scripts.postprocess_corpus import clean_statement  # noqa: E402

SHARD_SIZE = 4000   # match build_index_mp so embed batches are identical
META_KEYS = ("doc_id", "name", "slogan", "statement", "source", "title",
             "label", "citations", "category")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower().strip()


# ── 1. shard : dolma JSONL -> build_index_mp docs_*.jsonl (synced schema) ───────
def cmd_shard(args):
    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    marker = os.path.join(wd, "SHARD_DONE")
    if os.path.exists(marker):
        print(f"[shard] already done ({len(glob.glob(os.path.join(wd, 'docs_*.jsonl')))} shards)", flush=True)
        return
    shards = sorted(glob.glob(os.path.join(args.corpus, "theorems_*.jsonl")))
    out_idx = 0
    buf: list = []
    n_in = n_out = dropped = dup = 0
    cur_paper = None
    seen: set = set()

    def flush():
        nonlocal out_idx, buf
        if not buf:
            return
        sp = os.path.join(wd, f"docs_{out_idx:05d}.jsonl")
        with open(sp + ".tmp", "w") as f:
            for rec in buf:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(sp + ".tmp", sp)
        out_idx += 1
        buf = []

    for s in shards:
        for line in open(s):
            n_in += 1
            r = json.loads(line)
            pid = r.get("paper_id")
            if pid != cur_paper:                      # paper-contiguous -> per-paper dedup window
                cur_paper, seen = pid, set()
            stmt = clean_statement(r.get("statement") or "")
            if not stmt:
                dropped += 1
                continue
            k = _norm(stmt)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            buf.append({
                # dolma's hash is per-PAPER (shared by all its theorems); append the global
                # theorem counter so every doc_id is unique (needed to key slogans / reference docs).
                "doc_id": f"dolma:{r.get('doc_id')}#{n_out}",
                "embed_text": stmt,                   # no slogan -> the cleaned statement is the unit
                "name": r.get("name") or None,
                "slogan": None,
                "statement": stmt,
                "source": "arXiv (dolma-v1_7)",
                "title": None,
                "label": r.get("label") or None,
                "citations": None,
                "category": None,
            })
            n_out += 1
            if len(buf) >= SHARD_SIZE:
                flush()
    flush()
    with open(marker, "w") as f:
        f.write(str(n_out))
    print(f"[shard] in={n_in:,}  out={n_out:,}  dropped={dropped:,}  intra-paper-dup={dup:,}  "
          f"-> {out_idx} docs shards in {wd}", flush=True)


# ── 3. merge : base + dolma vectors -> dense EXACT union index (npz) + meta sidecar ──
def _emb_path(wd, docs_shard):
    return os.path.join(wd, "emb",
                        os.path.basename(docs_shard).replace("docs_", "emb_").replace(".jsonl", ".npy"))


def cmd_merge(args):
    base = np.load(args.base, allow_pickle=True)
    base_matrix = base["matrix"]                      # (1.34M, dim) fp16
    dim = int(base_matrix.shape[1])
    base_meta = os.path.splitext(args.base)[0] + ".meta.jsonl"

    docs_shards = sorted(glob.glob(os.path.join(args.workdir, "docs_*.jsonl")))
    emb_shards = [_emb_path(args.workdir, d) for d in docs_shards]
    missing = [e for e in emb_shards if not os.path.exists(e)]
    if missing:
        sys.exit(f"[merge] {len(missing)}/{len(emb_shards)} dolma emb shards missing — run `embed` first")
    if emb_shards and int(np.load(emb_shards[0], mmap_mode="r").shape[1]) != dim:
        sys.exit("[merge] dim mismatch base vs dolma — different embedder; cannot merge")

    out_meta = args.out + ".meta.jsonl"
    n_base = base_matrix.shape[0]
    n_docs = 0
    with open(out_meta + ".tmp", "w") as mf:
        for line in open(base_meta):                  # base meta verbatim, row-aligned
            mf.write(line)
            n_docs += 1
        for d in docs_shards:                          # dolma meta, normalised to the schema
            for line in open(d):
                r = json.loads(line)
                mf.write(json.dumps({k: r.get(k) for k in META_KEYS}) + "\n")
                n_docs += 1
    os.replace(out_meta + ".tmp", out_meta)

    # EXACT dense union -- NO product quantisation. PQ (4096d -> 64 bytes) crushed recall in
    # practice (slogan R@1 0.98 -> 0.61; nprobe-invariant, so it was the PQ codes, not the IVF
    # coarse search). We instead concatenate the base matrix + dolma emb shards in meta-row order
    # and save a dense fp16 npz, served by HybridRetriever.from_index via brute-force cosine --
    # exactly how the base 1.34M index is served. For ~1.6M x 4096 that is ~13 GB.
    n_total = n_base + sum(int(np.load(e, mmap_mode="r").shape[0]) for e in emb_shards)
    mats = [base_matrix.astype(np.float16)]
    for e in emb_shards:                               # dolma, shard by shard (bounded peak RAM)
        mats.append(np.load(e).astype(np.float16))
    matrix = np.vstack(mats)
    if not (matrix.shape[0] == n_total == n_docs):
        sys.exit(f"[merge] count mismatch: matrix {matrix.shape[0]}, expected {n_total}, meta {n_docs}")
    out_npz = args.out + "_dense.npz"
    np.savez(out_npz, matrix=matrix, dim=dim, model=base["model"], embedder="qwen3",
             n_docs=n_docs, n_base=n_base, n_dolma=n_total - n_base,
             meta_file=os.path.basename(out_meta))
    print(f"[merge] {n_docs:,} docs ({n_base:,} base + {n_total-n_base:,} dolma) x {dim}d "
          f"-> {out_npz} ({matrix.nbytes/1e9:.1f} GB) + {out_meta}", flush=True)
    print("[merge] serve via HybridRetriever.from_index (dense, exact); it is the default index "
          "if present, else set MATHLAS_INDEX to it.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("shard")
    ps.add_argument("--corpus", required=True)
    ps.add_argument("--workdir", required=True)
    pm = sub.add_parser("merge")
    pm.add_argument("--base", required=True, help="the 1.34M index.npz")
    pm.add_argument("--workdir", required=True, help="dolma workdir (with embedded emb/ shards)")
    pm.add_argument("--out", required=True, help="output prefix (writes .faiss/.meta.jsonl/.info.npz)")
    args = ap.parse_args()
    {"shard": cmd_shard, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
