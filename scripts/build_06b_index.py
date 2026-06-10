#!/usr/bin/env python3
"""Build the LAPTOP-TIER dense index: the served corpus re-embedded with
Qwen3-Embedding-0.6B (dim 1024).

Why: the flagship index (index_full_dense.npz) was built with
Qwen3-Embedding-8B, so even the quantized document tier still needs 8B query
vectors at query time (docs/QUANTIZED_TIER.md "honest query-encoder caveat") -
a small encoder lives in a DIFFERENT embedding space. This script closes that
gap: the SAME corpus, the SAME served representation channel (each doc's
``embed_text`` = "name -- slogan", statement fallback - exactly what
``mathlas.retrieve.corpus.doc_from_meta(...).embed_text`` reconstructs from the
served meta), embedded by the 0.6B model. The output is a drop-in sibling
index whose queries can be encoded on a laptop CPU (~1.2 GB encoder).

Same resumable-shard pattern as scripts/build_statement_channel.py (kill
anytime; rerun resumes), single-GPU (GPU1 per the lab's GPU discipline),
sourced from the SERVED meta (reference/downloads/index_full.meta.jsonl) so
row i of the output aligns EXACTLY with row i of index_full_dense.npz and the
same meta sidecar serves both. The served files are never touched.

Steps:
  1. shard    : stream the meta jsonl -> docs_NNNNN.jsonl shards of
                {row, text} where text = doc_from_meta(rec).embed_text.
  2. embed    : embed shards' texts (is_query=False, bare doc embed) ->
                emb_NNNNN.npy fp16, atomic, resumable. max_seq_length capped
                (slogans are short prose; almost nothing truncates).
  3. finalize : concatenate -> <out>.npz {matrix, dim=1024, embedder="qwen3",
                model, meta_file -> index_full.meta.jsonl}. REFUSES while
                shards are missing; ``--partial`` writes <out>.partial.npz
                over the first K *contiguous* shards (early mechanism check).

Usage (resumable; safe to relaunch):
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 HF_HUB_CACHE=reference/downloads/hf \\
    PYTHONPATH=. nohup python3 scripts/build_06b_index.py all \\
      --meta reference/downloads/index_full.meta.jsonl \\
      --workdir reference/downloads/index_06b_build \\
      --out reference/downloads/index_full_dense_06b.npz &
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mathlas.retrieve.corpus import doc_from_meta  # noqa: E402

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _shards(wd):
    return sorted(glob.glob(os.path.join(wd, "docs_*.jsonl")))


def _emb_path(wd, shard):
    return os.path.join(wd, "emb",
                        os.path.basename(shard).replace("docs_", "emb_").replace(".jsonl", ".npy"))


def cmd_shard(args):
    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    marker = os.path.join(wd, "SHARD_DONE")
    if os.path.exists(marker):
        print(f"[shard] already done ({len(_shards(wd))} shards) - resuming", flush=True)
        return
    S = args.shard_size
    n = 0
    buf = []

    def _flush(i0):
        sp = os.path.join(wd, f"docs_{i0 // S:05d}.jsonl")
        tmp = sp + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(buf)
        os.replace(tmp, sp)
        buf.clear()

    print(f"[shard] streaming {args.meta} ...", flush=True)
    with open(args.meta) as f:
        for line in f:
            rec = json.loads(line)
            # THE served representation channel: "name -- slogan" (statement
            # fallback) - identical to what the 8B slogan matrix embeds, so
            # this index is a drop-in for the same corpus + meta.
            text = doc_from_meta(rec).embed_text
            buf.append(json.dumps({"row": n, "text": text}) + "\n")
            n += 1
            if len(buf) == S:
                _flush(n - S)
    if buf:
        _flush(n - len(buf))
    with open(marker, "w") as f:
        f.write(str(n))
    print(f"[shard] {n} rows -> {len(_shards(wd))} shards", flush=True)


def cmd_embed(args):
    wd = args.workdir
    os.makedirs(os.path.join(wd, "emb"), exist_ok=True)
    shards = _shards(wd)
    todo = [s for s in shards if not os.path.exists(_emb_path(wd, s))]
    print(f"[embed] {len(shards)} shards, {len(todo)} remaining (resume)", flush=True)
    if not todo:
        return
    from mathlas.embed import Qwen3Embedder
    emb = Qwen3Embedder(model=args.model, device="cuda")
    emb._st.max_seq_length = args.max_seq_length   # throughput cap
    print(f"[embed] model loaded dim={emb.dim} max_seq={args.max_seq_length} "
          f"batch={args.batch_size}", flush=True)
    t0 = time.time()
    n_done = 0
    for k, s in enumerate(todo, 1):
        rows = [json.loads(l) for l in open(s)]
        vecs = emb._st.encode([r["text"] for r in rows],
                              normalize_embeddings=True, convert_to_numpy=True,
                              batch_size=args.batch_size).astype(np.float16)
        out = _emb_path(wd, s)
        np.save(out + ".tmp.npy", vecs)
        os.replace(out + ".tmp.npy", out)
        n_done += len(rows)
        if k % 5 == 0 or k == len(todo):
            dt = time.time() - t0
            print(f"[embed] {k}/{len(todo)} shards ({n_done} docs, {dt:.0f}s, "
                  f"{n_done / max(dt, 1):.0f} docs/s)", flush=True)
    print(f"[embed] DONE {len(todo)} shards in {time.time()-t0:.0f}s", flush=True)


def _finalize(wd, shards, out, model, meta_name):
    mats = [np.load(_emb_path(wd, s)) for s in shards]
    matrix = np.vstack(mats).astype(np.float16)
    np.savez(out, matrix=matrix, dim=matrix.shape[1], embedder="qwen3",
             model=model, channel="slogan", meta_file=meta_name,
             n_docs=matrix.shape[0])
    print(f"[finalize] {matrix.shape[0]} x {matrix.shape[1]} -> {out}", flush=True)


def cmd_finalize(args):
    wd = args.workdir
    shards = _shards(wd)
    meta_name = os.path.basename(args.meta)
    done = []
    for s in shards:                       # longest contiguous embedded prefix
        if os.path.exists(_emb_path(wd, s)):
            done.append(s)
        else:
            break
    missing = len(shards) - len(done)
    if args.partial:
        if not done:
            sys.exit("[finalize --partial] no contiguous embedded prefix yet")
        out = os.path.splitext(args.out)[0] + ".partial.npz"
        _finalize(wd, done, out, args.model, meta_name)
        print(f"[finalize --partial] covers rows of the first {len(done)} shards "
              f"({missing} shards still pending)", flush=True)
        return
    not_done = [s for s in shards if not os.path.exists(_emb_path(wd, s))]
    if not_done:
        print(f"[finalize] {len(not_done)} shards NOT embedded yet - not finalizing "
              f"(use --partial for a prefix artifact)", flush=True)
        sys.exit(2)
    _finalize(wd, shards, args.out, args.model, meta_name)


def cmd_all(args):
    cmd_shard(args)
    cmd_embed(args)
    cmd_finalize(args)
    print("[all] 0.6B LAPTOP-TIER INDEX COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("shard", "embed", "finalize", "all"):
        p = sub.add_parser(name)
        p.add_argument("--workdir", required=True)
        p.add_argument("--meta",
                       default=os.path.join(_REPO, "reference/downloads/index_full.meta.jsonl"))
        p.add_argument("--model", default=MODEL)
        if name in ("shard", "all"):
            p.add_argument("--shard-size", type=int, default=4000)
        if name in ("embed", "all"):
            p.add_argument("--max-seq-length", type=int, default=512)
            p.add_argument("--batch-size", type=int, default=128)
        if name in ("finalize", "all"):
            p.add_argument("--out", required=True)
            p.add_argument("--partial", action="store_true",
                           help="write <out>.partial.npz over the contiguous "
                                "embedded prefix (early mechanism validation)")
    args = ap.parse_args()
    if args.cmd == "all":
        args.partial = False
    {"shard": cmd_shard, "embed": cmd_embed,
     "finalize": cmd_finalize, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
