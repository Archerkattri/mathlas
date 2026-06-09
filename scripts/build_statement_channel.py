#!/usr/bin/env python3
"""Build the SECOND dense channel: every doc embedded by its CLEANED STATEMENT.

The served index (index_full_dense.npz) embeds each doc's NL *slogan*; the
body->slogan self-recall gap (R@1 0.614) is cross-representation -- LaTeX-shaped
queries vs symbol-stripped slogan docs. This script produces a row-aligned
sibling matrix over the same corpus where the embedded text is the doc's
``clean_statement``'d LaTeX statement, so a statement-shaped query has a channel
in its own surface form. Served opt-in via
``HybridRetriever.from_index(..., statement_index=...)`` as a third RRF channel.

Same resumable-shard pattern as scripts/build_index_mp.py (kill anytime; rerun
resumes), but single-GPU (GPU1 per the lab's GPU discipline) and sourced from
the SERVED index meta (reference/downloads/index_full.meta.jsonl) so row i of
the output matrix aligns EXACTLY with row i of index_full_dense.npz. The served
files are never touched; the output is a NEW artifact.

Steps:
  1. shard    : stream index_full.meta.jsonl -> docs_NNNNN.jsonl shards of
                {row, text} where text = clean_statement(statement) with an
                embed_text (name -- slogan) fallback for empty/filtered rows
                (keeps row alignment; those rows just duplicate the slogan
                channel and are harmless).
  2. embed    : embed shards' texts (is_query=False, bare doc embed) ->
                emb_NNNNN.npy fp16, atomic, resumable. max_seq_length is capped
                (default 1024 tokens) for throughput; statements are 20-2000
                chars so almost nothing truncates.
  3. finalize : concatenate -> <out>.npz {matrix, dim, model, meta_file ->
                ../index_full.meta.jsonl}.  finalize REFUSES while shards are
                missing; ``--partial`` writes <out>.partial.npz over the first
                K *contiguous* embedded shards (for the early mechanism check).

Usage (resumable; safe to relaunch):
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 HF_HUB_CACHE=reference/downloads/hf \\
    PYTHONPATH=. nohup python3 scripts/build_statement_channel.py all \\
      --meta reference/downloads/index_full.meta.jsonl \\
      --workdir reference/downloads/statement_channel \\
      --out reference/downloads/index_full_statement.npz &
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

from scripts.postprocess_corpus import clean_statement  # noqa: E402


def _shards(wd):
    return sorted(glob.glob(os.path.join(wd, "docs_*.jsonl")))


def _emb_path(wd, shard):
    return os.path.join(wd, "emb",
                        os.path.basename(shard).replace("docs_", "emb_").replace(".jsonl", ".npy"))


def statement_text(rec: dict) -> str:
    """The text this channel embeds for one meta record: the cleaned statement,
    else (rare: empty/figure-polluted/too-short statements) the same name+slogan
    text the slogan channel used -- keeping the row non-empty and aligned."""
    cleaned = clean_statement(rec.get("statement") or "")
    if cleaned:
        return cleaned
    parts = [p for p in (rec.get("name"), rec.get("slogan")) if p]
    return " -- ".join(parts) if parts else (rec.get("statement") or "")


def cmd_shard(args):
    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    marker = os.path.join(wd, "SHARD_DONE")
    if os.path.exists(marker):
        print(f"[shard] already done ({len(_shards(wd))} shards) — resuming", flush=True)
        return
    S = args.shard_size
    n = n_fallback = 0
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
            text = statement_text(rec)
            if not clean_statement(rec.get("statement") or ""):
                n_fallback += 1
            buf.append(json.dumps({"row": n, "text": text}) + "\n")
            n += 1
            if len(buf) == S:
                _flush(n - S)
    if buf:
        _flush(n - len(buf))
    with open(marker, "w") as f:
        f.write(str(n))
    print(f"[shard] {n} rows -> {len(_shards(wd))} shards "
          f"({n_fallback} fell back to name+slogan)", flush=True)


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
    emb._st.max_seq_length = args.max_seq_length   # throughput cap; see module doc
    print(f"[embed] model loaded dim={emb.dim} max_seq={args.max_seq_length}", flush=True)
    t0 = time.time()
    n_done = 0
    for k, s in enumerate(todo, 1):
        rows = [json.loads(l) for l in open(s)]
        vecs = emb.encode([r["text"] for r in rows], is_query=False).astype(np.float16)
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
             model=model, channel="statement", meta_file=meta_name,
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
        print(f"[finalize] {len(not_done)} shards NOT embedded yet — not finalizing "
              f"(use --partial for a prefix artifact)", flush=True)
        sys.exit(2)
    _finalize(wd, shards, args.out, args.model, meta_name)


def cmd_all(args):
    cmd_shard(args)
    cmd_embed(args)
    cmd_finalize(args)
    print("[all] STATEMENT CHANNEL COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("shard", "embed", "finalize", "all"):
        p = sub.add_parser(name)
        p.add_argument("--workdir", required=True)
        p.add_argument("--meta",
                       default=os.path.join(_REPO, "reference/downloads/index_full.meta.jsonl"))
        p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
        if name in ("shard", "all"):
            p.add_argument("--shard-size", type=int, default=4000)
        if name in ("embed", "all"):
            p.add_argument("--max-seq-length", type=int, default=1024)
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
