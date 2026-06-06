#!/usr/bin/env python3
"""Resumable, sharded, MULTI-GPU build of the mathlas production retrieval index.

PAUSABLE + RESUMABLE: work is split into fixed shards; every completed shard
writes a file, and every step SKIPS shards already done. Kill anytime (Ctrl-C /
TaskStop) -> rerun resumes from where it stopped. MULTI-GPU: one `embed` worker
per GPU, handling disjoint shards (by proc-id), saved atomically.

Steps (the `all` subcommand runs all three; each is independently resumable):
  1. shard    : load all docs (corpus.load_documents) -> docs_NNNNN.jsonl shards
  2. embed    : per GPU, embed assigned shards' slogans -> emb_NNNNN.npy (fp16)
  3. finalize : concatenate in shard order -> <out>.npz (matrix + meta) for serving

Usage:
  HF_HUB_CACHE=reference/downloads/hf python scripts/build_index_mp.py all \\
      --corpus reference/theorem-search-dataset --model Qwen/Qwen3-Embedding-8B \\
      --workdir reference/downloads/index_build --out reference/downloads/index.npz --ngpu 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        print(f"[shard] already done ({len(_shards(wd))} shards) — resuming", flush=True)
        return
    sys.path.insert(0, _REPO)
    from mathlas.retrieve.corpus import load_documents
    print("[shard] loading the full corpus (one pass)...", flush=True)
    docs = load_documents(args.corpus, limit=None)
    S = args.shard_size
    print(f"[shard] {len(docs)} docs -> {(len(docs)+S-1)//S} shards of {S}", flush=True)
    for i in range(0, len(docs), S):
        sp = os.path.join(wd, f"docs_{i//S:05d}.jsonl")
        tmp = sp + ".tmp"
        with open(tmp, "w") as f:
            for d in docs[i:i + S]:
                f.write(json.dumps({"doc_id": d.doc_id, "embed_text": d.embed_text,
                                    "name": d.name, "slogan": d.slogan,
                                    "statement": d.statement, "source": d.source,
                                    "title": d.title, "label": d.label,
                                    "citations": d.citations,
                                    "category": d.category}) + "\n")
        os.replace(tmp, sp)
    with open(marker, "w") as f:
        f.write(str(len(docs)))
    print(f"[shard] wrote {len(_shards(wd))} shards", flush=True)


def cmd_embed(args):
    wd = args.workdir
    os.makedirs(os.path.join(wd, "emb"), exist_ok=True)
    sys.path.insert(0, _REPO)
    shards = _shards(wd)
    mine = [s for k, s in enumerate(shards) if k % args.procs == args.proc_id]
    todo = [s for s in mine if not os.path.exists(_emb_path(wd, s))]
    print(f"[embed p{args.proc_id}] {len(mine)} assigned, {len(todo)} remaining (resume)", flush=True)
    if not todo:
        return
    from mathlas.embed import Qwen3Embedder
    emb = Qwen3Embedder(model=args.model, device="cuda")
    print(f"[embed p{args.proc_id}] model loaded dim={emb.dim}", flush=True)
    t0 = time.time()
    for k, s in enumerate(todo, 1):
        rows = [json.loads(l) for l in open(s)]
        vecs = emb.encode([r["embed_text"] for r in rows], is_query=False).astype(np.float16)
        out = _emb_path(wd, s)
        np.save(out + ".tmp.npy", vecs)
        os.replace(out + ".tmp.npy", out)
        if k % 5 == 0 or k == len(todo):
            print(f"[embed p{args.proc_id}] {k}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[embed p{args.proc_id}] DONE {len(todo)} shards in {time.time()-t0:.0f}s", flush=True)


def cmd_finalize(args):
    wd = args.workdir
    shards = _shards(wd)
    missing = [s for s in shards if not os.path.exists(_emb_path(wd, s))]
    if missing:
        print(f"[finalize] {len(missing)} shards NOT embedded yet — not finalizing", flush=True)
        sys.exit(2)
    mats, meta = [], []
    keys = ("doc_id", "name", "slogan", "statement", "source", "title",
            "label", "citations", "category")
    for s in shards:
        mats.append(np.load(_emb_path(wd, s)))
        for l in open(s):
            r = json.loads(l)
            # .get tolerates older shards written before label/citations/category.
            meta.append({k: r.get(k) for k in keys})
    matrix = np.vstack(mats).astype(np.float16)
    n_cited = sum(1 for m in meta if isinstance(m.get("citations"), int))
    n_cat = sum(1 for m in meta if m.get("category"))
    print(f"[finalize] {matrix.shape[0]} docs x {matrix.shape[1]}d "
          f"({n_cited} with citations, {n_cat} with category) -> {args.out}", flush=True)
    np.savez(args.out, matrix=matrix, meta=json.dumps(meta), dim=matrix.shape[1],
             embedder="qwen3", model=args.model,
             has_citations=int(n_cited), has_category=int(n_cat))
    print("[finalize] done.", flush=True)


def cmd_all(args):
    cmd_shard(args)
    env = dict(os.environ)
    procs = []
    for gid in range(args.ngpu):
        e = dict(env)
        e["CUDA_VISIBLE_DEVICES"] = str(gid)
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "embed", "--workdir", args.workdir,
             "--procs", str(args.ngpu), "--proc-id", str(gid), "--model", args.model], env=e))
    rc = [p.wait() for p in procs]
    if any(rc):
        print(f"[all] worker exit codes {rc} — fix + rerun (resumes)", flush=True)
        sys.exit(1)
    cmd_finalize(args)
    print("[all] INDEX BUILD COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("shard", "all"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", required=True)
        p.add_argument("--workdir", required=True)
        p.add_argument("--shard-size", type=int, default=4000)
        p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
        if name == "all":
            p.add_argument("--out", required=True)
            p.add_argument("--ngpu", type=int, default=2)
    pe = sub.add_parser("embed")
    pe.add_argument("--workdir", required=True)
    pe.add_argument("--procs", type=int, default=2)
    pe.add_argument("--proc-id", type=int, required=True)
    pe.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    pf = sub.add_parser("finalize")
    pf.add_argument("--workdir", required=True)
    pf.add_argument("--out", required=True)
    pf.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    args = ap.parse_args()
    {"shard": cmd_shard, "embed": cmd_embed, "finalize": cmd_finalize, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
