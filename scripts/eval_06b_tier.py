#!/usr/bin/env python3
"""Measure the TRUE laptop tier: the 0.6B-built index end-to-end.

The quantized tier (docs/QUANTIZED_TIER.md) shrank the document side but still
needed Qwen3-Embedding-8B query vectors. scripts/build_06b_index.py closes
that: the SAME 3.68M corpus, the SAME served slogan channel, re-embedded with
Qwen3-Embedding-0.6B (dim 1024) so the query encoder runs on a laptop CPU.
This script measures that tier HONESTLY against the published 8B numbers.

Protocol = the published baseline's exactly (scripts/eval_dense_recall.py /
eval_quantized_tier.py): the SAME n=3000 sampled doc rows (idx from
retrieval_upgrades/queries_n3000_seed0.npz), query = the doc's LaTeX
statement, target = its own slogan-embedded entry (body->slogan
cross-representation self-recall, the honest hard regime). The cached 8B
query VECTORS are useless here (different embedding space), so the query
TEXTS are re-encoded with the 0.6B model (stage ``queries``).

Stages (artifacts cache under retrieval_upgrades/, results merge into
results.json there; each stage is run-once / resumable):

  queries  : re-encode the n=3000 query texts with the 0.6B model (GPU)
             -> queries_n3000_seed0_06b.npz
  ranks    : exact fp16 top-100 of the 0.6B index for those queries (GPU
             torch GEMM; CPU fallback) -> dense_top100_06b.npy
  quantize : int8 + binary sidecars beside index_full_dense_06b.npz
             (~3.8 GB + ~0.47 GB at dim 1024 - the real laptop footprint)
  eval     : recall + fp16-agreement of int8 / raw binary / binary->rescore,
             reusing eval_quantized_tier's exact chunked-GEMM machinery
  latency  : the laptop headline - PURE-CPU end-to-end seconds/query:
             0.6B query encode on N CPU threads + production
             QuantizedDenseIndex.search (binary Hamming + int8 rescore)

TheoremSearch-110 corpus-only probe: run benchmarks/webaug_110_bench.py
baseline --index reference/downloads/index_full_dense_06b.npz (the bench reads
the encoder from the npz header; symlink the BM25 cache first, see README).

Run (GPU stages on GPU1, per the lab's GPU discipline):
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 HF_HUB_CACHE=reference/downloads/hf \\
    PYTHONPATH=. python3 scripts/eval_06b_tier.py queries
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONPATH=. python3 scripts/eval_06b_tier.py ranks
  OMP_NUM_THREADS=4 PYTHONPATH=. python3 scripts/eval_06b_tier.py quantize
  OMP_NUM_THREADS=4 PYTHONPATH=. python3 scripts/eval_06b_tier.py eval
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 HF_HUB_CACHE=reference/downloads/hf \\
    PYTHONPATH=. python3 scripts/eval_06b_tier.py latency --threads 4
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

from scripts.eval_quantized_tier import (                     # noqa: E402
    DEPTH, SHORTLIST, _agreement, _recall, _save_results, _topk_sweep)
from mathlas.retrieve.quantized import (                      # noqa: E402
    QuantizedDenseIndex, build_quantized_artifacts, npz_member_memmap,
    quant_paths, _unit_rows)

DL = os.path.join(_REPO, "reference", "downloads")
WORK = os.path.join(DL, "retrieval_upgrades")
INDEX = os.path.join(DL, "index_full_dense_06b.npz")
META = os.path.join(DL, "index_full.meta.jsonl")
QUERIES_8B = os.path.join(WORK, "queries_n3000_seed0.npz")    # idx source only
QUERIES = os.path.join(WORK, "queries_n3000_seed0_06b.npz")
FP16_TOP100 = os.path.join(WORK, "dense_top100_06b.npy")
MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _log(msg):
    print(msg, flush=True)


def _query_texts(idx):
    """One streaming pass over the meta; the SAME query text rule as the
    published baseline (statement, slogan fallback)."""
    wanted = set(int(i) for i in idx)
    rows = {}
    with open(META) as f:
        for row, line in enumerate(f):
            if row in wanted:
                rows[row] = json.loads(line)
                if len(rows) == len(wanted):
                    break
    return [(rows[int(i)].get("statement") or rows[int(i)].get("slogan") or "")
            for i in idx]


def _load_queries():
    if not os.path.exists(QUERIES):
        sys.exit(f"{QUERIES} missing - run the 'queries' stage first")
    d = np.load(QUERIES)
    Q = d["Q"].astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-12)
    return d["idx"].astype(np.int64), Q


# --------------------------------------------------------------------- #
def cmd_queries(args):
    if os.path.exists(QUERIES) and not args.force:
        _log(f"{QUERIES} already exists - skipping (use --force to redo)")
        return
    idx = np.load(QUERIES_8B)["idx"].astype(np.int64)
    _log(f"# {len(idx)} query rows (same sample as the 8B baseline)")
    texts = _query_texts(idx)
    from mathlas.embed import Qwen3Embedder
    _log(f"# loading {MODEL} on {args.device} ...")
    emb = Qwen3Embedder(model=MODEL, device=args.device)
    t0 = time.time()
    Q = emb.encode(texts, is_query=True).astype(np.float32)
    _log(f"# embedded {len(texts)} queries in {time.time()-t0:.0f}s")
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-12)
    np.savez(QUERIES, idx=idx, Q=Q.astype(np.float16), n=len(idx), model=MODEL)
    _log(f"# saved -> {QUERIES}")


def cmd_ranks(args):
    if os.path.exists(FP16_TOP100) and not args.force:
        _log(f"{FP16_TOP100} already exists - skipping (use --force to redo)")
        return
    idx, Q = _load_queries()
    M = npz_member_memmap(INDEX)                      # (N, 1024) fp16
    n_rows = M.shape[0]
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
        _log(f"# GPU exact top-{DEPTH} over {n_rows} x {M.shape[1]} fp16 ...")
        dev = torch.device("cuda")
        Qt = torch.from_numpy(Q).to(dev, torch.float16)
        best_v = torch.full((Q.shape[0], DEPTH), -torch.inf,
                            device=dev, dtype=torch.float32)
        best_i = torch.full((Q.shape[0], DEPTH), -1,
                            device=dev, dtype=torch.int64)
        chunk = 500_000
        t0 = time.time()
        for lo in range(0, n_rows, chunk):
            rows = torch.from_numpy(np.ascontiguousarray(M[lo:lo + chunk])).to(dev)
            sims = (rows @ Qt.T).float().T            # (nq, c)
            v, i = torch.topk(sims, min(DEPTH, sims.shape[1]), dim=1)
            all_v = torch.cat([best_v, v], dim=1)
            all_i = torch.cat([best_i, i + lo], dim=1)
            best_v, sel = torch.topk(all_v, DEPTH, dim=1)
            best_i = torch.gather(all_i, 1, sel)
            _log(f"  fp16 {min(lo + chunk, n_rows)}/{n_rows} ({time.time()-t0:.0f}s)")
        ranks = best_i.cpu().numpy()
    except Exception as e:
        _log(f"# GPU path unavailable ({e}); CPU chunked GEMM fallback")
        ranks = _topk_sweep(lambda lo, hi: _unit_rows(M[lo:hi]), n_rows, Q,
                            log="fp16-cpu")
    np.save(FP16_TOP100, ranks)
    results = {}
    _recall(ranks, idx, "0.6B fp16 exact", results, "q06b_fp16_exact")
    _save_results(results)
    _log(f"# saved -> {FP16_TOP100}")


def cmd_quantize(args):
    t0 = time.time()
    paths = build_quantized_artifacts(INDEX, chunk_rows=args.chunk, log=_log)
    _log(f"done in {time.time() - t0:.0f}s")
    for k in ("int8", "int8_scale", "binary"):
        _log(f"  {paths[k]}  {os.path.getsize(paths[k]) / 1e9:.2f} GB")


def cmd_eval(args):
    idx, Q = _load_queries()
    if args.n:
        idx, Q = idx[:args.n], Q[:args.n]
    fp16_ranks = np.load(FP16_TOP100)[:len(idx)]
    p = quant_paths(INDEX)
    q8 = np.load(p["int8"], mmap_mode="r")
    scale = np.load(p["int8_scale"]).astype(np.float32)
    n_rows = q8.shape[0]
    results = {}
    _log(f"=== 0.6B laptop tier, n={len(idx)}, N={n_rows} ===")
    _recall(fp16_ranks, idx, "0.6B fp16 exact (baseline)", results,
            "q06b_fp16_exact")

    cache = os.path.join(WORK, f"q06b_int8_top{DEPTH}_n{len(idx)}.npy")
    if os.path.exists(cache):
        int8_ranks = np.load(cache)
    else:
        Qs = Q * scale
        int8_ranks = _topk_sweep(
            lambda lo, hi: q8[lo:hi].astype(np.float32), n_rows, Qs,
            log="int8")
        np.save(cache, int8_ranks)
    _recall(int8_ranks, idx, "0.6B int8 (exact dequant dot)", results,
            "q06b_int8", extra=_agreement(int8_ranks, fp16_ranks))

    bits = np.load(p["binary"], mmap_mode="r")
    cache = os.path.join(WORK, f"q06b_binary_top{SHORTLIST}_n{len(idx)}.npy")
    if os.path.exists(cache):
        ham_ranks = np.load(cache)
    else:
        Qsign = np.where(Q > 0, 1.0, -1.0).astype(np.float32)
        ham_ranks = _topk_sweep(
            lambda lo, hi: (np.unpackbits(bits[lo:hi], axis=1)
                            .astype(np.float32) * 2.0 - 1.0),
            n_rows, Qsign, depth=SHORTLIST, chunk=100_000, log="binary")
        np.save(cache, ham_ranks)
    _recall(ham_ranks[:, :DEPTH], idx, "0.6B binary raw (Hamming only)",
            results, "q06b_binary_raw",
            extra=_agreement(ham_ranks[:, :DEPTH], fp16_ranks))

    fp16 = npz_member_memmap(INDEX)
    for src, key, label in (("int8", "q06b_binary_rescore_int8",
                             f"0.6B binary top{SHORTLIST} -> int8 rescore"),
                            ("fp16", "q06b_binary_rescore_fp16",
                             f"0.6B binary top{SHORTLIST} -> fp16 rescore")):
        out = np.empty((len(idx), DEPTH), dtype=np.int64)
        t0 = time.time()
        for i in range(len(idx)):
            cand = ham_ranks[i]
            if src == "int8":
                scores = (q8[cand].astype(np.float32)) @ (Q[i] * scale)
            else:
                scores = _unit_rows(fp16[cand]) @ Q[i]
            out[i] = cand[np.argsort(-scores)[:DEPTH]]
            if i and i % 500 == 0:
                _log(f"  rescore[{src}] {i}/{len(idx)} ({time.time()-t0:.0f}s)")
        _recall(out, idx, label, results, key,
                extra=_agreement(out, fp16_ranks))
    _save_results(results)


def cmd_latency(args):
    """THE laptop headline: pure-CPU end-to-end seconds per query =
    0.6B query encode (CPU, --threads) + production QuantizedDenseIndex
    search. CUDA is masked off in-process; run with CUDA_VISIBLE_DEVICES=
    anyway for belt and braces."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(args.threads)
    idx, Q = _load_queries()
    texts = _query_texts(idx[:args.reps + 1])
    from mathlas.embed import Qwen3Embedder
    _log(f"# loading {MODEL} on CPU ({args.threads} threads) ...")
    t0 = time.time()
    emb = Qwen3Embedder(model=MODEL, device="cpu")
    load_s = time.time() - t0
    _log(f"# model ready in {load_s:.0f}s")
    results = {}
    enc_times = []
    for t in texts:
        t0 = time.time()
        qv = emb.encode([t], is_query=True)[0]
        enc_times.append(time.time() - t0)
    enc_cold, enc_warm = enc_times[0], enc_times[1:]
    _log(f"  0.6B CPU query encode: cold={enc_cold:.2f}s "
         f"warm_median={np.median(enc_warm):.2f}s (reps={args.reps})")
    for kind in ("binary", "int8"):
        qi = QuantizedDenseIndex(INDEX, kind=kind, shortlist=SHORTLIST)
        times = []
        for i in range(args.reps + 1):
            t0 = time.time()
            qi.search(Q[i], k=10)
            times.append(time.time() - t0)
        cold, warm = times[0], times[1:]
        rec = {"search_cold_s": round(cold, 2),
               "search_warm_median_s": round(float(np.median(warm)), 2),
               "encode_cold_s": round(enc_cold, 2),
               "encode_warm_median_s": round(float(np.median(enc_warm)), 2),
               "model_load_s": round(load_s, 1),
               "end_to_end_warm_s": round(
                   float(np.median(enc_warm)) + float(np.median(warm)), 2),
               "reps": args.reps, "threads": args.threads}
        _log(f"  {kind:<8} search cold={rec['search_cold_s']}s "
             f"warm={rec['search_warm_median_s']}s  "
             f"END-TO-END warm={rec['end_to_end_warm_s']}s/query")
        results[f"q06b_latency_{kind}_t{args.threads}"] = rec
        del qi
    _save_results(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queries")
    q.add_argument("--device", default="cuda")
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_queries)
    r = sub.add_parser("ranks")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_ranks)
    z = sub.add_parser("quantize")
    z.add_argument("--chunk", type=int, default=100_000)
    z.set_defaults(fn=cmd_quantize)
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=0, help="query subsample (0=all 3000)")
    e.set_defaults(fn=cmd_eval)
    l = sub.add_parser("latency")
    l.add_argument("--reps", type=int, default=5)
    l.add_argument("--threads", type=int, default=4)
    l.set_defaults(fn=cmd_latency)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
