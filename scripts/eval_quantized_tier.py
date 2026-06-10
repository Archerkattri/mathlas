#!/usr/bin/env python3
"""Measure the quantized laptop tier against the served 3.68M index.

CPU-only by design (the tier exists so no big GPU is needed). All stages use
the SAME cached n=3000 Qwen3-Embedding-8B query embeddings the published
dense-recall baseline used (retrieval_upgrades/queries_n3000_seed0.npz), so
every number is apples-to-apples with the fp16 exact ranking already cached in
retrieval_upgrades/dense_top100.npy (scripts/eval_retrieval_upgrades.py ranks).

Honesty caveat (same as the baseline): queries are document *bodies* against
their slogan-embedded entries — the hard cross-representation self-recall
regime. And the laptop tier still needs 8B-space query vectors: a 0.6B query
encoder would need its own document index (different embedding space).

Stages (each cached under retrieval_upgrades/, results merged into
results.json there):

  quantize : build the int8 (~15 GB) + binary (~1.9 GB) sidecars (streamed,
             never materializes the 30 GB fp16 matrix)
  eval     : recall + fp16-agreement of (a) int8 exact dot, (b) raw binary
             Hamming, (c) binary Hamming top-1000 -> int8 rescore,
             (d) binary -> fp16-row rescore.  Eval ranking trick: ranking by
             Hamming distance == ranking by descending dot of the +/-1 sign
             vectors (dot = dim - 2*hamming), so the binary channel is scored
             with the same chunked GEMM as int8 (exact, hours -> minutes);
             the production packed-bit kernel is separately pinned equal to
             brute-force popcount in tests/test_quantized_tier.py.
  latency  : wall-clock of the PRODUCTION QuantizedDenseIndex.search paths
             (packed-bit Hamming + rescore, and int8 full dot), cold + warm.

Run (background, moderate CPU):
  OMP_NUM_THREADS=4 PYTHONPATH=. python3 scripts/eval_quantized_tier.py quantize
  OMP_NUM_THREADS=4 PYTHONPATH=. python3 scripts/eval_quantized_tier.py eval
  OMP_NUM_THREADS=4 PYTHONPATH=. python3 scripts/eval_quantized_tier.py latency
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

from mathlas.retrieve.quantized import (                     # noqa: E402
    QuantizedDenseIndex, build_quantized_artifacts, quant_paths)

DL = os.path.join(_REPO, "reference", "downloads")
WORK = os.path.join(DL, "retrieval_upgrades")
INDEX = os.path.join(DL, "index_full_dense.npz")
QUERIES = os.path.join(WORK, "queries_n3000_seed0.npz")
FP16_TOP100 = os.path.join(WORK, "dense_top100.npy")   # cached exact baseline
DEPTH = 100
SHORTLIST = 1000
KS = (1, 5, 10)


def _log(msg):
    print(msg, flush=True)


def _load_queries():
    d = np.load(QUERIES)
    Q = d["Q"].astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-12)
    return d["idx"].astype(np.int64), Q


def _recall(rank_lists, targets, label, results, key, extra=None):
    n = len(targets)
    hits = {k: 0 for k in KS}
    rr = 0.0
    for ranked, t in zip(rank_lists, targets):
        where = np.where(np.asarray(ranked) == t)[0]
        if where.size:
            r = int(where[0]) + 1
            rr += 1.0 / r
            for k in KS:
                if r <= k:
                    hits[k] += 1
    rec = {f"R@{k}": round(hits[k] / n, 4) for k in KS}
    rec["MRR"] = round(rr / n, 4)
    rec["n"] = n
    if extra:
        rec.update(extra)
    _log(f"  {label:<34}" + "  ".join(f"R@{k}={rec[f'R@{k}']:.3f}" for k in KS)
         + f"  MRR={rec['MRR']:.3f}  (n={n})")
    results[key] = rec
    return rec


def _agreement(ranks, fp16_ranks):
    """How closely a ranking tracks the exact fp16 ranking (the real contract:
    quantization should change *retrieval results*, not just recall)."""
    top1 = float(np.mean(ranks[:, 0] == fp16_ranks[:, 0]))
    ov10 = float(np.mean([len(set(a[:10]) & set(b[:10])) / 10.0
                          for a, b in zip(ranks, fp16_ranks)]))
    return {"top1_match_fp16": round(top1, 4), "top10_overlap_fp16": round(ov10, 4)}


def _save_results(results):
    path = os.path.join(WORK, "results.json")
    merged = json.load(open(path)) if os.path.exists(path) else {}
    merged.update(results)
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    _log(f"[results merged into {path}]")


# --------------------------------------------------------------------- #
# chunked exact top-DEPTH GEMM over a row-source callback
# --------------------------------------------------------------------- #
def _topk_sweep(get_rows, n_rows, Q, depth=DEPTH, chunk=200_000, log=""):
    """One pass over the corpus; per query keep the global top-`depth` by
    merging each chunk's candidates with the running best (exact)."""
    nq = Q.shape[0]
    best_v = np.full((nq, depth), -np.inf, dtype=np.float32)
    best_i = np.full((nq, depth), -1, dtype=np.int64)
    t0 = time.time()
    for lo in range(0, n_rows, chunk):
        rows = get_rows(lo, min(lo + chunk, n_rows))      # (c, dim) fp32
        sims = rows @ Q.T                                  # (c, nq)
        c = sims.shape[0]
        d = min(depth, c)
        part = np.argpartition(-sims, d - 1, axis=0)[:d]   # (d, nq) unordered
        cand_v = np.take_along_axis(sims, part, axis=0).T  # (nq, d)
        cand_i = (part + lo).T
        all_v = np.concatenate([best_v, cand_v], axis=1)
        all_i = np.concatenate([best_i, cand_i], axis=1)
        sel = np.argpartition(-all_v, depth - 1, axis=1)[:, :depth]
        best_v = np.take_along_axis(all_v, sel, axis=1)
        best_i = np.take_along_axis(all_i, sel, axis=1)
        if log:
            _log(f"  {log} {min(lo + chunk, n_rows)}/{n_rows} "
                 f"({time.time() - t0:.0f}s)")
    order = np.argsort(-best_v, axis=1)
    return np.take_along_axis(best_i, order, axis=1)


# --------------------------------------------------------------------- #
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
    _log(f"=== quantized tier vs fp16 exact, n={len(idx)}, N={n_rows} ===")

    # fp16 exact baseline (recall of the cached exact ranking, for the table)
    _recall(fp16_ranks, idx, "fp16 exact (cached baseline)", results,
            "quant_fp16_exact")

    # (a) int8 exact dequantized dot
    cache = os.path.join(WORK, f"quant_int8_top{DEPTH}_n{len(idx)}.npy")
    if os.path.exists(cache):
        int8_ranks = np.load(cache)
    else:
        Qs = Q * scale                       # fold the per-dim scale into Q
        int8_ranks = _topk_sweep(
            lambda lo, hi: q8[lo:hi].astype(np.float32), n_rows, Qs,
            log="int8")
        np.save(cache, int8_ranks)
    _recall(int8_ranks, idx, "int8 (exact dequant dot)", results,
            "quant_int8", extra=_agreement(int8_ranks, fp16_ranks))

    # (b)+(c)+(d) binary: Hamming ranking == sign-dot ranking (exact identity)
    bits = np.load(p["binary"], mmap_mode="r")
    cache = os.path.join(WORK, f"quant_binary_top{SHORTLIST}_n{len(idx)}.npy")
    if os.path.exists(cache):
        ham_ranks = np.load(cache)
    else:
        Qsign = np.where(Q > 0, 1.0, -1.0).astype(np.float32)
        ham_ranks = _topk_sweep(
            lambda lo, hi: (np.unpackbits(bits[lo:hi], axis=1)
                            .astype(np.float32) * 2.0 - 1.0),
            n_rows, Qsign, depth=SHORTLIST, chunk=100_000, log="binary")
        np.save(cache, ham_ranks)
    _recall(ham_ranks[:, :DEPTH], idx, "binary raw (Hamming only)", results,
            "quant_binary_raw",
            extra=_agreement(ham_ranks[:, :DEPTH], fp16_ranks))

    # rescore the top-SHORTLIST Hamming candidates per query, exactly
    from mathlas.retrieve.quantized import npz_member_memmap, _unit_rows
    fp16 = npz_member_memmap(INDEX)
    for src, key, label in (("int8", "quant_binary_rescore_int8",
                             f"binary top{SHORTLIST} -> int8 rescore"),
                            ("fp16", "quant_binary_rescore_fp16",
                             f"binary top{SHORTLIST} -> fp16 rescore")):
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
    """Production-path single-query latency, measured on the real artifacts."""
    _, Q = _load_queries()
    results = {}
    for kind in ("binary", "int8"):
        qi = QuantizedDenseIndex(INDEX, kind=kind, shortlist=SHORTLIST)
        times = []
        for i in range(args.reps + 1):
            t0 = time.time()
            qi.search(Q[i], k=10)
            times.append(time.time() - t0)
        cold, warm = times[0], times[1:]
        rec = {"cold_s": round(cold, 2),
               "warm_median_s": round(float(np.median(warm)), 2),
               "reps": args.reps, "threads": os.environ.get("OMP_NUM_THREADS")}
        _log(f"  {kind:<8} cold={rec['cold_s']}s "
             f"warm_median={rec['warm_median_s']}s (reps={args.reps})")
        results[f"quant_latency_{kind}"] = rec
        del qi
    _save_results(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--chunk", type=int, default=100_000)
    q.set_defaults(fn=cmd_quantize)
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=0, help="query subsample (0=all 3000)")
    e.set_defaults(fn=cmd_eval)
    l = sub.add_parser("latency")
    l.add_argument("--reps", type=int, default=5)
    l.set_defaults(fn=cmd_latency)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
