#!/usr/bin/env python3
"""Re-score the index-growth regression with source-aware retrieval. CPU-ONLY.

Context (RESULTS.md §3b, measured 2026-06-10): growing the served index
1.34M -> 3.68M by adding 2.34M web-mined Dolma docs crowded canonical targets
out of the paper-level top-20 on the TheoremSearch-110 (corpus-only baseline
13.6% -> 11.8%; reachable-15 hybrid 80.0 -> 73.3% theorem). This script
measures what the opt-in ``source_weights`` / ``source_filter`` knobs
(mathlas/retrieve/hybrid.py) buy back — and what they cost on queries whose
true target IS a Dolma doc — using only cached artifacts (no GPU, no model):

  * the 110 query vectors: reference/downloads/retrieval_upgrades/
    queries_webaug110.npz (built once by scripts/embed_webaug110_queries.py,
    Qwen3-Embedding-8B query-side on CPU);
  * the n=3000 self-recall channel caches: dense_top100.npy / bm25_top100.npy
    (scripts/eval_retrieval_upgrades.py ranks stage, same 3.68M index).

Stages (artifacts cached under reference/downloads/retrieval_upgrades/):

  codes       one streaming pass over the 3.5GB meta -> per-row canonical
              source code (int8 into KNOWN_SOURCE_KEYS), cached as
              source_codes_full.npy (3.7MB).
  selfrecall  n=3000 body->slogan self-recall (targets INCLUDE dolma rows:
              ~64% of the sample) re-fused from the cached channel ranks for
              the weight matrix {off, 0.5, 0.25, 0 (pool-drop), exclude
              (in-channel)}. The off row must REPRODUCE results.json
              hybrid_rrf_k10 exactly (the default-unchanged proof); the
              dolma-target split quantifies the honest cost of the knob.
  eval110     the TheoremSearch-110 corpus-only baseline re-scored through the
              REAL retriever (HybridRetriever.from_index on the served index:
              real meta, real BM25 cache, real retrieve()/source plumbing).
              Only the dense channel is precomputed: the exact fp32 cosine
              ranks come from one streamed pass over the npz matrix (the same
              renormalised-fp32 math from_index serves), batched over all 110
              queries — because 110 separate passes over a 30GB matrix at 2
              CPU threads is hours, and from_index(quantized=...) would swap
              the measured-exact dense path for an approximate one. retrieve()
              then runs unmodified (fusion, weights, filter, BM25 over-fetch).

Honest scope notes: the cached self-recall channel lists are 100 deep, so the
in-channel 'exclude' refill there is capped at the cached depth (the eval110
stage has no such cap — its dense mask is exact and BM25 over-fetches live).
Results land in source_weights_results.json + stdout tables, and feed
RESULTS.md §3b / docs/02_eval_vs_theoremsearch.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # NEVER touch the GPUs
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mathlas.retrieve.corpus import KNOWN_SOURCE_KEYS, source_key  # noqa: E402
from mathlas.retrieve.hybrid import rrf_fuse  # noqa: E402

DL = os.path.join(_REPO, "reference", "downloads")
WORK = os.path.join(DL, "retrieval_upgrades")
INDEX = os.path.join(DL, "index_full_dense.npz")
META = os.path.join(DL, "index_full.meta.jsonl")
TEST = os.path.join(_REPO, "reference", "theorem-search-dataset",
                    "theorems-test.parquet")
Q110 = os.path.join(WORK, "queries_webaug110.npz")
CODES = os.path.join(WORK, "source_codes_full.npy")
RESULTS = os.path.join(WORK, "source_weights_results.json")
DENSE110 = os.path.join(WORK, "dense110_top200.npz")

DOLMA = KNOWN_SOURCE_KEYS.index("dolma")


def _save_results(update: dict) -> None:
    cur = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    cur.update(update)
    json.dump(cur, open(RESULTS, "w"), indent=1)
    print(f"# results -> {RESULTS}", flush=True)


# --------------------------------------------------------------------- #
# stage: codes  (one meta pass -> per-row canonical source code)
# --------------------------------------------------------------------- #
def cmd_codes(_args) -> None:
    code = {k: i for i, k in enumerate(KNOWN_SOURCE_KEYS)}
    codes = []
    t0 = time.time()
    with open(META) as f:
        for row, line in enumerate(f):
            m = json.loads(line)
            codes.append(code[source_key(m.get("doc_id"), m.get("source"))])
            if row % 500000 == 0:
                print(f"  meta row {row} ({time.time()-t0:.0f}s)", flush=True)
    arr = np.asarray(codes, dtype=np.int8)
    np.save(CODES, arr)
    dist = {k: int((arr == i).sum()) for i, k in enumerate(KNOWN_SOURCE_KEYS)}
    print(f"# {len(arr)} rows -> {CODES}; source distribution: {dist}",
          flush=True)


# --------------------------------------------------------------------- #
# stage: selfrecall  (n=3000 cached channel ranks, weight matrix)
# --------------------------------------------------------------------- #
KS = (1, 5, 10)


def _metrics(rank_lists, targets) -> dict:
    n = len(targets)
    hits = {k: 0 for k in KS}
    rr = 0.0
    for ranked, t in zip(rank_lists, targets):
        where = np.where(np.asarray(ranked) == t)[0]
        r = int(where[0]) + 1 if where.size else None
        if r:
            rr += 1.0 / r
            for k in KS:
                if r <= k:
                    hits[k] += 1
    out = {f"R@{k}": round(hits[k] / n, 4) for k in KS}
    out["MRR"] = round(rr / n, 4)
    out["n"] = n
    return out


def _fuse_weighted(dense, bm25, codes, rrf_k=10, w_dolma=1.0,
                   exclude=False) -> list:
    """Replicate retrieve()'s hybrid fusion from cached channel lists.
    ``exclude`` = in-channel removal (refill capped at the cached depth);
    ``w_dolma`` = the fused-score multiplier (0 = pool drop)."""
    out = []
    for dr, br in zip(dense, bm25):
        dl = [int(d) for d in dr]
        bl = [int(d) for d in br if d >= 0]
        if exclude:
            dl = [d for d in dl if codes[d] != DOLMA]
            bl = [d for d in bl if codes[d] != DOLMA]
        fused = rrf_fuse([dl, bl], rrf_k=rrf_k)
        if not exclude and w_dolma != 1.0:
            for d in list(fused):
                if codes[d] == DOLMA:
                    if w_dolma <= 0.0:
                        del fused[d]
                    else:
                        fused[d] *= w_dolma
        out.append([d for d, _ in
                    sorted(fused.items(), key=lambda kv: kv[1], reverse=True)])
    return out


def cmd_selfrecall(_args) -> None:
    codes = np.load(CODES)
    q = np.load(os.path.join(WORK, "queries_n3000_seed0.npz"))
    idx = q["idx"].astype(np.int64)
    dense = np.load(os.path.join(WORK, "dense_top100.npy"))
    bm25 = np.load(os.path.join(WORK, "bm25_top100.npy"))
    is_dolma_t = codes[idx] == DOLMA
    print(f"# n={len(idx)} self-recall queries; dolma-target share: "
          f"{int(is_dolma_t.sum())}/{len(idx)}", flush=True)

    results = {}
    configs = [("off", 1.0, False), ("w0.5", 0.5, False),
               ("w0.25", 0.25, False), ("w0", 0.0, False),
               ("exclude", None, True)]
    for label, w, ex in configs:
        fused = _fuse_weighted(dense, bm25, codes, w_dolma=(w or 0.0),
                               exclude=ex)
        rec = {"all": _metrics(fused, idx),
               "dolma_targets": _metrics(
                   [f for f, d in zip(fused, is_dolma_t) if d],
                   idx[is_dolma_t]),
               "nondolma_targets": _metrics(
                   [f for f, d in zip(fused, is_dolma_t) if not d],
                   idx[~is_dolma_t])}
        results[label] = rec
        print(f"  dolma={label:<8} all R@1={rec['all']['R@1']:.3f} "
              f"R@10={rec['all']['R@10']:.3f} MRR={rec['all']['MRR']:.3f} | "
              f"dolma-tgt R@10={rec['dolma_targets']['R@10']:.3f} | "
              f"non-dolma-tgt R@10={rec['nondolma_targets']['R@10']:.3f}",
              flush=True)

    # default-off must REPRODUCE the published hybrid_rrf_k10 row exactly.
    pub = json.load(open(os.path.join(WORK, "results.json")))["hybrid_rrf_k10"]
    got = results["off"]["all"]
    assert all(abs(got[k] - pub[k]) < 5e-5 for k in ("R@1", "R@5", "R@10", "MRR")), \
        f"default-off self-recall {got} != published {pub}"
    print("# default-off REPRODUCES published hybrid_rrf_k10 exactly", flush=True)
    _save_results({"selfrecall_dolma_matrix": results})


# --------------------------------------------------------------------- #
# stage: dense110  (one streamed pass over the matrix -> exact fp32 ranks)
# --------------------------------------------------------------------- #
def _stream_matrix_sims(Q: np.ndarray, chunk_rows: int = 65536) -> np.ndarray:
    """Exact renormalised-fp32 cosine sims (N, nq) in ONE streamed pass over the
    npz's uncompressed matrix member — the same math from_index serves
    (fp16 rows -> fp32 -> L2-renormalise; see hybrid._ensure_unit_rows)."""
    zf = zipfile.ZipFile(INDEX)
    info = zf.getinfo("matrix.npy")
    f = zf.open(info)
    version = np.lib.format.read_magic(f)
    shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    assert not fortran and dtype == np.float16, (shape, fortran, dtype)
    n, dim = shape
    assert dim == Q.shape[1]
    sims = np.empty((n, Q.shape[0]), dtype=np.float32)
    qt = np.ascontiguousarray(Q.astype(np.float32).T)
    t0 = time.time()
    row = 0
    bytes_per_row = dim * 2
    while row < n:
        m = min(chunk_rows, n - row)
        buf = f.read(m * bytes_per_row)
        chunk = np.frombuffer(buf, dtype=np.float16).reshape(m, dim) \
            .astype(np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims[row:row + m] = (chunk / norms) @ qt
        row += m
        if (row // chunk_rows) % 8 == 0:
            print(f"  sims {row}/{n} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    return sims


def cmd_dense110(_args) -> None:
    d = np.load(Q110, allow_pickle=True)
    Q = d["Q"].astype(np.float32)
    queries = [str(s) for s in d["queries"]]
    codes = np.load(CODES)
    sims = _stream_matrix_sims(Q)
    depth = 200
    nondolma = codes != DOLMA
    top = np.zeros((len(queries), depth), dtype=np.int64)
    top_nd = np.zeros((len(queries), depth), dtype=np.int64)
    for j in range(len(queries)):
        col = sims[:, j]
        part = np.argpartition(-col, depth)[:depth]
        top[j] = part[np.argsort(-col[part])]
        masked = np.where(nondolma, col, -np.inf)
        part = np.argpartition(-masked, depth)[:depth]
        part = part[np.argsort(-masked[part])]
        top_nd[j] = part
    np.savez(DENSE110, queries=np.array(queries, dtype=object),
             top=top, top_nodolma=top_nd)
    print(f"# wrote exact dense top-{depth} (full + no-dolma) -> {DENSE110}",
          flush=True)


# --------------------------------------------------------------------- #
# stage: eval110  (real retriever, real retrieve(); dense ranks precomputed)
# --------------------------------------------------------------------- #
def _load_bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "webaug_110_bench",
        os.path.join(_REPO, "benchmarks", "webaug_110_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_eval110(_args) -> None:
    bench = _load_bench()
    test = bench._load_test(TEST)
    dd = np.load(DENSE110, allow_pickle=True)
    dense_top = {str(q): list(map(int, r))
                 for q, r in zip(dd["queries"], dd["top"])}
    dense_top_nd = {str(q): list(map(int, r))
                    for q, r in zip(dd["queries"], dd["top_nodolma"])}

    from mathlas.retrieve.hybrid import HybridRetriever
    print("# loading served index via from_index (real meta + BM25 cache; "
          "binary sidecar only so the 30GB matrix stays on disk) ...",
          flush=True)
    t0 = time.time()
    retr = HybridRetriever.from_index(INDEX, embedder=None, quantized="binary")
    print(f"# index ready in {time.time()-t0:.0f}s: {len(retr.docs)} docs",
          flush=True)

    # cross-check: the cached codes file == the retriever's own classification
    codes = np.load(CODES)
    own = retr._source_codes()
    assert len(own) == len(codes) and (own == codes).all(), \
        "cached source codes diverge from the retriever's classifier"

    # dense channel: exact precomputed fp32 ranks (see module docstring).
    def dense_stub(query: str, k: int, allowed_mask=None):
        ranks = dense_top[query] if allowed_mask is None else dense_top_nd[query]
        assert k <= len(ranks), f"need deeper dense cache than {len(ranks)}"
        return ranks[:k]
    retr._dense_rank = dense_stub

    reach = bench._reachable(retr, test)
    print(f"# reachable (GT paper in corpus): {len(reach)}/{len(test)}",
          flush=True)

    configs = [
        ("off", {}),
        ("w0.5", {"source_weights": {"dolma": 0.5}}),
        ("w0.25", {"source_weights": {"dolma": 0.25}}),
        ("w0", {"source_weights": {"dolma": 0.0}}),
        ("exclude", {"source_filter": {"exclude": ["dolma"]}}),
    ]
    k = 20
    results = {}
    for label, kw in configs:
        t0 = time.time()
        ph = th = ph_r = th_r = 0
        for i, t in enumerate(test):
            cands = retr.retrieve(t["query"], k=max(k, 50), **kw)
            hit_p = hit_t = False
            for c in cands[:k]:
                meta = {"source": c.source, "title": (c.meta or {}).get("title")}
                if bench._paper_matches(meta, t["gt_arxiv"], t["gt_title_key"]):
                    hit_p = True
                    if t["gt_thm"] and bench._norm_num(c.name or "") == t["gt_thm"]:
                        hit_t = True
                        break
            ph += hit_p
            th += hit_t
            if i in reach:
                ph_r += hit_p
                th_r += hit_t
        n, nr = len(test), len(reach)
        rec = {"paper_hit20": ph, "thm_hit20": th, "n": n,
               "paper_pct": round(100 * ph / n, 1),
               "thm_pct": round(100 * th / n, 1),
               "reach_paper_hit20": ph_r, "reach_thm_hit20": th_r,
               "n_reach": nr,
               "reach_paper_pct": round(100 * ph_r / nr, 1),
               "reach_thm_pct": round(100 * th_r / nr, 1)}
        results[label] = rec
        print(f"  dolma={label:<8} full-110 thm {th}/{n}={rec['thm_pct']}% "
              f"paper {ph}/{n}={rec['paper_pct']}% | reachable-{nr} "
              f"thm {th_r}/{nr}={rec['reach_thm_pct']}% "
              f"paper {ph_r}/{nr}={rec['reach_paper_pct']}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    # the off row must reproduce the measured 2026-06-10 baseline exactly.
    off = results["off"]
    assert (off["paper_hit20"], off["thm_hit20"]) == (13, 11), \
        f"default-off baseline {off} != measured 13/11 (RESULTS.md §3b)"
    print("# default-off REPRODUCES the measured 11.8%/10.0% baseline exactly",
          flush=True)
    _save_results({"webaug110_dolma_matrix": results})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["codes", "selfrecall", "dense110",
                                      "eval110"])
    args = ap.parse_args()
    {"codes": cmd_codes, "selfrecall": cmd_selfrecall,
     "dense110": cmd_dense110, "eval110": cmd_eval110}[args.stage](args)


if __name__ == "__main__":
    main()
