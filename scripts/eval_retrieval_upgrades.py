#!/usr/bin/env python3
"""Measure the 2026-06 retrieval upgrades against the served 3.68M index.

Baseline being attacked: body->slogan dense self-recall R@1 0.614 / R@10 0.832
(scripts/eval_dense_recall.py, n=3000, seed 0). Three upgrades, each measured
on the SAME query sample (the saved queries npz, so every number is
apples-to-apples):

  rrf     : dense + BM25 fused by RRF, rrf_k in {10, 20, 60}  (k tune)
  rerank  : Qwen3-Reranker-0.6B cross-encoder over the top-100 RRF candidates
  dual    : second dense channel (docs embedded by cleaned STATEMENT, see
            scripts/build_statement_channel.py) -- validated on the embedded
            row-prefix slice while the full 3.68M embed runs

Caveat stated up front: this is the same self-retrieval proxy as the baseline
(query = a doc's LaTeX statement; target = that doc). For BM25 (and partly the
rerank + statement channel) the query text literally appears in the target's
indexed text, so absolute hybrid numbers are OPTIMISTIC vs human queries; the
dense baseline is the honest cross-representation number. Deltas between
configurations on the same sample are still informative, which is what the k
tune and ablations use.

Stage artifacts are cached under reference/downloads/retrieval_upgrades/ so
each stage runs once; results accumulate in results.json there.

Typical sequence (GPU stages on GPU1):
  PYTHONPATH=. OMP_NUM_THREADS=2 python3 scripts/eval_retrieval_upgrades.py ranks
  PYTHONPATH=. OMP_NUM_THREADS=2 python3 scripts/eval_retrieval_upgrades.py rrf
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 HF_HUB_CACHE=reference/downloads/hf \\
    PYTHONPATH=. python3 scripts/eval_retrieval_upgrades.py rerank --n 1000
  CUDA_VISIBLE_DEVICES=1 ... python3 scripts/eval_retrieval_upgrades.py slice-queries
  PYTHONPATH=. OMP_NUM_THREADS=2 python3 scripts/eval_retrieval_upgrades.py dual
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

DL = os.path.join(_REPO, "reference", "downloads")
WORK = os.path.join(DL, "retrieval_upgrades")
INDEX = os.path.join(DL, "index_full_dense.npz")
META = os.path.join(DL, "index_full.meta.jsonl")
BM25_CACHE = os.path.join(DL, "index_full_dense.bm25")
QUERIES = os.path.join(WORK, "queries_n3000_seed0.npz")
STMT_INDEX = os.path.join(DL, "index_full_statement.npz")
STMT_PARTIAL = os.path.join(DL, "index_full_statement.partial.npz")
DEPTH = 100   # per-channel depth; also the rerank pool size

KS = (1, 5, 10)


# --------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------- #
def _load_queries():
    d = np.load(QUERIES)
    return d["idx"].astype(np.int64), d["Q"].astype(np.float32)


def _meta_rows(wanted: set):
    """One streaming pass over the 3.5GB meta; returns {row: record}."""
    out = {}
    with open(META) as f:
        for row, line in enumerate(f):
            if row in wanted:
                out[row] = json.loads(line)
                if len(out) == len(wanted):
                    break
    return out


def _query_texts(idx):
    rows = _meta_rows(set(int(i) for i in idx))
    return [(rows[int(i)].get("statement") or rows[int(i)].get("slogan") or "")
            for i in idx]


def _dense_topk(M, Q, depth, batch=64, log=""):
    """Exact top-`depth` doc rows per query (rows of Q), best-first."""
    n = Q.shape[0]
    out = np.zeros((n, depth), dtype=np.int64)
    t0 = time.time()
    for b in range(0, n, batch):
        sims = M @ Q[b:b + batch].T                          # (N, bs)
        part = np.argpartition(-sims, depth, axis=0)[:depth]  # unordered
        for j in range(sims.shape[1]):
            col = part[:, j]
            out[b + j] = col[np.argsort(-sims[col, j])]
        if log and (b // batch) % 8 == 0:
            print(f"  {log} {b + sims.shape[1]}/{n} ({time.time()-t0:.0f}s)",
                  flush=True)
    return out


def _recall(rank_lists, targets, label, results, key):
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
    rec = {f"R@{k}": round(hits[k] / n, 4) for k in KS}
    rec["MRR"] = round(rr / n, 4)
    rec["n"] = n
    print(f"  {label:<42}" + "  ".join(f"R@{k}={rec[f'R@{k}']:.3f}" for k in KS)
          + f"  MRR={rec['MRR']:.3f}  (n={n})", flush=True)
    results[key] = rec
    return rec


def _rrf(rankings, rrf_k):
    fused = {}
    for ranking in rankings:
        for r, d in enumerate(ranking):
            fused[d] = fused.get(d, 0.0) + 1.0 / (rrf_k + r + 1)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


def _save_results(results):
    path = os.path.join(WORK, "results.json")
    merged = {}
    if os.path.exists(path):
        merged = json.load(open(path))
    merged.update(results)
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[results merged into {path}]", flush=True)


def _load_index_matrix(path=INDEX, rows=None):
    data = np.load(path, allow_pickle=True)
    M = np.asarray(data["matrix"])
    if rows is not None:
        M = M[:rows]
    M = M.astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True).clip(1e-12)
    return M


# --------------------------------------------------------------------- #
# stage: ranks  (dense + bm25 top-DEPTH for the n=3000 sample; cached)
# --------------------------------------------------------------------- #
def cmd_ranks(args):
    idx, Q = _load_queries()
    dense_p = os.path.join(WORK, f"dense_top{DEPTH}.npy")
    bm25_p = os.path.join(WORK, f"bm25_top{DEPTH}.npy")
    if not os.path.exists(dense_p):
        print(f"# dense channel: loading {INDEX} ...", flush=True)
        M = _load_index_matrix()
        d = _dense_topk(M, Q, DEPTH, log="dense")
        np.save(dense_p, d)
        del M
        print(f"# wrote {dense_p}", flush=True)
    else:
        print(f"# {dense_p} cached", flush=True)
    if not os.path.exists(bm25_p):
        from mathlas.retrieve.bm25 import BM25Index
        print("# bm25 channel: loading cache ...", flush=True)
        bm = BM25Index.load(BM25_CACHE)
        texts = _query_texts(idx)
        out = np.full((len(idx), DEPTH), -1, dtype=np.int64)
        t0 = time.time()
        for i, q in enumerate(texts):
            hits = bm.search(q, DEPTH)
            for j, (doc, _) in enumerate(hits):
                out[i, j] = doc
            if i % 200 == 0:
                print(f"  bm25 {i}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
        np.save(bm25_p, out)
        print(f"# wrote {bm25_p}", flush=True)
    else:
        print(f"# {bm25_p} cached", flush=True)


# --------------------------------------------------------------------- #
# stage: rrf  (k tune over the cached channel ranks)
# --------------------------------------------------------------------- #
def cmd_rrf(args):
    idx, _ = _load_queries()
    dense = np.load(os.path.join(WORK, f"dense_top{DEPTH}.npy"))
    bm25 = np.load(os.path.join(WORK, f"bm25_top{DEPTH}.npy"))
    results = {}
    print(f"\n=== hybrid (dense+BM25+RRF) self-recall, n={len(idx)} ===")
    _recall([list(r) for r in dense], idx, "dense only", results, "dense_only")
    _recall([[d for d in r if d >= 0] for r in bm25], idx, "bm25 only",
            results, "bm25_only")
    for k in args.ks:
        fused = [_rrf([list(dr), [d for d in br if d >= 0]], k)[:DEPTH]
                 for dr, br in zip(dense, bm25)]
        _recall(fused, idx, f"hybrid rrf_k={k}", results, f"hybrid_rrf_k{k}")
    print("\n(caveat: BM25 sees the query text inside the target's indexed "
          "statement -> hybrid numbers are optimistic vs human queries; use "
          "the deltas, and the dense row for the honest baseline)")
    _save_results(results)


# --------------------------------------------------------------------- #
# stage: rerank  (Qwen3-Reranker-0.6B over the top-100 fused candidates)
# --------------------------------------------------------------------- #
def cmd_rerank(args):
    idx, _ = _load_queries()
    dense = np.load(os.path.join(WORK, f"dense_top{DEPTH}.npy"))
    bm25 = np.load(os.path.join(WORK, f"bm25_top{DEPTH}.npy"))
    n = min(args.n, len(idx))
    idx, dense, bm25 = idx[:n], dense[:n], bm25[:n]
    fused = [_rrf([list(dr), [d for d in br if d >= 0]], args.rrf_k)[:DEPTH]
             for dr, br in zip(dense, bm25)]

    wanted = set(int(i) for i in idx)
    for f in fused:
        wanted.update(int(d) for d in f)
    for dr in dense:                 # the dense-pool rerank pass needs these too
        wanted.update(int(d) for d in dr)
    print(f"# loading meta for {len(wanted)} rows (queries + candidates) ...",
          flush=True)
    rows = _meta_rows(wanted)
    qtexts = [(rows[int(i)].get("statement") or rows[int(i)].get("slogan") or "")
              for i in idx]

    from mathlas.retrieve.rerank import Qwen3Reranker, doc_rerank_text
    rr = Qwen3Reranker(device="cuda", batch_size=args.batch_size,
                       max_length=args.max_length)
    # Two rerank passes per query:
    #  * PRODUCTION: hybrid RRF top-100, full doc text (name+slogan+statement).
    #    In this self-retrieval proxy the doc statement EQUALS the query, so the
    #    absolute number is optimistic -- it validates the production wiring.
    #  * HONEST cross-rep: dense top-100, doc text = name+slogan ONLY (the same
    #    surface form the dense index stores) -- the reranker never sees the
    #    query text inside the doc, so this is the real body->slogan lift.
    reranked, dense_rr = [], []
    t0 = time.time()
    for i, (q, cand) in enumerate(zip(qtexts, fused)):
        texts = [doc_rerank_text(rows[d].get("name"), rows[d].get("slogan"),
                                 rows[d].get("statement")) for d in cand]
        order = np.argsort(-rr.score(q, texts), kind="stable")
        reranked.append([cand[int(j)] for j in order])
        dcand = [int(d) for d in dense[i]]
        dtexts = [doc_rerank_text(rows[d].get("name"), rows[d].get("slogan"),
                                  None) for d in dcand]
        dorder = np.argsort(-rr.score(q, dtexts), kind="stable")
        dense_rr.append([dcand[int(j)] for j in dorder])
        if i % 25 == 0:
            print(f"  reranked {i}/{n} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(WORK, f"rerank_lists_n{n}_k{args.rrf_k}.npz"),
             idx=idx, fused=np.array(fused), reranked=np.array(reranked),
             dense_rr=np.array(dense_rr))   # for offline failure analysis
    results = {}
    print(f"\n=== rerank (Qwen3-Reranker-0.6B over top-{DEPTH}), n={n} ===")
    _recall([list(r) for r in dense], idx, "dense only (no rerank)",
            results, "rr_dense_only")
    _recall(dense_rr, idx, "dense top-100 + rerank (slogan-only doc text)",
            results, "rr_dense_rerank_sloganonly")
    _recall(fused, idx, f"hybrid rrf_k={args.rrf_k} (no rerank)",
            results, "rr_hybrid_norerank")
    _recall(reranked, idx, f"hybrid rrf_k={args.rrf_k} + rerank (full doc text)",
            results, "rr_hybrid_rerank")
    # The SHIPPED behaviour (HybridRetriever._maybe_rerank): rerank order is
    # RRF-BLENDED with the first-stage order, never a replacement.
    _recall([_rrf([list(map(int, d)), r], args.rrf_k)
             for d, r in zip(dense, dense_rr)], idx,
            "RRF(dense, rerank) -- shipped blend", results, "rr_dense_blend")
    _recall([_rrf([f, r], args.rrf_k) for f, r in zip(fused, reranked)], idx,
            "RRF(hybrid, rerank) -- shipped blend", results, "rr_hybrid_blend")
    _save_results(results)


# --------------------------------------------------------------------- #
# stage: slice-queries  (sample + embed queries whose target row < --rows,
# so the dual-channel mechanism check needs no GPU later)
# --------------------------------------------------------------------- #
def cmd_slice_queries(args):
    rng = np.random.default_rng(args.seed)
    print(f"# scanning meta for eligible rows < {args.rows} ...", flush=True)
    elig, texts = [], {}
    with open(META) as f:
        for row, line in enumerate(f):
            if row >= args.rows:
                break
            r = json.loads(line)
            s = r.get("statement") or r.get("slogan") or ""
            if 20 <= len(s) <= 2000:
                elig.append(row)
                texts[row] = s
    sel = np.sort(rng.choice(np.array(elig), size=min(args.n, len(elig)),
                             replace=False))
    print(f"# sampled {len(sel)} of {len(elig)} eligible; embedding on "
          f"{args.device} ...", flush=True)
    from mathlas.embed import Qwen3Embedder
    emb = Qwen3Embedder(model="Qwen/Qwen3-Embedding-8B", device=args.device)
    Q = emb.encode([texts[int(i)] for i in sel], is_query=True)
    out = os.path.join(WORK, f"slice_queries_rows{args.rows}_n{len(sel)}_seed{args.seed}.npz")
    np.savez(out, idx=sel, Q=Q.astype(np.float16), rows=args.rows, seed=args.seed)
    print(f"# wrote {out}", flush=True)


# --------------------------------------------------------------------- #
# stage: dual  (statement channel mechanism check on the embedded prefix)
# --------------------------------------------------------------------- #
def cmd_dual(args):
    stmt_path = STMT_INDEX if os.path.exists(STMT_INDEX) else STMT_PARTIAL
    if args.stmt_index:
        stmt_path = args.stmt_index
    print(f"# statement channel: {stmt_path}", flush=True)
    S = _load_index_matrix(stmt_path)
    rows = S.shape[0]
    qfile = args.queries
    if qfile is None:
        cands = [p for p in os.listdir(WORK)
                 if p.startswith("slice_queries_") and p.endswith(".npz")]
        if not cands:
            sys.exit("run the slice-queries stage first")
        qfile = os.path.join(WORK, sorted(cands)[-1])
    d = np.load(qfile)
    idx, Q = d["idx"].astype(np.int64), d["Q"].astype(np.float32)
    keep = idx < rows
    idx, Q = idx[keep], Q[keep]
    print(f"# {len(idx)} queries with target inside the embedded prefix "
          f"({rows} rows)", flush=True)
    M = _load_index_matrix(rows=rows)         # slogan channel, same prefix

    dense = _dense_topk(M, Q, DEPTH, log="slogan")
    stmt = _dense_topk(S, Q, DEPTH, log="stmt")
    results = {}
    print(f"\n=== dual-channel mechanism check (prefix of {rows} docs), "
          f"n={len(idx)} ===")
    _recall([list(r) for r in dense], idx, "slogan channel only",
            results, "dual_slogan_only")
    _recall([list(r) for r in stmt], idx, "statement channel only",
            results, "dual_stmt_only")
    fused = [_rrf([list(a), list(b)], args.rrf_k)[:DEPTH]
             for a, b in zip(dense, stmt)]
    _recall(fused, idx, f"slogan+statement RRF (k={args.rrf_k})",
            results, "dual_rrf")
    # max-sim union: for each doc take max(slogan-sim, stmt-sim)
    print("# max-sim union ...", flush=True)
    out = []
    B = 64
    for b in range(0, len(idx), B):
        sims = np.maximum(M @ Q[b:b + B].T, S @ Q[b:b + B].T)
        part = np.argpartition(-sims, DEPTH, axis=0)[:DEPTH]
        for j in range(sims.shape[1]):
            col = part[:, j]
            out.append(list(col[np.argsort(-sims[col, j])]))
    _recall(out, idx, "max-sim(slogan, statement)", results, "dual_maxsim")
    results["dual_rows"] = int(rows)
    _save_results(results)


# --------------------------------------------------------------------- #
# stage: final  (assemble the headline comparison; if the FULL statement
# matrix exists, also run the full-3.68M dual-channel eval on the n=3000
# baseline sample so the dual-channel number is directly comparable to the
# 0.614/0.832 baseline)
# --------------------------------------------------------------------- #
def cmd_final(args):
    results = {}
    path = os.path.join(WORK, "results.json")
    if os.path.exists(path):
        results = json.load(open(path))

    if os.path.exists(STMT_INDEX):
        if "full_dual_maxsim" not in results or args.rerun_dual:
            print("# FULL statement matrix found -> full-corpus dual-channel "
                  "eval on the n=3000 baseline sample ...", flush=True)
            idx, Q = _load_queries()
            M = _load_index_matrix()
            S = _load_index_matrix(STMT_INDEX)
            out = []
            B = 64
            t0 = time.time()
            for b in range(0, len(idx), B):
                sims = np.maximum(M @ Q[b:b + B].T, S @ Q[b:b + B].T)
                part = np.argpartition(-sims, DEPTH, axis=0)[:DEPTH]
                for j in range(sims.shape[1]):
                    col = part[:, j]
                    out.append(list(col[np.argsort(-sims[col, j])]))
                if (b // B) % 8 == 0:
                    print(f"  dual {b + sims.shape[1]}/{len(idx)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            del S
            new = {}
            _recall(out, idx, "FULL dual-channel max-sim (dense)", new,
                    "full_dual_maxsim")
            # and fused with BM25 at the tuned k
            bm25 = np.load(os.path.join(WORK, f"bm25_top{DEPTH}.npy"))
            fused = [_rrf([o, [d for d in br if d >= 0]], 10)[:DEPTH]
                     for o, br in zip(out, bm25)]
            _recall(fused, idx, "FULL dual-dense + BM25 rrf_k=10", new,
                    "full_dual_hybrid_k10")
            _save_results(new)
            results.update(new)
    else:
        print("# full statement matrix not built yet "
              f"({STMT_INDEX} missing) -- dual-channel full-corpus number "
              "pending; prefix mechanism-check numbers shown instead.",
              flush=True)

    print("\n================ RETRIEVAL UPGRADES vs BASELINE ================")
    print("BASELINE (eval_dense_recall.py, n=3000, seed 0, full 3.68M index):")
    print("  dense slogan-channel    R@1=0.614  R@10=0.832   (reconfirmed 2026-06-09)")
    rows = [
        ("dense_only", "dense only (this harness, same sample)"),
        ("hybrid_rrf_k60", "hybrid dense+BM25, rrf_k=60 (old default)"),
        ("hybrid_rrf_k10", "hybrid dense+BM25, rrf_k=10 (NEW default)"),
        ("rr_dense_only", "dense only [n=1000 subset]"),
        ("rr_dense_rerank_sloganonly", "dense + rerank, replacement [n=1000]"),
        ("rr_hybrid_norerank", "hybrid k=10, no rerank [n=1000]"),
        ("rr_hybrid_rerank", "hybrid k=10 + rerank, replacement [n=1000]"),
        ("rr_dense_blend", "RRF(dense, rerank) SHIPPED blend [n=1000]"),
        ("rr_hybrid_blend", "RRF(hybrid, rerank) blend [n=1000]"),
        ("dual_slogan_only", "slogan-only dense [200k prefix, n=1000]"),
        ("dual_stmt_only", "statement-only dense [200k prefix]"),
        ("dual_maxsim", "dual-channel max-sim [200k prefix]"),
        ("dual_rrf", "dual as 3rd RRF channel [200k prefix]"),
        ("full_dual_maxsim", "FULL dual-channel max-sim (n=3000)"),
        ("full_dual_hybrid_k10", "FULL dual-dense + BM25 rrf_k=10 (n=3000)"),
    ]
    for key, label in rows:
        r = results.get(key)
        if r:
            print(f"  {label:<46} R@1={r['R@1']:.3f}  R@10={r['R@10']:.3f}  "
                  f"MRR={r['MRR']:.3f}  (n={r['n']})")
    print("(hybrid/statement-channel numbers on this self-retrieval proxy "
          "carry an exact-text advantage -- see the caveat in the module "
          "docstring and docs/RETRIEVAL_UPGRADE_NOTES.md)")


def main():
    os.makedirs(WORK, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ranks")
    p = sub.add_parser("rrf")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 60])
    p = sub.add_parser("rerank")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048)
    p = sub.add_parser("slice-queries")
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("dual")
    p.add_argument("--stmt-index", default=None)
    p.add_argument("--queries", default=None)
    p.add_argument("--rrf-k", type=int, default=10)
    p = sub.add_parser("final")
    p.add_argument("--rerun-dual", action="store_true")
    args = ap.parse_args()
    {"ranks": cmd_ranks, "rrf": cmd_rrf, "rerank": cmd_rerank,
     "slice-queries": cmd_slice_queries, "dual": cmd_dual,
     "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    main()
