#!/usr/bin/env python3
"""Comprehensive retrieval benchmark on the held-out test split.

Exact (PQ-free) dense cosine over the full 1.6M index. Each test theorem is queried two ways --
by its raw STATEMENT (cross-representation) and by its SLOGAN (NL-query form) -- and we check
whether its own row is retrieved. Reports Recall@{1,5,10,20} + MRR@20, overall AND split by
source (base=Stacks/formal, dolma=arXiv). Dual-GPU: `all` spawns one embed worker per GPU,
then does the exact search on cuda:0.

  python eval_benchmark.py all --procs 2          # prep -> 2-GPU embed -> exact search + report
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

WD = "reference/downloads"
META = f"{WD}/index_full.meta.jsonl"
DENSE = f"{WD}/index_full_dense.npz"
TEST = f"{WD}/splits/test.txt"
PREP = "/tmp/bench_prep.npz"
QV = "/tmp/bench_qv_{}.npz"
KS = (1, 5, 10, 20)


def bucket(s: str) -> str:
    return "dolma" if "dolma" in (s or "").lower() else "base"


def cmd_prep(args):
    keep = set(l.strip() for l in open(TEST))
    if args.n:
        import random
        keep = set(random.Random(0).sample(sorted(keep), min(args.n, len(keep))))
    ids, rows, stmt, slog, src = [], [], [], [], []
    row = 0
    for line in open(META):
        r = json.loads(line)
        did = str(r["doc_id"])
        if did in keep:
            ids.append(did)
            rows.append(row)
            stmt.append((r.get("statement") or "")[:1200])
            slog.append((r.get("slogan") or "")[:1200])
            src.append(bucket(r.get("source")))
        row += 1
    np.savez(PREP, rows=np.array(rows), stmt=np.array(stmt, dtype=object),
             slog=np.array(slog, dtype=object), src=np.array(src), n_index=row)
    print(f"[prep] {len(ids):,} test docs | index rows {row:,}", flush=True)


def cmd_embed(args):
    sys.path.insert(0, ".")
    d = np.load(PREP, allow_pickle=True)
    n = len(d["rows"])
    mine = [i for i in range(n) if i % args.procs == args.proc_id]
    from mathlas.embed import Qwen3Embedder
    emb = Qwen3Embedder(model="Qwen/Qwen3-Embedding-8B", device="cuda")

    def enc(texts):
        import torch
        import time
        idxs = [i for i in mine if str(texts[i]).strip()]
        out = []
        _t0 = time.time()
        for _k, j in enumerate(range(0, len(idxs), 32)):
            out.append(np.array(emb.encode([str(texts[i]) for i in idxs[j:j + 32]], is_query=True)))
            if (_k + 1) % 50 == 0:
                _d = j + 32
                print(f"[embed p{args.proc_id}] {_d}/{len(idxs)} ({_d / (time.time() - _t0):.0f}/s, "
                      f"{(len(idxs) - _d) / max(1e-9, _d / (time.time() - _t0)):.0f}s left)", flush=True)
        vec = np.vstack(out).astype("float16") if out else np.zeros((0, 4096), "float16")
        return np.array(idxs), vec

    si, sv = enc(d["stmt"])
    li, lv = enc(d["slog"])
    np.savez(QV.format(args.proc_id), s_idx=si, s_vec=sv, l_idx=li, l_vec=lv)
    print(f"[embed p{args.proc_id}] statement {len(si)} | slogan {len(li)}", flush=True)


def cmd_finalize(args):
    import torch
    d = np.load(PREP, allow_pickle=True)
    rows, src = d["rows"], d["src"]
    M = np.load(DENSE)["matrix"]
    print(f"[search] matrix {M.shape} -> cuda:0", flush=True)
    Mt = torch.nn.functional.normalize(torch.from_numpy(M).to("cuda:0").half(), dim=1)
    del M
    qd = [np.load(QV.format(p)) for p in range(args.procs)]

    def report(field, ik, vk):
        gi = np.concatenate([q[ik] for q in qd])
        qv = np.concatenate([q[vk] for q in qd])
        q = torch.nn.functional.normalize(torch.from_numpy(qv).to("cuda:0").half(), dim=1)
        gold = torch.tensor(rows[gi], device="cuda:0")
        bsrc = src[gi]
        hit = {k: np.zeros(len(gi), bool) for k in KS}
        rr = np.zeros(len(gi))
        for i in range(0, len(gi), 256):
            topk = (q[i:i + 256] @ Mt.T).topk(max(KS), dim=1).indices
            match = (topk == gold[i:i + 256].unsqueeze(1)).cpu().numpy()
            for k in KS:
                hit[k][i:i + 256] = match[:, :k].any(1)
            for j in range(match.shape[0]):
                w = np.where(match[j])[0]
                rr[i + j] = 1.0 / (w[0] + 1) if len(w) else 0.0
        print(f"\n=== query = {field}  (n={len(gi):,}) ===", flush=True)
        for grp in ("ALL", "base", "dolma"):
            m = np.ones(len(gi), bool) if grp == "ALL" else (bsrc == grp)
            if not m.any():
                continue
            cells = "  ".join(f"R@{k} {hit[k][m].mean():.3f}" for k in KS)
            print(f"  {grp:5s} (n={m.sum():>6,}): {cells}  MRR {rr[m].mean():.3f}", flush=True)

    report("STATEMENT", "s_idx", "s_vec")
    report("SLOGAN", "l_idx", "l_vec")


def cmd_all(args):
    cmd_prep(args)
    procs = []
    for g in range(args.procs):
        e = dict(os.environ)
        e["CUDA_VISIBLE_DEVICES"] = str(g)
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "embed",
             "--procs", str(args.procs), "--proc-id", str(g)], env=e))
    if any(p.wait() for p in procs):
        sys.exit("[all] an embed worker failed")
    cmd_finalize(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "prep", "embed", "finalize"])
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--proc-id", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="subsample test docs (0 = full test split)")
    args = ap.parse_args()
    {"all": cmd_all, "prep": cmd_prep, "embed": cmd_embed, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
