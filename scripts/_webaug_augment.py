#!/usr/bin/env python3
"""Augment stage: load index (+register in server cache), ingest worklist findings
with caller-supplied dense_vec (embedded by the SAME loaded encoder), then re-run
the full 110 with live findings fused via server._merge_live_findings."""
from __future__ import annotations
import argparse, json, os, re, sys, time
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
from scripts._webaug_eval import (_load_test, _load_index, _reachable, _eval)  # reuse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--index", default="reference/downloads/index_full_dense.npz")
    ap.add_argument("--test", default="reference/theorem-search-dataset/theorems-test.parquet")
    ap.add_argument("--worklist", default="reference/downloads/splits/_findings_worklist.json")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args=ap.parse_args()

    test=_load_test(args.test)
    R=_load_index(args.index, args.device)
    from mathlas.webaug import add_finding
    import numpy as np

    work=json.load(open(args.worklist))
    print(f"# ingesting {len(work)} findings (dense_vec from the loaded encoder) ...", flush=True)
    added=0
    for w in work:
        # slogan = the real web-found theorem denotation; dense_vec embeds the slogan
        # in the SAME space as the served index (doc-side encoding, not query-side).
        vec = R.embedder.encode([w["slogan"]], is_query=False)[0].astype(np.float32)
        res = add_finding(statement=w["statement"], slogan=w["slogan"],
                          source=w["source"], name=w["name"],
                          dense_vec=[float(x) for x in vec])
        if res.ok and res.dense_added: added+=1
        else: print("  ! add failed:", res.note[:120])
    print(f"# added {added}/{len(work)} dense findings; corpus now has {res.n_findings} findings", flush=True)

    ph,th,perq=_eval(R, test, args.k, use_findings=True)
    n=len(test); nn=n or 1
    print(f"\n=== STAGE augmented Hit@{args.k} (full {n}) ===")
    print(f"  theorem-level: {th}/{n} = {100*th/nn:.1f}%   (TheoremSearch 45.0%, Gemini3Pro 27.0%, ChatGPT5.2 19.8%)")
    print(f"  paper-level:   {ph}/{n} = {100*ph/nn:.1f}%   (TheoremSearch 56.8%, Google 37.8%)")

if __name__=="__main__": main()
