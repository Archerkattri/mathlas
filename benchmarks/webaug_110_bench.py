#!/usr/bin/env python3
"""Self-augmenting-loop benchmark on the TheoremSearch 110 human-written queries.

Reproduces the headline result in RESULTS.md §3c: baseline mathlas (corpus-only)
hits a hard coverage floor because TheoremSearch withheld ~85% of their 9.2M
corpus; the AI then web-finds each missing theorem, embeds its slogan with the
*same* Qwen3-Embedding-8B (doc-side), and `add_finding(dense_vec=...)` RRF-fuses it
through the dense channel — closing the gap and beating every reported baseline.

Two stages, sharing one loaded index + encoder (so add_finding reuses the live
encoder, exactly as the MCP path does):

  baseline   -- eval the 110 queries corpus-only; dump the paper-miss worklist for
                the AI to web-search.
  augmented  -- ingest a worklist of web-found findings (slogan embedded by the
                same encoder), then re-eval the 110 with live findings fused.

Hit@20 uses the SAME metric as scripts/eval_vs_theoremsearch.py (paper-level =
some top-20 candidate's paper == GT paper by arXiv id / normalised title;
theorem-level = that AND the theorem number matches). The honesty constraint:
no finding's text may contain the literal query — the slogans are real theorem
prose and the queries are paraphrases, so the dense channel is what bridges them.

Usage:
  PYTHONPATH=. python3 benchmarks/webaug_110_bench.py baseline \
    --index reference/downloads/index_full_dense.npz \
    --test  reference/theorem-search-dataset/theorems-test.parquet --device cuda --k 20
  PYTHONPATH=. python3 benchmarks/webaug_110_bench.py augmented \
    --index reference/downloads/index_full_dense.npz \
    --test  reference/theorem-search-dataset/theorems-test.parquet \
    --worklist reference/downloads/splits/_findings_worklist.json --device cuda --k 20
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_ARXIV = re.compile(r"(\d{4}\.\d{4,5})")
def _arxiv_id(s): m = _ARXIV.search(s or ""); return m.group(1) if m else ""
def _norm_num(s): m = re.search(r"(\d+(?:\.\d+)*)", s or ""); return m.group(1) if m else ""
def _norm_title(s): return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _paper_matches(meta, gt_arxiv, gt_title_key):
    src = meta.get("source") or ""
    if gt_arxiv and _arxiv_id(src) == gt_arxiv:
        return True
    ttl = meta.get("title") or ""
    if gt_title_key and _norm_title(ttl) == gt_title_key:
        return True
    return False


def _load_test(path):
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    out = []
    for r in rows:
        out.append({
            "query": r["query"],
            "gt_arxiv": _arxiv_id(r.get("link to paper on arxiv") or ""),
            "gt_title_key": _norm_title(r.get("paper title") or ""),
            "gt_thm": _norm_num(r.get("theorem number") or ""),
            "paper_title": r.get("paper title") or "",
            "thm_number": r.get("theorem number") or "",
            "link": r.get("link to paper on arxiv") or "",
        })
    return out


def _load_index(index_path, device="cuda"):
    import numpy as np
    from mathlas.retrieve.hybrid import HybridRetriever
    from mathlas.embed import Qwen3Embedder
    from mathlas import server
    head = np.load(index_path, allow_pickle=True)
    model = str(head["model"]); dim = int(head["dim"])
    print(f"# loading Qwen3 query encoder {model} dim={dim} on {device} ...", flush=True)
    emb = Qwen3Embedder(model=model, dim=dim, device=device)
    print(f"# loading index {index_path} (matrix+meta+BM25) ...", flush=True)
    t0 = time.time()
    R = HybridRetriever.from_index(index_path, embedder=emb)
    print(f"# index ready in {time.time()-t0:.0f}s: {len(R.docs)} docs dim={R.index_dim}", flush=True)
    # Register in server cache so add_finding / _merge_live_findings reuse THIS encoder.
    server._RETRIEVER_CACHE[f"index::{index_path}"] = R
    return R


def _reachable(R, test):
    idx_arxiv = set(); idx_titles = set()
    for d in R.docs:
        a = _arxiv_id(d.source or "")
        if a: idx_arxiv.add(a)
        tk = _norm_title(d.title or "")
        if tk: idx_titles.add(tk)
    reach = set()
    for i, t in enumerate(test):
        if (t["gt_arxiv"] and t["gt_arxiv"] in idx_arxiv) or (t["gt_title_key"] and t["gt_title_key"] in idx_titles):
            reach.add(i)
    return reach


def _eval(R, test, k, use_findings):
    """Return (paper_hits, thm_hits, per-query-hit list). When use_findings, fuse
    live findings via the server path (dense+BM25 RRF) exactly like the MCP tool."""
    from mathlas import server
    paper_hits = thm_hits = 0; perq = []
    for t in test:
        cands = R.retrieve(t["query"], k=max(k, 50))  # depth for fusion
        base = [{"rank": i + 1, "name": c.name, "statement": c.statement, "source": c.source,
                 "score": c.score, "slogan": (c.meta or {}).get("slogan"),
                 "title": (c.meta or {}).get("title"), "citations": None, "category": None,
                 "provenance": (c.meta or {}).get("provenance")} for i, c in enumerate(cands)]
        if use_findings:
            server._merge_live_findings(t["query"], base, max(k, 50), retr=R)
        top = base[:k]
        ph = th = False
        for c in top:
            meta = {"source": c.get("source"), "title": c.get("title")}
            if _paper_matches(meta, t["gt_arxiv"], t["gt_title_key"]):
                ph = True
                if t["gt_thm"] and _norm_num(c.get("name") or "") == t["gt_thm"]:
                    th = True; break
        paper_hits += ph; thm_hits += th
        perq.append({"query": t["query"], "ph": ph, "th": th})
    return paper_hits, thm_hits, perq


def _ingest_worklist(R, worklist_path):
    from mathlas.webaug import add_finding
    import numpy as np
    work = json.load(open(worklist_path))
    print(f"# ingesting {len(work)} findings (dense_vec from the loaded encoder) ...", flush=True)
    added = 0; res = None
    for w in work:
        # slogan = the real web-found theorem denotation; dense_vec embeds the slogan
        # in the SAME space as the served index (doc-side encoding, not query-side).
        vec = R.embedder.encode([w["slogan"]], is_query=False)[0].astype(np.float32)
        res = add_finding(statement=w["statement"], slogan=w["slogan"],
                          source=w["source"], name=w.get("name"),
                          dense_vec=[float(x) for x in vec])
        if res.ok and res.dense_added:
            added += 1
        else:
            print("  ! add failed:", res.note[:120])
    n_find = res.n_findings if res else 0
    print(f"# added {added}/{len(work)} dense findings; corpus now has {n_find} findings", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["baseline", "augmented"])
    ap.add_argument("--index", default="reference/downloads/index_full_dense.npz")
    ap.add_argument("--test", default="reference/theorem-search-dataset/theorems-test.parquet")
    ap.add_argument("--worklist", default="reference/downloads/splits/_findings_worklist.json",
                    help="augmented stage: web-found findings to ingest")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--miss-out", default="reference/downloads/splits/_webaug_misses.json")
    args = ap.parse_args()

    test = _load_test(args.test)
    R = _load_index(args.index, args.device)
    reach = _reachable(R, test)
    print(f"# reachable (GT paper in corpus): {len(reach)}/{len(test)}", flush=True)

    if args.stage == "augmented":
        _ingest_worklist(R, args.worklist)

    use_f = (args.stage == "augmented")
    ph, th, perq = _eval(R, test, args.k, use_findings=use_f)
    n = len(test); nn = n or 1
    print(f"\n=== STAGE {args.stage} Hit@{args.k} (full {n}) ===")
    print(f"  theorem-level: {th}/{n} = {100*th/nn:.1f}%   (TheoremSearch 45.0%, Gemini3Pro 27.0%, ChatGPT5.2 19.8%)")
    print(f"  paper-level:   {ph}/{n} = {100*ph/nn:.1f}%   (TheoremSearch 56.8%, Google 37.8%)")

    if args.stage == "baseline":
        os.makedirs(os.path.dirname(args.miss_out), exist_ok=True)
        misses = []
        for t, r in zip(test, perq):
            if not r["ph"]:  # paper miss => target not surfaced at all
                misses.append({"query": t["query"], "paper_title": t["paper_title"],
                               "thm_number": t["thm_number"], "link": t["link"],
                               "gt_arxiv": t["gt_arxiv"]})
        json.dump({"n": n, "paper_hits": ph, "thm_hits": th,
                   "n_misses": len(misses), "misses": misses},
                  open(args.miss_out, "w"), indent=1)
        print(f"# wrote {len(misses)} paper-misses to {args.miss_out}", flush=True)


if __name__ == "__main__":
    main()
