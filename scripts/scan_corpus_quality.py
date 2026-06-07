#!/usr/bin/env python3
"""Data-quality scan of the arxiv_fulltext_theorems corpus.

Surfaces what needs post-processing before this can be an open-source theorem dataset
(better than TheoremSearch). Light CPU (samples N records). Run:
    python3 scripts/scan_corpus_quality.py [--n 50000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ME, "reference", "downloads", "datasets", "arxiv_fulltext_theorems")
ARXIV = re.compile(r"^\d{4}\.\d{4,5}|^[a-z\-]+/\d{7}")
NOISE = re.compile(r"\\(label|textcolor|cite|ref|input|includegraphics)\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    args = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(D, "theorems_*.jsonl")))
    n = empty = short = no_slogan = arxiv = hsh = noise = dup = 0
    lens = []
    envs = Counter()
    seen = set()
    for s in shards:
        if n >= args.n:
            break
        for line in open(s):
            if n >= args.n:
                break
            r = json.loads(line)
            n += 1
            st = (r.get("statement") or "")
            if not st.strip():
                empty += 1
            if len(st) < 30:
                short += 1
            lens.append(len(st))
            sl = r.get("slogan")
            if not sl or sl == st:
                no_slogan += 1
            pid = str(r.get("paper_id") or "")
            if ARXIV.match(pid):
                arxiv += 1
            elif len(pid) >= 16:
                hsh += 1
            if NOISE.search(st):
                noise += 1
            envs[r.get("env_type")] += 1
            k = st[:120]
            if k in seen:
                dup += 1
            else:
                seen.add(k)

    def pct(x):
        return f"{x:>6} ({100 * x / n:4.1f}%)"

    lens.sort()
    print(f"scanned {n} records across {len(shards)} shards")
    print(f"  empty statement:          {pct(empty)}")
    print(f"  short (<30 chars):        {pct(short)}")
    print(f"  no NL slogan (slogan==statement / missing): {pct(no_slogan)}")
    print(f"  arxiv-id paper_id:        {pct(arxiv)}")
    print(f"  hash paper_id (no arxiv): {pct(hsh)}")
    print(f"  LaTeX noise in statement: {pct(noise)}")
    print(f"  duplicate stmt (120-char prefix): {pct(dup)}")
    print(f"  statement length: median {lens[len(lens)//2]}, p90 {lens[int(len(lens)*0.9)]}")
    print(f"  env_type top-10: {envs.most_common(10)}")


if __name__ == "__main__":
    main()
