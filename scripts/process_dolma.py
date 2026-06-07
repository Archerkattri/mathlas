#!/usr/bin/env python3
"""Dedup the slogan-bearing dolma docs into a clean workdir ready for embedding.

NO length / content filtering — every slogan is kept regardless of length. The only
removals are EXACT duplicates:
  - within dolma  (same normalized statement appearing twice → redundant index entry)
  - vs the old corpus (same normalized statement already indexed → train/test leakage)
Writes <workdir>/clean/docs_*.jsonl with the kept docs (same schema).
Use --keep-dups to disable even that and pass through literally everything.
"""
import argparse
import glob
import json
import os


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="reference/downloads/dolma_index_build")
    ap.add_argument("--base-meta", default="reference/downloads/index_full.meta.jsonl")
    ap.add_argument("--keep-dups", action="store_true", help="pass through everything, no dedup")
    a = ap.parse_args()

    clean = os.path.join(a.workdir, "clean")
    os.makedirs(clean, exist_ok=True)

    old_stmt: set = set()
    if not a.keep_dups:
        for line in open(a.base_meta):
            old_stmt.add(norm(json.loads(line).get("statement")))
        print(f"old corpus: {len(old_stmt):,} statements hashed", flush=True)

    seen_stmt: set = set()
    n = kept = d_din = d_dold = 0
    for s in sorted(glob.glob(os.path.join(a.workdir, "docs_*.jsonl"))):
        out = []
        for line in open(s):
            r = json.loads(line)
            if not r.get("slogan"):
                continue
            n += 1
            if a.keep_dups:
                out.append(line)
                kept += 1
                continue
            ks = norm(r.get("statement"))
            if ks in old_stmt:
                d_dold += 1
                continue
            if ks in seen_stmt:
                d_din += 1
                continue
            seen_stmt.add(ks)
            out.append(line)
            kept += 1
        if out:
            with open(os.path.join(clean, os.path.basename(s)), "w") as f:
                f.writelines(out)

    print(f"dolma slogan docs: {n:,}")
    print(f"  dropped dup-vs-old (exact statement):   {d_dold}")
    print(f"  dropped dup-in-dolma (exact statement): {d_din}")
    print(f"  KEPT: {kept:,}  ->  {clean}/   (no length/content filtering)")


if __name__ == "__main__":
    main()
