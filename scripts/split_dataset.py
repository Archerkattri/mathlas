#!/usr/bin/env python3
"""Partition the merged index into train/val/test (90/5/5), stratified by source.

Deterministic: the split is a function of md5(doc_id), so re-running is stable and adding
docs later never reshuffles existing ones. Stratification falls out of the uniform hash —
we verify per-source proportions and print them. Also extracts the 110 real-world gold
queries (theorems-test.parquet) as a separate held-out benchmark.

Writes <out>/{train,val,test}.txt (doc_id per line) + <out>/gold_queries.jsonl.
"""
import argparse
import collections
import hashlib
import json
import os


def bucket(src: str) -> str:
    return "dolma" if "dolma" in (src or "").lower() else "base"


def assign(doc_id: str) -> str:
    h = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 90 else ("val" if h < 95 else "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="reference/downloads/index_full.meta.jsonl")
    ap.add_argument("--gold", default="reference/theorem-search-dataset/theorems-test.parquet")
    ap.add_argument("--out", default="reference/downloads/splits")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fh = {sp: open(os.path.join(a.out, sp + ".txt"), "w") for sp in ("train", "val", "test")}
    stats: collections.Counter = collections.Counter()
    n = 0
    for line in open(a.meta):
        r = json.loads(line)
        did = str(r["doc_id"])
        sp = assign(did)
        fh[sp].write(did + "\n")
        b = bucket(r.get("source"))
        stats[(b, sp)] += 1
        stats[("ALL", sp)] += 1
        n += 1
    for f in fh.values():
        f.close()

    # 110 real-world gold queries (kept separate from the auto splits)
    n_gold = 0
    try:
        import pandas as pd
        g = pd.read_parquet(a.gold)
        with open(os.path.join(a.out, "gold_queries.jsonl"), "w") as gf:
            for _, row in g.iterrows():
                gf.write(json.dumps({k: (None if pd.isna(row[k]) else row[k]) for k in g.columns}) + "\n")
        n_gold = len(g)
    except Exception as e:  # noqa: BLE001
        print(f"[gold] skipped ({e})")

    print(f"merged index rows: {n:,}")
    print("split (doc counts), stratified by source:")
    for b in ("base", "dolma", "ALL"):
        t, v, te = stats[(b, "train")], stats[(b, "val")], stats[(b, "test")]
        tot = t + v + te or 1
        print(f"  {b:5s}: train {t:>9,} ({100*t/tot:4.1f}%)  "
              f"val {v:>7,} ({100*v/tot:4.1f}%)  test {te:>7,} ({100*te/tot:4.1f}%)  | total {tot:,}")
    print(f"gold real-world queries: {n_gold} -> {a.out}/gold_queries.jsonl")


if __name__ == "__main__":
    main()
