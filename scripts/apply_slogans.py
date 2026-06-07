#!/usr/bin/env python3
"""Fold agent-generated slogans back into the dolma docs shards.

The slogan workflow writes `slogan_out/slog_*.jsonl` ({doc_id, slogan}). This applies them:
for each `docs_*.jsonl` record whose doc_id has a slogan, set `slogan` and rewrite
`embed_text = "name -- slogan"` (or just the slogan) — matching the 1.34M index's regime, so the
GPU embed step encodes the NL slogan instead of the raw LaTeX statement. Idempotent + resumable
(only rewrites a shard when something changed).

Run:  python3 scripts/apply_slogans.py [--workdir reference/downloads/dolma_index_build]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(ME, "reference", "downloads", "dolma_index_build"))
    args = ap.parse_args()
    wd = args.workdir

    slmap: dict[str, str] = {}
    for f in glob.glob(os.path.join(wd, "slogan_out", "slog_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sl = (r.get("slogan") or "").strip()
            if r.get("doc_id") and sl:
                slmap[r["doc_id"]] = sl
    print(f"loaded {len(slmap):,} slogans from slogan_out/")

    applied = 0
    touched = 0
    for shard in sorted(glob.glob(os.path.join(wd, "docs_*.jsonl"))):
        rows = [json.loads(l) for l in open(shard)]
        changed = False
        for r in rows:
            sl = slmap.get(r.get("doc_id"))
            if sl and r.get("slogan") != sl:
                r["slogan"] = sl
                r["embed_text"] = f"{r['name']} -- {sl}" if r.get("name") else sl
                applied += 1
                changed = True
        if changed:
            with open(shard + ".tmp", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(shard + ".tmp", shard)
            touched += 1
    print(f"applied {applied:,} slogans across {touched} updated shards")


if __name__ == "__main__":
    main()
