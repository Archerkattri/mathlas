#!/usr/bin/env python3
"""Post-process raw theorem shards into clean records (the CPU, no-id, no-GPU steps).

Applies the steps from docs/05_open_dataset.md that need neither arXiv-id recovery, NL slogans,
nor a GPU:
  - clean_statement : strip \\label (kept in `label`), unwrap \\textcolor{c}{X}->X,
                      \\cite/\\ref -> [REF], drop \\includegraphics bodies, collapse whitespace
  - sanitize env    : reject markup / single-char / overlong env_type -> 'unknown';
                      map custom names to the canonical taxonomy by substring
  - quality filter  : drop empty / <30-char statements and figure-bearing bodies
  - per-paper dedup : the raw corpus is written in paper order, so intra-paper duplicates
                      (the dominant extraction-bug dups) are contiguous — dedup on a per-paper
                      window (memory-light; cross-paper near-dup is deferred to a MinHash pass)

Reads  reference/downloads/datasets/arxiv_fulltext_theorems/theorems_*.jsonl
Writes reference/downloads/datasets/arxiv_fulltext_theorems_clean/theorems_NNNNN.jsonl

Run:  python3 scripts/postprocess_corpus.py [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ME, "reference", "downloads", "datasets", "arxiv_fulltext_theorems")
DST = os.path.join(ME, "reference", "downloads", "datasets", "arxiv_fulltext_theorems_clean")
SHARD_SIZE = 500_000

CANONICAL = (
    "theorem", "lemma", "proposition", "corollary", "definition", "conjecture",
    "claim", "remark", "example", "fact", "observation", "note", "notation",
    "axiom", "hypothesis", "assumption",
)

_LABEL = re.compile(r"\\label\{[^}]*\}")
_TEXTCOLOR = re.compile(r"\\(?:textcolor|colorbox)\{[^}]*\}\{")
_CITE = re.compile(r"\\cite(?:\[[^\]]*\])?\{[^}]+\}")
_REF = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\{[^}]+\}")
_GRAPHICS = re.compile(r"\\includegraphics")
_WS = re.compile(r"\s{2,}")


def clean_statement(stmt: str):
    """Return a display/embedding-clean statement, or None if it should be filtered out."""
    if not stmt:
        return None
    if _GRAPHICS.search(stmt):
        return None  # a figure leaked into the theorem body — not a statement
    stmt = _LABEL.sub("", stmt)
    stmt = _TEXTCOLOR.sub("", stmt)          # keep the wrapped content (its closing } is benign)
    stmt = _CITE.sub("[REF]", stmt)
    stmt = _REF.sub("[REF]", stmt)
    stmt = _WS.sub(" ", stmt).strip()
    return stmt if len(stmt) >= 30 else None


def sanitize_env(env):
    """Canonicalise the env_type; markup/single-char/overlong -> 'unknown'."""
    if not env or "\\" in env or len(env) > 30 or len(env) < 2:
        return "unknown"
    e = env.lower()
    for c in CANONICAL:
        if c in e:
            return c
    return "unknown"


def _norm(stmt: str) -> str:
    return re.sub(r"\s+", " ", stmt).lower().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N input records (test)")
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(SRC, "theorems_*.jsonl")))
    if not args.dry_run:
        os.makedirs(DST, exist_ok=True)

    n_in = n_out = dropped_clean = dropped_dup = sanitized = 0
    cur_paper = None
    paper_seen: set[str] = set()
    out_idx = 0
    out_count = 0
    fh = None

    def _open(idx):
        return open(os.path.join(DST, f"theorems_{idx:05d}.jsonl"), "w", buffering=1)

    for s in shards:
        for line in open(s):
            if args.limit and n_in >= args.limit:
                break
            n_in += 1
            r = json.loads(line)

            pid = r.get("paper_id")
            if pid != cur_paper:               # records are paper-contiguous -> reset window
                cur_paper = pid
                paper_seen = set()

            cleaned = clean_statement(r.get("statement") or "")
            if cleaned is None:
                dropped_clean += 1
                continue

            k = _norm(cleaned)
            if k in paper_seen:
                dropped_dup += 1
                continue
            paper_seen.add(k)

            env = sanitize_env(r.get("env_type"))
            if env != r.get("env_type"):
                sanitized += 1
            r["statement"] = cleaned
            r["env_type"] = env

            n_out += 1
            if not args.dry_run:
                if fh is None or out_count >= SHARD_SIZE:
                    if fh:
                        fh.close()
                    fh = _open(out_idx)
                    out_idx += 1
                    out_count = 0
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_count += 1
        if args.limit and n_in >= args.limit:
            break
    if fh:
        fh.close()

    def pct(x):
        return f"{x:>8,} ({100 * x / max(n_in, 1):4.1f}%)"

    print(f"in:            {n_in:>8,}")
    print(f"out:           {pct(n_out)}")
    print(f"dropped clean: {pct(dropped_clean)}  (empty/<30/figure)")
    print(f"dropped dup:   {pct(dropped_dup)}  (intra-paper)")
    print(f"env sanitized: {pct(sanitized)}")
    if not args.dry_run:
        print(f"written -> {DST}")


if __name__ == "__main__":
    main()
