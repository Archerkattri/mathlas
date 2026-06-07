#!/usr/bin/env python3
"""Write MANIFEST.md for the arxiv_fulltext_theorems corpus.

Run after build_arxiv_fulltext_corpus.py completes (or to checkpoint).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ME = Path(__file__).resolve().parent.parent
OUT_DIR = ME / "reference/downloads/datasets/arxiv_fulltext_theorems"
MANIFEST_PATH = OUT_DIR / "MANIFEST.md"


def main():
    if not OUT_DIR.exists():
        print(f"ERROR: {OUT_DIR} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load progress
    prog_file = OUT_DIR / "progress.json"
    prog = {}
    if prog_file.exists():
        with open(prog_file) as f:
            prog = json.load(f)

    # Count all records across all shards
    shard_files = sorted(OUT_DIR.glob("theorems_*.jsonl"))
    print(f"Counting records in {len(shard_files)} shard(s)...")

    total = 0
    env_type_counts: Counter = Counter()
    named_count = 0
    labeled_count = 0
    shard_counts = []

    for sf in shard_files:
        n = 0
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    env_type_counts[rec.get("env_type", "?")] += 1
                    if rec.get("name"):
                        named_count += 1
                    if rec.get("label"):
                        labeled_count += 1
                    n += 1
                except json.JSONDecodeError:
                    pass
        shard_counts.append((sf.name, n))
        total += n
        print(f"  {sf.name}: {n:,} records")

    # Disk usage
    try:
        result = subprocess.run(
            ['du', '-sh', str(OUT_DIR)], capture_output=True, text=True
        )
        disk_used = result.stdout.split('\t')[0].strip()
    except Exception:
        disk_used = "unknown"

    # Format
    now = datetime.utcnow().strftime("%Y-%m-%d")
    papers_processed = prog.get("papers_processed", "?")
    start_ts = prog.get("start_time")
    end_ts = prog.get("last_update")
    elapsed_h = (end_ts - start_ts) / 3600 if start_ts and end_ts else None

    lines = [
        f"# Corpus Manifest — arxiv_fulltext_theorems",
        f"",
        f"**Generated**: {now}",
        f"**Source**: `emozilla/dolma-v1_7-arxiv` (HuggingFace, full arXiv LaTeX)",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total theorems | {total:,} |",
        f"| Papers processed | {papers_processed:,} |",
        f"| Shards | {len(shard_files)} |",
        f"| Disk used | {disk_used} |",
        f"| Records with names | {named_count:,} ({named_count/total*100:.1f}%) |",
        f"| Records with labels | {labeled_count:,} ({labeled_count/total*100:.1f}%) |",
    ]
    if elapsed_h is not None:
        lines.append(f"| Extraction time | {elapsed_h:.1f}h |")
    lines += [
        f"",
        f"## Shard Files",
        f"",
    ]
    for name, n in shard_counts:
        lines.append(f"- `{name}`: {n:,} records")

    lines += [
        f"",
        f"## Environment Type Distribution",
        f"",
        f"| env_type | count | % |",
        f"|----------|-------|---|",
    ]
    for env, cnt in env_type_counts.most_common(20):
        pct = cnt / total * 100
        lines.append(f"| {env} | {cnt:,} | {pct:.1f}% |")

    lines += [
        f"",
        f"## Record Schema",
        f"",
        f"Each JSONL line has:",
        f"```json",
        f'{{ "doc_id": "sha256-of-dolma-doc", "paper_id": "same-as-doc_id",',
        f'   "env_type": "theorem|lemma|...", "name": "optional-title-or-null",',
        f'   "statement": "LaTeX body (max 3000 chars)", "label": "\\\\label{{...}} or null",',
        f'   "source": "emozilla/dolma-v1_7-arxiv" }}',
        f"```",
        f"",
        f"## Notes",
        f"",
        f"- Full LaTeX bodies (not abstracts or PDF-extracted text)",
        f"- Theorem environments extracted by regex, including `\\\\newtheorem` custom env declarations",
        f"- For local research/benchmarking only; NOT redistributed",
        f"- Statement bodies capped at 3000 characters",
        f"- `\\\\label{{}}` commands stripped from the `statement` field and stored in `label`",
    ]

    manifest_text = "\n".join(lines) + "\n"
    with open(MANIFEST_PATH, 'w') as f:
        f.write(manifest_text)

    print(f"\nManifest written to {MANIFEST_PATH}")
    print(f"Total: {total:,} theorems from {papers_processed:,} papers")


if __name__ == "__main__":
    main()
