"""
Fix stacks/theorems.jsonl: the original fetch_stacks.py had the tag→label map
reversed. The stacks tags/tags file format is: `XXXX,label` (tag FIRST, label second)
but it was being used as label→tag. Additionally the labels in the LaTeX files
don't include the chapter prefix, so they match directly.

This script:
1. Rebuilds the correct label→tag map (by reversing: split by comma, [0]=tag, [1]=label)
2. For each theorem record, looks up label in the map
3. Updates source URL and doc_id for matched records
4. Rewrites the file in-place.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reference" / "downloads" / "datasets" / "stacks"
CLONE_DIR = OUT_DIR / "_repo"
OUT_FILE = OUT_DIR / "theorems.jsonl"
TAGS_FILE = CLONE_DIR / "tags" / "tags"


def _load_label_to_tag(tags_file: Path) -> dict[str, str]:
    """Parse the stacks tags/tags file: each non-comment line is `XXXX,label`.
    Returns label -> tag dict."""
    label_to_tag: dict[str, str] = {}
    if not tags_file.exists():
        print(f"  [WARN] tags file not found: {tags_file}")
        return label_to_tag
    for line in tags_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            tag, label = parts[0].strip(), parts[1].strip()
            label_to_tag[label] = tag
    return label_to_tag


def main() -> None:
    label_to_tag = _load_label_to_tag(TAGS_FILE)
    print(f"Loaded {len(label_to_tag)} label→tag mappings")

    records = []
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Records to process: {len(records)}")

    # The stacks tags file uses `chapter-label` format (e.g. `algebra-example-local-regular`)
    # but the LaTeX \label commands use just `label` (e.g. `example-local-regular`).
    # We can reconstruct `chapter-label` from the doc_id which is `stacks::<chapter>::env_N`.
    # So: doc_id = stacks::adequate::env_0000, label = definition-module-valued-functor
    # -> full_label = adequate-definition-module-valued-functor (chapter + "-" + label)
    updated = 0
    for rec in records:
        label = rec.get("label")
        if not label:
            continue
        # Extract chapter name from current doc_id: stacks::<chapter>::env_N
        doc_parts = rec["doc_id"].split("::")
        chapter = doc_parts[1] if len(doc_parts) >= 2 else ""
        # Try chapter-label combination
        full_label = f"{chapter}-{label}" if chapter and not chapter.startswith("env_") else label
        tag = label_to_tag.get(full_label) or label_to_tag.get(label)
        if tag:
            rec["doc_id"] = f"stacks::{tag}"
            rec["name"] = tag
            rec["source"] = f"https://stacks.math.columbia.edu/tag/{tag}"
            updated += 1

    print(f"Records updated with tag ID: {updated}/{len(records)}")

    # Rewrite
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Rewrote {OUT_FILE}")

    # Update MANIFEST
    manifest = OUT_DIR / "MANIFEST.md"
    text = manifest.read_text(encoding="utf-8")
    text = text.replace("Records      | 16467", f"Records      | {len(records)}")
    text = text.replace("tag → label when known", f"{updated}/{len(records)} records have stacks.math.columbia.edu tag URLs")
    manifest.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
