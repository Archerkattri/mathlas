"""
Phase 3b: Clone stacks/stacks-project, parse LaTeX tag files → theorems.jsonl.

The Stacks Project uses a custom tag system: each result has a unique 4-char tag
(e.g. 04CF) plus a name. The LaTeX source is in *.tex files; tags map to
\label{...} inside theorem/lemma/proposition/... environments.

Output: reference/downloads/datasets/stacks/theorems.jsonl
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reference" / "downloads" / "datasets" / "stacks"
CLONE_DIR = OUT_DIR / "_repo"
OUT_FILE = OUT_DIR / "theorems.jsonl"
MANIFEST = OUT_DIR / "MANIFEST.md"

STACKS_URL = "https://github.com/stacks/stacks-project.git"

_ENVS = (
    "theorem", "lemma", "proposition", "corollary", "definition",
    "example", "remark", "situation",
)

_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_OPT_NAME_RE = re.compile(r"^\[([^\]]{0,120})\]")

# The Stacks Project's tags.py or tags/tags file maps tag -> label
_TAG_LINE_RE = re.compile(r"^([0-9A-Z]{4}),([^\s,]+)")


def _clone_stacks() -> bool:
    if CLONE_DIR.exists() and (CLONE_DIR / ".git").exists():
        print(f"  stacks-project already cloned at {CLONE_DIR}")
        return True
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("  Shallow-cloning stacks-project ...")
    ret = subprocess.run(
        ["git", "clone", "--depth=1", STACKS_URL, str(CLONE_DIR)],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  [ERROR] git clone failed:\n{ret.stderr}")
        return False
    print("  Clone done.")
    return True


def _load_tags(clone_dir: Path) -> dict[str, str]:
    """
    Load the stacks tags file: maps label -> 4-char tag.
    The tags/tags file has lines: TAG,label
    """
    tags_file = clone_dir / "tags" / "tags"
    label_to_tag: dict[str, str] = {}
    if not tags_file.exists():
        # Try alternate location
        tags_file = clone_dir / "tags"
        if tags_file.is_file():
            pass
        else:
            return label_to_tag
    try:
        for line in tags_file.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _TAG_LINE_RE.match(line.strip())
            if m:
                label_to_tag[m.group(2)] = m.group(1)
    except Exception as e:
        print(f"  [WARN] tags file parse error: {e}")
    return label_to_tag


def _extract_from_tex(path: Path, label_to_tag: dict) -> list[dict]:
    """Extract theorem-like environments from one .tex file."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    alt = "|".join(re.escape(e) for e in _ENVS)
    pat = re.compile(
        r"\\begin\{(" + alt + r")\*?\}(.*?)\\end\{\1\*?\}",
        re.DOTALL | re.IGNORECASE,
    )

    results = []
    for i, m in enumerate(pat.finditer(src)):
        env_type = m.group(1).lower().rstrip("*")
        body = m.group(2)
        opt_m = _OPT_NAME_RE.match(body.lstrip())
        opt_name = opt_m.group(1).strip() if opt_m else None
        lbl_m = _LABEL_RE.search(body)
        label = lbl_m.group(1).strip() if lbl_m else None
        tag = label_to_tag.get(label, None) if label else None

        doc_id = f"stacks::{path.stem}::env_{i:04d}"
        if tag:
            doc_id = f"stacks::{tag}"
        name = tag or label or f"{path.stem}-{i}"

        statement = body.strip()[:3000]
        results.append({
            "doc_id": doc_id,
            "name": name,
            "env_type": env_type,
            "statement": statement,
            "slogan": f"{env_type} {name}: {statement[:300]}",
            "label": label,
            "source": (
                f"https://stacks.math.columbia.edu/tag/{tag}"
                if tag else
                f"https://github.com/stacks/stacks-project/blob/master/{path.name}"
            ),
            "title": f"Stacks Project — {path.stem}",
            "category": "math.AG",  # Stacks is algebraic geometry
            "citations": None,
        })
    return results


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _clone_stacks():
        return

    label_to_tag = _load_tags(CLONE_DIR)
    print(f"  Loaded {len(label_to_tag)} tag→label mappings")

    tex_files = [f for f in CLONE_DIR.glob("*.tex") if f.name != "preamble.tex"]
    print(f"  Found {len(tex_files)} top-level .tex chapter files")

    n_total = 0
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for tf in sorted(tex_files):
            recs = _extract_from_tex(tf, label_to_tag)
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_total += len(recs)

    size_mb = OUT_FILE.stat().st_size / 1024 ** 2 if OUT_FILE.exists() else 0
    print(f"\n=== Stacks Phase 3b Complete ===")
    print(f"TeX files: {len(tex_files)}")
    print(f"Environments extracted: {n_total}")
    print(f"Output: {OUT_FILE}  ({size_mb:.1f} MB)")

    MANIFEST.write_text(f"""# MANIFEST — Stacks Project

| Field        | Value |
|--------------|-------|
| Source       | https://github.com/stacks/stacks-project |
| License      | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) |
| Clone depth  | 1 (shallow) |
| Date fetched | (see file mtime) |
| Records      | {n_total} theorem-like environments |
| Format       | JSONL — doc_id, name, env_type, statement, slogan, label, source, title, category |
| File         | theorems.jsonl |
| Size         | {size_mb:.1f} MB |

## Notes
- doc_id: `stacks::<TAG>` when Stacks tag known, else `stacks::<file>::env_NNNN`
- source: `https://stacks.math.columbia.edu/tag/<TAG>` when available
- category: math.AG (algebraic geometry / commutative algebra)
- label: LaTeX \\label{{}} value
""", encoding="utf-8")


if __name__ == "__main__":
    main()
