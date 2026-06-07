"""
Phase 3a: Shallow-clone leanprover-community/mathlib4 and extract theorem/lemma
declaration names + statement signatures.

Output: reference/downloads/datasets/mathlib/theorems.jsonl
License: Apache-2.0 (mathlib4)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reference" / "downloads" / "datasets" / "mathlib"
CLONE_DIR = REPO_ROOT / "reference" / "downloads" / "datasets" / "mathlib" / "_repo"
OUT_FILE = OUT_DIR / "theorems.jsonl"
MANIFEST = OUT_DIR / "MANIFEST.md"

MATHLIB_URL = "https://github.com/leanprover-community/mathlib4.git"

# Pattern to match Lean 4 theorem/lemma/def declarations.
# We want: theorem|lemma <name> (<params>) : <type> :=
# We capture up to the first `:=` or `where` after the colon type.
_DECL_RE = re.compile(
    r"^(?P<kw>theorem|lemma|noncomputable def|protected theorem|protected lemma|"
    r"private theorem|private lemma)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.']*)"
    r"(?P<rest>[^:=\n]*?)\s*:\s*"
    r"(?P<type_sig>[^:=\n](?:[^:=]|:(?!=))*?)(?=\s*:=|\s*where|\s*by\b|\n\n)",
    re.MULTILINE,
)

# Alternative: line-by-line grab of `theorem NAME ... : TYPE` blocks
_SIMPLE_RE = re.compile(
    r"^(theorem|lemma)\s+([\w.']+)",
    re.MULTILINE,
)


def _clone_mathlib() -> bool:
    if CLONE_DIR.exists() and (CLONE_DIR / ".git").exists():
        print(f"  mathlib4 already cloned at {CLONE_DIR}")
        return True
    print(f"  Shallow-cloning mathlib4 (depth=1) ...")
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    ret = subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none",
         "--no-checkout", MATHLIB_URL, str(CLONE_DIR)],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        # Try without blob:none filter
        ret = subprocess.run(
            ["git", "clone", "--depth=1", MATHLIB_URL, str(CLONE_DIR)],
            capture_output=True, text=True,
        )
    if ret.returncode != 0:
        print(f"  [ERROR] git clone failed:\n{ret.stderr}")
        return False
    # Sparse checkout of .lean files only
    subprocess.run(["git", "-C", str(CLONE_DIR), "checkout"], capture_output=True)
    print("  Clone done.")
    return True


def _extract_from_file(path: Path, rel: str) -> list[dict]:
    """Extract theorem-like declarations from one .lean file."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    results = []
    lines = src.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _SIMPLE_RE.match(line)
        if m:
            kw = m.group(1)
            name = m.group(2)
            # Gather the full declaration: up to the next blank line or next theorem
            decl_lines = [line]
            j = i + 1
            while j < n and lines[j].strip() and not _SIMPLE_RE.match(lines[j]):
                decl_lines.append(lines[j])
                j += 1
                if j - i > 40:  # cap at 40 lines per decl
                    break
            statement = "\n".join(decl_lines).strip()
            # Extract the type signature: text between first `:` and `:=`/`by`/`where`
            sig_m = re.search(r":\s*(.+?)(?:\s*:=|\s*\bby\b|\s*\bwhere\b|$)",
                              statement, re.DOTALL)
            type_sig = sig_m.group(1).strip() if sig_m else ""

            results.append({
                "doc_id": f"mathlib4::{rel}::{name}",
                "name": name,
                "env_type": kw,
                "statement": statement[:2000],   # cap at 2000 chars
                "slogan": f"{kw} {name}: {type_sig[:500]}",
                "source": f"https://github.com/leanprover-community/mathlib4/blob/master/{rel}",
                "title": rel,
                "label": None,
                "category": "math",
                "citations": None,
            })
            i = j
        else:
            i += 1
    return results


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not _clone_mathlib():
        # Try sparse approach: ls-tree to find .lean files + cat them
        print("  [FALLBACK] Clone failed. Attempting ls-tree approach ...")
        sys.exit(1)

    # Walk all .lean files
    lean_files = list(CLONE_DIR.rglob("Mathlib/**/*.lean"))
    if not lean_files:
        lean_files = list(CLONE_DIR.rglob("*.lean"))
    print(f"  Found {len(lean_files)} .lean files")

    n_total = 0
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for lf in lean_files:
            try:
                rel = lf.relative_to(CLONE_DIR).as_posix()
            except ValueError:
                rel = lf.name
            recs = _extract_from_file(lf, rel)
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_total += len(recs)

    size_mb = OUT_FILE.stat().st_size / 1024 ** 2 if OUT_FILE.exists() else 0
    print(f"\n=== mathlib4 Phase 3a Complete ===")
    print(f"Lean files processed: {len(lean_files)}")
    print(f"Declarations extracted: {n_total}")
    print(f"Output: {OUT_FILE}  ({size_mb:.1f} MB)")

    MANIFEST.write_text(f"""# MANIFEST — mathlib4

| Field        | Value |
|--------------|-------|
| Source       | https://github.com/leanprover-community/mathlib4 |
| License      | Apache-2.0 |
| Clone depth  | 1 (shallow) |
| Date fetched | (see file mtime) |
| Records      | {n_total} theorem/lemma declarations |
| Format       | JSONL — doc_id, name, env_type, statement, slogan, source, title, category |
| File         | theorems.jsonl |
| Size         | {size_mb:.1f} MB |

## Document field mapping (to corpus.py Document)
- doc_id: `mathlib4::<file_path>::<name>`
- name: Lean declaration name
- statement: raw Lean declaration text (capped 2000 chars)
- slogan: `theorem NAME: TYPE` summary
- source: GitHub blob URL
- title: relative file path
- category: "math"
- citations: None
- label: None
""", encoding="utf-8")


if __name__ == "__main__":
    main()
