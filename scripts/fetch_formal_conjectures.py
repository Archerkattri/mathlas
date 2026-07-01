#!/usr/bin/env python3
"""
Ingest google-deepmind/formal-conjectures — a Lean-4 corpus of formalized
conjecture STATEMENTS (arXiv:2605.13171) — into mathlas index documents.

The repo (Apache-2.0, actively maintained) formalizes open/solved mathematical
conjectures as Lean theorems with a `sorry` proof placeholder, each tagged with
a `@[category research solved|research open|...]` attribute and a `/-- ... -/`
docstring that states the conjecture in prose. That docstring is exactly a
ready-made SLOGAN (the NL denotation mathlas embeds), and the Lean theorem is the
STATEMENT — so this is a free, high-quality, already-structured corpus.

Output (mirrors scripts/fetch_mathlib.py, so it folds into the SAME index build
pipeline; every doc carries `source_tag="formal_conjectures"`, and its `doc_id`
uses the `fc::` prefix that mathlas.retrieve.corpus.source_key maps to the
`formal_conjectures` source key for source_filter / source_weights):

  reference/downloads/datasets/formal_conjectures/statements.jsonl
      one JSON doc per Lean declaration: doc_id / name / statement / slogan /
      source (github blob URL) / source_tag / title / category / label / citations
  reference/downloads/datasets/formal_conjectures/benchmarks.json
      the two FROZEN eval subsets, FC100SolvedSet1 and FC100OpenSet1 (100
      statements each, defined by import-lists in FormalConjectures/Subsets/),
      as {set_name: [member module paths]} plus the doc_ids that fall in each set
  reference/downloads/datasets/formal_conjectures/MANIFEST.md

License: Apache-2.0 (formal-conjectures). We read the repo as RAW DATA only.

Manual command (run this yourself if the automatic clone is blocked here):
  git clone --depth=1 https://github.com/google-deepmind/formal-conjectures \\
      reference/downloads/datasets/formal_conjectures/_repo
  python scripts/fetch_formal_conjectures.py            # re-uses an existing clone

This script NEVER fabricates statements: if the clone fails and no local checkout
is present it prints the manual command and exits non-zero (no partial/fake data).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reference" / "downloads" / "datasets" / "formal_conjectures"
CLONE_DIR = OUT_DIR / "_repo"
OUT_FILE = OUT_DIR / "statements.jsonl"
BENCH_FILE = OUT_DIR / "benchmarks.json"
MANIFEST = OUT_DIR / "MANIFEST.md"

FC_URL = "https://github.com/google-deepmind/formal-conjectures.git"
BLOB_BASE = "https://github.com/google-deepmind/formal-conjectures/blob/main"

# Directories that are library plumbing / aggregators, not conjecture statements.
_SKIP_DIRS = {"Util"}
_SUBSETS_DIR = "Subsets"  # the frozen FC100*Set*.lean eval lists live here

# A Lean declaration we treat as a statement doc. Names can carry dots, primes,
# and «guillemet» segments (e.g. erdos_350.variants.strengthening).
_DECL_RE = re.compile(
    r"^(?P<kw>theorem|lemma|def|abbrev|noncomputable def|"
    r"protected theorem|protected lemma)\s+"
    r"(?P<name>[A-Za-z_«][A-Za-z0-9_.'«»]*)",
)
# `@[category research solved, AMS 5 11]` -> ("research solved", "5 11")
_CATEGORY_RE = re.compile(r"@\[[^]]*category\s+([a-zA-Z ]+?)\s*(?:,|\])")
_IMPORT_RE = re.compile(r"^\s*import\s+(FormalConjectures[\w.«»]+)")


def _clone() -> bool:
    """Shallow-clone the repo, re-using an existing checkout. Returns False (with
    a printed manual command) if no clone can be obtained — never fabricates."""
    if (CLONE_DIR / ".git").exists() or (CLONE_DIR / "FormalConjectures").exists():
        print(f"  formal-conjectures already present at {CLONE_DIR}")
        return True
    print("  Shallow-cloning formal-conjectures (depth=1) ...")
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    ret = subprocess.run(
        ["git", "clone", "--depth=1", FC_URL, str(CLONE_DIR)],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  [ERROR] git clone failed:\n{ret.stderr.strip()}")
        print("\n  Network clone blocked. Run this manually, then re-run me:\n"
              f"    git clone --depth=1 {FC_URL} \\\n        {CLONE_DIR}")
        return False
    print("  Clone done.")
    return True


def _module_to_relpath(module: str) -> str:
    """`FormalConjectures.ErdosProblems.«350»` ->
    `FormalConjectures/ErdosProblems/«350».lean` (the dots between «...» segments
    are path separators; dots INSIDE guillemets, e.g. «1308.0994», are not)."""
    parts, buf, depth = [], [], 0
    for ch in module:
        if ch == "«":
            depth += 1
            buf.append(ch)
        elif ch == "»":
            depth -= 1
            buf.append(ch)
        elif ch == "." and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return "/".join(parts) + ".lean"


def _preamble(lines: list[str], decl_idx: int) -> str:
    """The contiguous docstring + `@[...]` attribute block directly above a
    declaration. Attributes span multiple lines (a multi-line `@[category ...,
    formal_proof ... ]`) and the `/-- ... -/` docstring may contain blank lines,
    so we walk upward collecting non-blank lines and only stop at a blank line
    that is OUTSIDE a docstring."""
    pre: list[str] = []
    k = decl_idx - 1
    in_doc = False  # True while (scanning up) we are between a `-/` and its `/--`
    while k >= 0 and decl_idx - k <= 60:
        s = lines[k]
        if not s.strip() and not in_doc:
            break
        pre.append(s)
        if "-/" in s:
            in_doc = True
        if "/--" in s or "/-!" in s:
            in_doc = False
        k -= 1
    pre.reverse()
    return "\n".join(pre)


def _docstring_text(preamble: str) -> str | None:
    """Extract the `/-- ... -/` prose from a preamble block, collapsed to one line."""
    m = re.search(r"/--(.*?)-/", preamble, re.DOTALL)
    if not m:
        return None
    text = re.sub(r"\s+", " ", m.group(1)).strip()
    return text or None


def _extract_file(path: Path, rel: str) -> list[dict]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = src.split("\n")
    n = len(lines)
    out: list[dict] = []
    i = 0
    while i < n:
        m = _DECL_RE.match(lines[i])
        if not m:
            i += 1
            continue
        kw, name = m.group("kw"), m.group("name")
        # gather the declaration block: to the next blank line / next decl,
        # capped so a runaway file can't produce a giant record.
        decl_lines = [lines[i]]
        j = i + 1
        while j < n and lines[j].strip() and not _DECL_RE.match(lines[j]):
            decl_lines.append(lines[j])
            j += 1
            if j - i > 60:
                break
        statement = "\n".join(decl_lines).strip()
        preamble = _preamble(lines, i)
        cm = _CATEGORY_RE.search(preamble)
        category = cm.group(1).strip() if cm else None
        doc = _docstring_text(preamble)
        # type signature: between the first top-level `:` and `:=`/`by`.
        sig_m = re.search(r":\s*(.+?)(?:\s*:=|\s*\bby\b|$)", statement, re.DOTALL)
        type_sig = re.sub(r"\s+", " ", sig_m.group(1)).strip() if sig_m else ""
        slogan = doc or f"{kw} {name}: {type_sig[:500]}"
        out.append({
            "doc_id": f"fc::{rel}::{name}",
            "name": name,
            "env_type": kw,
            "statement": statement[:2000],
            "slogan": slogan[:1000],
            "source": f"{BLOB_BASE}/{rel}",
            "source_tag": "formal_conjectures",
            "title": rel,
            "label": None,
            "category": category,  # e.g. "research solved" / "research open"
            "citations": None,
        })
        i = j
    return out


def _parse_subset(path: Path) -> list[str]:
    """The frozen FC100*Set*.lean files ARE their membership: a list of
    `import FormalConjectures...` module lines. Return those module paths."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    mods = []
    for line in src.split("\n"):
        m = _IMPORT_RE.match(line)
        if m and "Util." not in m.group(1):
            mods.append(m.group(1))
    return mods


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _clone():
        return 1

    fc_root = CLONE_DIR / "FormalConjectures"
    if not fc_root.exists():
        print(f"  [ERROR] {fc_root} missing — clone looks incomplete.")
        return 1

    lean_files = sorted(fc_root.rglob("*.lean"))
    docs: list[dict] = []
    # map each source file's relpath -> the doc_ids extracted from it (for the
    # eval-subset membership below).
    by_rel: dict[str, list[str]] = {}
    for lf in lean_files:
        rel = lf.relative_to(CLONE_DIR).as_posix()
        top = lf.relative_to(fc_root).parts[0] if lf != fc_root else ""
        if top in _SKIP_DIRS or top == _SUBSETS_DIR:
            continue  # library utils + the aggregator subset files aren't docs
        recs = _extract_file(lf, rel)
        for r in recs:
            docs.append(r)
            by_rel.setdefault(rel, []).append(r["doc_id"])

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for r in docs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # frozen eval subsets: FC100SolvedSet1 / FC100OpenSet1 (import-lists).
    benchmarks: dict[str, dict] = {}
    subsets_dir = fc_root / _SUBSETS_DIR
    for set_file in sorted(subsets_dir.glob("FC100*Set*.lean")) if subsets_dir.exists() else []:
        set_name = set_file.stem
        modules = _parse_subset(set_file)
        # resolve each imported module to the source relpath, then to its doc_ids.
        member_rels, member_doc_ids = [], []
        for mod in modules:
            rel = _module_to_relpath(mod)  # e.g. FormalConjectures/ErdosProblems/«350».lean
            member_rels.append(rel)
            member_doc_ids.extend(by_rel.get(rel, []))
        benchmarks[set_name] = {
            "source": f"{BLOB_BASE}/FormalConjectures/{_SUBSETS_DIR}/{set_file.name}",
            "n_modules": len(modules),
            "member_modules": member_rels,
            "member_doc_ids": member_doc_ids,
        }
    with open(BENCH_FILE, "w", encoding="utf-8") as fh:
        json.dump(benchmarks, fh, ensure_ascii=False, indent=2)

    size_mb = OUT_FILE.stat().st_size / 1024 ** 2 if OUT_FILE.exists() else 0
    n_solved = sum(1 for d in docs if (d["category"] or "").startswith("research solved"))
    n_open = sum(1 for d in docs if (d["category"] or "").startswith("research open"))
    print("\n=== formal-conjectures ingestion complete ===")
    print(f"Lean files scanned : {len(lean_files)}")
    print(f"Statements extracted: {len(docs)}  "
          f"(research solved={n_solved}, research open={n_open})")
    for name, b in benchmarks.items():
        print(f"  benchmark {name}: {b['n_modules']} modules, "
              f"{len(b['member_doc_ids'])} statements")
    print(f"Output: {OUT_FILE}  ({size_mb:.1f} MB)")
    print(f"Benchmarks: {BENCH_FILE}")

    MANIFEST.write_text(f"""# MANIFEST — formal-conjectures

| Field        | Value |
|--------------|-------|
| Source       | https://github.com/google-deepmind/formal-conjectures |
| Paper        | arXiv:2605.13171 |
| License      | Apache-2.0 |
| Clone depth  | 1 (shallow) |
| Date fetched | (see file mtime) |
| Records      | {len(docs)} Lean conjecture statements |
| Format       | JSONL — doc_id, name, env_type, statement, slogan, source, source_tag, title, category |
| Files        | statements.jsonl, benchmarks.json |
| Size         | {size_mb:.1f} MB |

## Document field mapping (to corpus.py Document + mathlas index)
- doc_id: `fc::<file_path>::<decl_name>`  (the `fc::` prefix -> source_key `formal_conjectures`)
- name: Lean declaration name
- statement: raw Lean declaration text (capped 2000 chars)
- slogan: the `/-- ... -/` docstring prose (the conjecture in NL) when present,
  else a `theorem NAME: TYPE` summary — this is the dense-embedded text
- source: GitHub blob URL; source_tag: `formal_conjectures`
- category: the `@[category ...]` label (e.g. `research solved`, `research open`)
- title: relative file path; label / citations: None

## Frozen eval subsets (benchmarks.json)
- **FC100SolvedSet1** / **FC100OpenSet1** — 100 statements each, defined by the
  import-lists in `FormalConjectures/Subsets/`. Compiled across all supported
  Lean versions, so the sets stay fixed and comparable as the repo evolves.
- Each entry lists the member modules and the doc_ids ingested from them, so an
  eval can restrict retrieval/verification to a frozen set.
""", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
