"""
Phase 1: Fetch LaTeX sources for the 61 test-target arXiv papers and extract
theorem-like environments.

Output:
  reference/downloads/datasets/test_targets/theorems.jsonl
  reference/downloads/datasets/test_targets/papers.jsonl

Polite: sleeps >= 3s between arXiv requests.
Uses OMP_NUM_THREADS=2, no GPU.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "reference" / "theorem-search-dataset"
OUT_DIR = REPO_ROOT / "reference" / "downloads" / "datasets" / "test_targets"
OUT_THEOREMS = OUT_DIR / "theorems.jsonl"
OUT_PAPERS = OUT_DIR / "papers.jsonl"

HEADERS = {
    "User-Agent": (
        "mathlas-benchmark/1.0 (research; github.com/Archerkattri/mathlas; "
        "contact: kattri@snu.ac.kr)"
    )
}

SLEEP_BETWEEN = 3.5   # seconds between arXiv requests (polite per their TOS)
LATEX_FETCH_TIMEOUT = 30   # seconds
API_FETCH_TIMEOUT = 15

# ---------------------------------------------------------------------------
# Theorem-environment regex
# A wide list of environments commonly used for results in math papers.
# ---------------------------------------------------------------------------
_KNOWN_ENVS = (
    "theorem", "lemma", "proposition", "corollary", "definition",
    "conjecture", "claim", "remark", "example", "fact", "observation",
    "axiom", "hypothesis", "criterion", "question", "problem",
)

# Regex to find \newtheorem{alias}{...} so we can recognise user-defined envs
_NEWTHEOREM_RE = re.compile(
    r"\\newtheorem\*?\s*\{([^}]+)\}", re.IGNORECASE
)

# Build the main environment pattern (lazy .+? so nested are handled)
def _env_pattern(envs: list[str]) -> re.Pattern:
    alt = "|".join(re.escape(e) for e in envs)
    return re.compile(
        r"\\begin\{(" + alt + r")\*?\}(.*?)\\end\{\1\*?\}",
        re.DOTALL | re.IGNORECASE,
    )


def _extract_theorem_envs(tex_source: str) -> list[dict]:
    """Return list of {env_type, name, statement} dicts from a .tex source."""
    # Discover any \newtheorem aliases
    aliases = list(_NEWTHEOREM_RE.findall(tex_source))
    envs = list(_KNOWN_ENVS) + [a.strip() for a in aliases if a.strip()]
    pat = _env_pattern(envs)

    # Also look for numbered labels: e.g. \begin{theorem}[name] or \label{...}
    label_in_optional = re.compile(r"^\[([^\]]{0,80})\]")
    label_cmd = re.compile(r"\\label\{([^}]+)\}")

    results = []
    for m in pat.finditer(tex_source):
        env_type = m.group(1).lower().rstrip("*")
        body = m.group(2)
        # Try to get optional name argument e.g. \begin{theorem}[Fermat's Last]
        opt_name_m = label_in_optional.match(body.lstrip())
        opt_name = opt_name_m.group(1).strip() if opt_name_m else None
        # Try to get \label{...} inside the body
        lbl_m = label_cmd.search(body)
        label = lbl_m.group(1).strip() if lbl_m else None
        # Clean body of \label command for the statement
        statement = body.strip()
        results.append({
            "env_type": env_type,
            "opt_name": opt_name,
            "label": label,
            "statement": statement,
        })
    return results


# ---------------------------------------------------------------------------
# arXiv fetch helpers
# ---------------------------------------------------------------------------

def _arxiv_eprintid(arxiv_id: str) -> str:
    """Normalise: strip leading 'math/' for old-style IDs."""
    return arxiv_id.strip()


def _fetch_latex_source(arxiv_id: str) -> Optional[str]:
    """
    Download https://arxiv.org/e-print/<id>, which is a gzipped tar of the
    LaTeX source. Returns the concatenated content of all .tex files found,
    or None on failure.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=LATEX_FETCH_TIMEOUT) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"    [WARN] e-print fetch failed for {arxiv_id}: {e}")
        return None

    # Try to decompress: may be .tar.gz or bare .gz (single file)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            tex_parts = []
            for member in tf.getmembers():
                if member.name.endswith(".tex"):
                    try:
                        f = tf.extractfile(member)
                        if f:
                            content = f.read().decode("utf-8", errors="replace")
                            tex_parts.append(content)
                    except Exception:
                        pass
            if tex_parts:
                return "\n".join(tex_parts)
    except tarfile.TarError:
        pass
    # Maybe it's a bare gzip (single .tex file)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            return gz.read().decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def _fetch_arxiv_abstract(arxiv_id: str) -> Optional[dict]:
    """
    Fall back to the Atom API to get title/abstract/categories.
    Returns {title, abstract, categories} or None.
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=API_FETCH_TIMEOUT) as resp:
            atom = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    [WARN] API fetch failed for {arxiv_id}: {e}")
        return None
    title_m = re.search(r"<title>(?!ArXiv Query)(.+?)</title>", atom, re.DOTALL)
    summary_m = re.search(r"<summary>(.+?)</summary>", atom, re.DOTALL)
    cats = re.findall(r'term="([^"]+)"', atom)
    return {
        "title": title_m.group(1).strip() if title_m else None,
        "abstract": summary_m.group(1).strip() if summary_m else None,
        "categories": cats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test parquet
    df = pd.read_parquet(DATASET_DIR / "theorems-test.parquet")
    print(f"Loaded {len(df)} test queries from theorems-test.parquet")

    # Extract unique arXiv IDs
    links = df["link to paper on arxiv"].unique()
    arxiv_ids: list[str] = []
    for link in links:
        m = re.search(r"(?:abs|pdf)/([0-9a-z./]+)", str(link))
        if m:
            arxiv_ids.append(m.group(1))
        else:
            print(f"[WARN] Could not parse arXiv ID from: {link}")
    arxiv_ids = list(dict.fromkeys(arxiv_ids))  # deduplicate, preserve order
    print(f"Unique arXiv IDs to fetch: {len(arxiv_ids)}")

    paper_fh = open(OUT_PAPERS, "w", encoding="utf-8")
    theorem_fh = open(OUT_THEOREMS, "w", encoding="utf-8")

    recovered_papers = 0
    total_theorems = 0

    for i, arxiv_id in enumerate(arxiv_ids):
        print(f"\n[{i+1}/{len(arxiv_ids)}] {arxiv_id}")
        time.sleep(SLEEP_BETWEEN)

        # --- Try to fetch LaTeX source ---
        tex = _fetch_latex_source(arxiv_id)

        # --- Always fetch abstract (for paper metadata) ---
        meta = _fetch_arxiv_abstract(arxiv_id)
        time.sleep(SLEEP_BETWEEN)

        title = meta.get("title") if meta else None
        abstract = meta.get("abstract") if meta else None
        categories = meta.get("categories", []) if meta else []

        # Write paper record
        paper_rec = {
            "paper_id": arxiv_id,
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "categories": categories,
            "source_link": f"https://arxiv.org/abs/{arxiv_id}",
            "has_latex": tex is not None,
        }
        paper_fh.write(json.dumps(paper_rec, ensure_ascii=False) + "\n")
        paper_fh.flush()

        if tex is None:
            print(f"  [SKIP] No LaTeX source; paper metadata saved.")
            continue

        # Extract theorems
        envs = _extract_theorem_envs(tex)
        print(f"  [OK] Extracted {len(envs)} theorem-like environments")
        recovered_papers += 1
        total_theorems += len(envs)

        for j, env in enumerate(envs):
            rec = {
                "doc_id": f"{arxiv_id}::env_{j:04d}",
                "paper_id": arxiv_id,
                "arxiv_id": arxiv_id,
                "name": env["opt_name"],
                "statement": env["statement"],
                "slogan": env["statement"],   # LaTeX body as statement (no NL slogan)
                "label": env["label"],
                "env_type": env["env_type"],
                "title": title,
                "source": f"https://arxiv.org/abs/{arxiv_id}",
                "category": categories[0] if categories else None,
                "citations": None,
            }
            theorem_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        theorem_fh.flush()

    paper_fh.close()
    theorem_fh.close()

    print(f"\n=== Phase 1 Complete ===")
    print(f"Papers attempted:  {len(arxiv_ids)}")
    print(f"LaTeX recovered:   {recovered_papers}")
    print(f"Total environments:{total_theorems}")
    print(f"Theorems output:   {OUT_THEOREMS}")
    print(f"Papers output:     {OUT_PAPERS}")


if __name__ == "__main__":
    main()
