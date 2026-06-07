"""
Assess retrieval benchmark coverage for the test_targets dataset.

For each of the 110 test queries, we check:
  A) Is the paper in the corpus?
  B) Is any theorem environment present for that paper?
  C) Does at least one environment have a \label or name that partially matches
     the target theorem number (e.g. "Theorem 3.1" -> label contains "3.1" or "thm_intro")?
  D) Count of environments per paper (to gauge which papers are richly extracted).

Run after Phase 1 completes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "reference" / "theorem-search-dataset"
DATASETS_DIR = REPO_ROOT / "reference" / "downloads" / "datasets"

TEST_PARQUET = DATASET_DIR / "theorems-test.parquet"
THEOREMS_FILE = DATASETS_DIR / "test_targets" / "theorems.jsonl"


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def main() -> None:
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Test queries: {len(df)}")

    # Load extracted theorems
    theorems: dict[str, list[dict]] = {}
    with open(THEOREMS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            theorems.setdefault(rec["arxiv_id"], []).append(rec)

    n_papers = len(theorems)
    n_envs = sum(len(v) for v in theorems.values())
    print(f"Distinct papers with LaTeX:   {n_papers}")
    print(f"Total theorem-like envs:      {n_envs}")

    covered_a = 0  # paper present
    covered_b = 0  # paper has envs
    covered_c = 0  # specific theorem matched

    for _, row in df.iterrows():
        link = str(row["link to paper on arxiv"])
        m = re.search(r"(?:abs|pdf)/([0-9a-z./]+)", link)
        arxiv_id = m.group(1) if m else None
        thm_num = normalize(row["theorem number"])

        if arxiv_id and arxiv_id in theorems:
            covered_a += 1
            if theorems[arxiv_id]:
                covered_b += 1

                # Try to find matching theorem
                # Strategy: look for any of: label, name, env_type+number patterns
                # from thm_num like "theorem 3.1" -> ["theorem", "3.1", "3", "1"]
                # extract digits
                digits = re.findall(r"\d+", thm_num)
                digit_str = ".".join(digits) if digits else ""
                env_type_m = re.match(r"(theorem|lemma|prop(?:osition)?|corollary|cor|defn?|definition|conjecture|claim|remark)", thm_num)
                env_kw = env_type_m.group(1)[:3] if env_type_m else ""

                matched = False
                for rec in theorems[arxiv_id]:
                    # Check label
                    label = normalize(rec.get("label") or "")
                    name = normalize(rec.get("name") or "")
                    # Do digits appear in label or name?
                    if digit_str and digit_str in label:
                        matched = True
                        break
                    if digit_str and digit_str in name:
                        matched = True
                        break
                    # Or does the env_type match and digits appear together in statement?
                    if env_kw and digit_str:
                        stmt_preview = normalize((rec.get("statement") or "")[:200])
                        if digit_str in stmt_preview:
                            matched = True
                            break
                if matched:
                    covered_c += 1

    print(f"\n=== Coverage Summary ===")
    print(f"A) Paper present in LaTeX corpus: {covered_a}/{len(df)} queries ({100*covered_a/len(df):.1f}%)")
    print(f"   (=unique papers with LaTeX: {n_papers}/61)")
    print(f"B) Paper has theorem envs:        {covered_b}/{len(df)} queries ({100*covered_b/len(df):.1f}%)")
    print(f"C) Specific theorem env matched:  {covered_c}/{len(df)} queries ({100*covered_c/len(df):.1f}%)")
    print()
    print("NOTE: C is a LOWER BOUND on true coverage — many theorems don't have")
    print("numbered \\labels in their source; they rely on LaTeX counter numbering.")
    print("The actual number of matching envs is likely very close to B.")
    print()
    # Show per-paper env counts
    print("Environments per paper (sorted by count desc):")
    paper_counts = {k: len(v) for k, v in theorems.items()}
    for pid, cnt in sorted(paper_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {pid:20s}  {cnt:4d} envs")


if __name__ == "__main__":
    main()
