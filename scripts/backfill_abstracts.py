"""
Backfill missing abstracts for test_targets/papers.jsonl entries where
abstract is None (API timeout during Phase 1 run).

Polite: 5s between requests.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_FILE = REPO_ROOT / "reference" / "downloads" / "datasets" / "test_targets" / "papers.jsonl"
HEADERS = {
    "User-Agent": (
        "mathlas-benchmark/1.0 (research; github.com/Archerkattri/mathlas; "
        "contact: krishiattriwork@gmail.com)"
    )
}
SLEEP = 5.0


def _fetch_abstract(arxiv_id: str) -> dict | None:
    url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            atom = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] {arxiv_id}: {e}")
        return None
    title_m = re.search(r"<title>(?!ArXiv Query)(.+?)</title>", atom, re.DOTALL)
    summary_m = re.search(r"<summary>(.+?)</summary>", atom, re.DOTALL)
    cats = re.findall(r'term="([^"]+)"', atom)
    return {
        "title": title_m.group(1).strip() if title_m else None,
        "abstract": summary_m.group(1).strip() if summary_m else None,
        "categories": cats,
    }


def main() -> None:
    records = []
    with open(PAPERS_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    missing = [r for r in records if r.get("abstract") is None and r.get("has_latex", False)]
    print(f"Records needing abstract backfill: {len(missing)}")

    updated = {r["paper_id"]: r for r in records}
    for i, rec in enumerate(missing):
        arxiv_id = rec["paper_id"]
        print(f"  [{i+1}/{len(missing)}] {arxiv_id}")
        time.sleep(SLEEP)
        meta = _fetch_abstract(arxiv_id)
        if meta:
            updated[arxiv_id]["title"] = meta.get("title") or updated[arxiv_id].get("title")
            updated[arxiv_id]["abstract"] = meta.get("abstract")
            updated[arxiv_id]["categories"] = meta.get("categories", [])
            print(f"    -> title: {(meta.get('title') or '')[:60]}")

    # Rewrite
    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(updated.get(rec["paper_id"], rec), ensure_ascii=False) + "\n")
    print(f"Backfill complete. Rewrote {PAPERS_FILE}")


if __name__ == "__main__":
    main()
