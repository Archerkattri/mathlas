"""
Phase 3c: Fetch ProofWiki theorem pages via the MediaWiki API.

Strategy: Use Special:AllPages with a continuation token to enumerate all pages
in the main namespace, then batch-fetch content via the parse API.
Cap at 10 000 pages (ProofWiki has ~23k theorem pages; 10k is a reasonable sample).

ProofWiki ToS: CC-BY-SA 3.0 (https://proofwiki.org/wiki/ProofWiki:Copyrights).
Rate limit: ≤ 1 request/sec to be polite.

Output: reference/downloads/datasets/proofwiki/theorems.jsonl
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reference" / "downloads" / "datasets" / "proofwiki"
OUT_FILE = OUT_DIR / "theorems.jsonl"
MANIFEST = OUT_DIR / "MANIFEST.md"

BASE_URL = "https://proofwiki.org/w/api.php"
HEADERS = {
    "User-Agent": (
        "mathlas-benchmark/1.0 (research; github.com/Archerkattri/mathlas; "
        "contact: krishiattriwork@gmail.com)"
    )
}

MAX_PAGES = 10_000
SLEEP_ALLPAGES = 1.0   # between AllPages calls
SLEEP_FETCH = 0.8      # between content fetches
API_TIMEOUT = 30

# Patterns in wikitext to identify theorem-like content
_THM_KEYWORDS = re.compile(
    r"\b(theorem|lemma|proposition|corollary|conjecture|definition)\b",
    re.IGNORECASE,
)
_NAMED_THM_RE = re.compile(
    r"==\s*(Theorem|Lemma|Statement|Proof)\s*==",
    re.IGNORECASE,
)


def _api_get(params: dict, retries: int = 3) -> dict:
    url = BASE_URL + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  [Warn] API error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return {}


def _enum_all_pages() -> list[dict]:
    """Use AllPages to enumerate all page titles, return list of {pageid, title}."""
    pages = []
    params = {
        "action": "query",
        "list": "allpages",
        "aplimit": "500",
        "apnamespace": "0",   # main namespace
    }
    while len(pages) < MAX_PAGES:
        data = _api_get(params)
        batch = data.get("query", {}).get("allpages", [])
        pages.extend(batch)
        cont = data.get("continue", {}).get("apcontinue")
        if not cont or len(pages) >= MAX_PAGES:
            break
        params["apfrom"] = cont
        time.sleep(SLEEP_ALLPAGES)
    return pages[:MAX_PAGES]


def _fetch_page_wikitext(page_id: int) -> str | None:
    """Fetch raw wikitext for a page by page ID."""
    data = _api_get({
        "action": "query",
        "pageids": str(page_id),
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
    })
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        revs = page.get("revisions", [])
        if revs:
            slots = revs[0].get("slots", {})
            return slots.get("main", {}).get("*", None)
    return None


def _extract_theorem(title: str, wikitext: str) -> dict | None:
    """
    Try to extract the main theorem statement from a ProofWiki wikitext.
    Returns a dict or None if not theorem-like.
    """
    # Quick filter: must contain theorem-like heading
    if not (_THM_KEYWORDS.search(title) or _NAMED_THM_RE.search(wikitext)):
        return None

    # Extract the Statement section
    stmt_m = re.search(
        r"==\s*(?:Theorem|Statement|Definition)\s*==\s*(.*?)(?:==|$)",
        wikitext, re.DOTALL | re.IGNORECASE,
    )
    if stmt_m:
        statement = stmt_m.group(1).strip()[:3000]
    else:
        # Fall back: use first 800 chars of wikitext as statement
        statement = wikitext.strip()[:800]

    # Extract categories
    cats = re.findall(r"\[\[Category:([^\]]+)\]\]", wikitext)

    return {
        "doc_id": f"proofwiki::{urllib.parse.quote(title)}",
        "name": title,
        "env_type": "theorem",  # rough label
        "statement": statement,
        "slogan": f"ProofWiki: {title}",
        "label": None,
        "source": f"https://proofwiki.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
        "title": title,
        "category": cats[0] if cats else None,
        "citations": None,
    }


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for resume
    n_already = 0
    seen_titles: set[str] = set()
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_titles.add(rec.get("name", ""))
                    n_already += 1
                except Exception:
                    pass
        print(f"Resuming: {n_already} pages already in {OUT_FILE}")

    print("Enumerating ProofWiki pages ...")
    all_pages = _enum_all_pages()
    print(f"Found {len(all_pages)} page titles")

    n_written = n_already
    n_skipped = 0
    mode = "a" if n_already > 0 else "w"

    with open(OUT_FILE, mode, encoding="utf-8", buffering=1) as fh:
        for i, page in enumerate(all_pages):
            title = page.get("title", "")
            if title in seen_titles:
                n_skipped += 1
                continue
            if n_written >= MAX_PAGES:
                break

            wikitext = _fetch_page_wikitext(page["pageid"])
            time.sleep(SLEEP_FETCH)

            if not wikitext:
                continue

            rec = _extract_theorem(title, wikitext)
            if rec:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(all_pages)}] written={n_written} skipped={n_skipped}")

    size_mb = OUT_FILE.stat().st_size / 1024 ** 2 if OUT_FILE.exists() else 0
    print(f"\n=== ProofWiki Phase 3c Complete ===")
    print(f"Pages inspected: {len(all_pages)}")
    print(f"Records written: {n_written}")
    print(f"Output: {OUT_FILE}  ({size_mb:.1f} MB)")

    MANIFEST.write_text(f"""# MANIFEST — ProofWiki

| Field        | Value |
|--------------|-------|
| Source       | https://proofwiki.org (MediaWiki API) |
| License      | CC BY-SA 3.0 (https://proofwiki.org/wiki/ProofWiki:Copyrights) |
| Date fetched | (see file mtime) |
| Records      | {n_written} pages |
| Max cap      | {MAX_PAGES} pages |
| Format       | JSONL — doc_id, name, env_type, statement, slogan, source, title, category |
| File         | theorems.jsonl |
| Size         | {size_mb:.1f} MB |

## Notes
- Only pages with a "Theorem", "Statement", or "Definition" section heading included.
- statement: content of ==Statement== / ==Theorem== section (up to 3000 chars).
- source: ProofWiki URL for the page.
- category: first [[Category:...]] tag.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
