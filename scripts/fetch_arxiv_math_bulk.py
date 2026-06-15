"""
Phase 2: Bulk arXiv math metadata (title + abstract + categories + id).

Strategy (in priority order):
  1. HuggingFace Hub dataset `togethercomputer/RedPajama-Data-1T-Sample` — too mixed.
     Better: `trizah/math-papers-dataset` or `rcav8/arxiv-math` etc.
     We try a few known HF datasets that contain arXiv math metadata.
  2. arXiv OAI-PMH: http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv&set=math
     Follow resumptionToken, ~1000/page, polite delays (20s between pages).

Output:
  reference/downloads/datasets/arxiv_math/papers.jsonl
  (one JSON per line: doc_id, paper_id/arxiv_id, title, abstract, categories, source_link)

CAP: 30 GB disk / 300 000 records (OAI-PMH gives ~350k math papers total, we cap at 200k).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "reference" / "downloads" / "datasets" / "arxiv_math"
OUT_FILE = OUT_DIR / "papers.jsonl"

HEADERS = {
    "User-Agent": (
        "mathlas-benchmark/1.0 (research; github.com/Archerkattri/mathlas; "
        "contact: krishiattriwork@gmail.com)"
    )
}

MAX_RECORDS = 200_000
DISK_CAP_BYTES = 28 * 1024 ** 3   # 28 GB
OAI_SLEEP = 22          # polite: OAI-PMH asks for >= 20s between pages
OAI_TIMEOUT = 60
OAI_RETRIES = 4

OAI_BASE = "http://export.arxiv.org/oai2"
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}


def _disk_used_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _oai_request(params: dict, retries: int = OAI_RETRIES) -> bytes:
    url = OAI_BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=OAI_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                retry_after = int(e.headers.get("Retry-After", 30))
                print(f"  [503] retry after {retry_after}s ...")
                time.sleep(retry_after + 5)
            elif e.code == 429:
                print(f"  [429] rate limited, sleeping 60s ...")
                time.sleep(60)
            else:
                print(f"  [HTTPError {e.code}] {e}")
                time.sleep(15)
        except Exception as e:
            print(f"  [Error {type(e).__name__}] {e}")
            time.sleep(15)
    raise RuntimeError(f"Failed OAI request after {retries} retries: {params}")


def _parse_oai_page(xml_bytes: bytes) -> tuple[list[dict], str | None]:
    """
    Parse one OAI-PMH ListRecords XML response.
    Returns (records, resumption_token_or_None).
    """
    root = ET.fromstring(xml_bytes)
    records = []

    for record in root.findall(".//oai:record", OAI_NS):
        # Check if deleted
        header = record.find("oai:header", OAI_NS)
        if header is not None and header.attrib.get("status") == "deleted":
            continue

        metadata = record.find("oai:metadata/arxiv:arXiv", OAI_NS)
        if metadata is None:
            continue

        def txt(tag: str) -> str | None:
            el = metadata.find(f"arxiv:{tag}", OAI_NS)
            return el.text.strip() if el is not None and el.text else None

        arxiv_id = txt("id")
        title = txt("title")
        abstract = txt("abstract")
        # categories
        cats_el = metadata.find("arxiv:categories", OAI_NS)
        categories = cats_el.text.strip().split() if (cats_el is not None and cats_el.text) else []
        # authors (first author only to save space)
        authors = []
        for a in metadata.findall("arxiv:authors/arxiv:author", OAI_NS):
            keyname = a.find("arxiv:keyname", OAI_NS)
            fname = a.find("arxiv:forenames", OAI_NS)
            name = " ".join(
                p for p in [(fname.text if fname is not None else None),
                            (keyname.text if keyname is not None else None)] if p
            )
            if name:
                authors.append(name)

        if not arxiv_id:
            continue

        records.append({
            "doc_id": f"arxiv::{arxiv_id}",
            "paper_id": arxiv_id,
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "categories": categories,
            "primary_category": categories[0] if categories else None,
            "authors": authors[:3],   # cap to 3 to keep JSON small
            "source_link": f"https://arxiv.org/abs/{arxiv_id}",
        })

    # Resumption token
    token_el = root.find(".//oai:resumptionToken", OAI_NS)
    resumption_token = None
    if token_el is not None and token_el.text and token_el.text.strip():
        resumption_token = token_el.text.strip()

    return records, resumption_token


def fetch_via_oai_pmh(out_fh, n_already: int = 0) -> int:
    """
    Stream arXiv math papers via OAI-PMH into out_fh.
    Returns number of records written.
    """
    print("Starting OAI-PMH fetch (math set) ...")
    params: dict = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": "math",
    }
    page = 0
    n_written = n_already

    while n_written < MAX_RECORDS:
        page += 1
        print(f"  [page {page}] fetching, total so far = {n_written} ...", flush=True)
        try:
            raw = _oai_request(params)
        except RuntimeError as e:
            print(f"  [ABORT] {e}")
            break

        records, token = _parse_oai_page(raw)
        for rec in records:
            if n_written >= MAX_RECORDS:
                break
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

        if _disk_used_bytes(OUT_FILE) > DISK_CAP_BYTES:
            print(f"  [DISK CAP] {DISK_CAP_BYTES // 1024**3} GB reached, stopping.")
            break

        if not token:
            print("  [DONE] No resumption token — OAI-PMH exhausted.")
            break

        params = {"verb": "ListRecords", "resumptionToken": token}
        print(f"  Sleeping {OAI_SLEEP}s (polite) ...")
        time.sleep(OAI_SLEEP)

    return n_written


def try_huggingface(out_fh) -> int:
    """
    Try to fetch a known HuggingFace arXiv math metadata dataset.
    Returns number of records written, or 0 if unavailable.

    We try datasets.load_dataset with streaming=True, which is memory-safe
    and doesn't require a GPU.
    """
    try:
        import datasets as hf_datasets  # type: ignore
    except ImportError:
        print("  [HF] datasets not installed, skipping HF path.")
        return 0

    # Try known HF datasets with arXiv math metadata
    candidates = [
        # (dataset_id, split, id_col, title_col, abstract_col, cat_col)
        ("rcav8/arxiv-math", "train", "id", "title", "abstract", "categories"),
        ("trizah/math-papers-dataset", "train", "id", "title", "abstract", None),
    ]

    for ds_id, split, id_col, title_col, abstr_col, cat_col in candidates:
        try:
            print(f"  [HF] Trying {ds_id} ...")
            ds = hf_datasets.load_dataset(ds_id, split=split, streaming=True, trust_remote_code=False)
            n = 0
            for row in ds:
                arxiv_id = str(row.get(id_col, "")).strip()
                if not arxiv_id:
                    continue
                cats = row.get(cat_col, []) if cat_col else []
                if isinstance(cats, str):
                    cats = cats.split()
                rec = {
                    "doc_id": f"arxiv::{arxiv_id}",
                    "paper_id": arxiv_id,
                    "arxiv_id": arxiv_id,
                    "title": row.get(title_col),
                    "abstract": row.get(abstr_col),
                    "categories": cats,
                    "primary_category": cats[0] if cats else None,
                    "source_link": f"https://arxiv.org/abs/{arxiv_id}",
                }
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n >= MAX_RECORDS:
                    break
                if n % 10000 == 0:
                    print(f"    ... {n} records written from HF")
            if n > 0:
                print(f"  [HF] Got {n} records from {ds_id}")
                return n
        except Exception as e:
            print(f"  [HF] {ds_id} failed: {e}")
            continue
    return 0


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count how many we already have (resumable)
    n_already = 0
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            n_already = sum(1 for _ in f)
        print(f"Resuming: {n_already} records already in {OUT_FILE}")
        if n_already >= MAX_RECORDS:
            print("Already at cap. Done.")
            return

    mode = "a" if n_already > 0 else "w"
    with open(OUT_FILE, mode, encoding="utf-8", buffering=1) as fh:
        # Try HF first
        n_hf = try_huggingface(fh)
        n_written = n_already + n_hf

        if n_written < MAX_RECORDS:
            # Fall back to OAI-PMH
            n_written = fetch_via_oai_pmh(fh, n_written)

    final_size = _disk_used_bytes(OUT_FILE)
    print(f"\n=== Phase 2 Complete ===")
    print(f"Records written: {n_written}")
    print(f"Disk used:       {final_size / 1024**2:.1f} MB")
    print(f"Output:          {OUT_FILE}")


if __name__ == "__main__":
    main()
