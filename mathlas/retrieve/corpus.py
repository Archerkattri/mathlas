"""Build OUR OWN retrieval corpus from the open theorem-search dataset.

The dataset (``uw-math-ai/theorem-search-dataset``, CC-BY/CC0) is used here as
RAW DATA only -- we read the released parquet files and assemble our own
``Document`` records. We do NOT call any external service or reuse any
third-party running code; the index is built and served entirely by us.

Per the load-bearing lesson, the *embedded text* of each document is the
natural-language **slogan** (an LLM's symbol-stripped denotation of the
theorem), not the raw LaTeX body. We additionally keep the body + name so the
BM25 (exact-term) channel can match on operator/space names the slogan elides,
and so the MAP step sees the real statement.

This module is pure data plumbing (pyarrow). Indexing/serving is in
``retrieve/hybrid.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


@dataclass(frozen=True)
class Document:
    """One corpus entry: an existing result, with its NL denotation + source."""
    doc_id: str
    slogan: str                 # NL denotation -> the dense-embedded text
    statement: str              # the real (LaTeX) statement -> shown to MAP/user
    name: Optional[str] = None
    source: Optional[str] = None  # arXiv link / identifier
    title: Optional[str] = None   # paper title

    @property
    def embed_text(self) -> str:
        """What the dense embedder sees: slogan first (the meaning), with the
        name appended so titled lemmas remain findable; never the raw body."""
        parts = [p for p in (self.name, self.slogan) if p]
        return " -- ".join(parts) if parts else self.statement

    @property
    def sparse_text(self) -> str:
        """What BM25 sees: name + slogan + statement, so exact symbol/term hits
        land even when the slogan paraphrased them away."""
        return " ".join(p for p in (self.name, self.slogan, self.statement) if p)


def _require_pyarrow() -> None:
    if pq is None:  # pragma: no cover
        raise ImportError("pip install pyarrow to read the dataset parquet files")


def load_documents(dataset_dir: str, limit: Optional[int] = None,
                   slogan_model: str = "DeepSeek-V3.1",
                   include_paper_ids: Optional[set] = None) -> List[Document]:
    """Read the dataset parquets and assemble ``Document`` records.

    ``limit`` caps how many theorems are loaded (use a SMALL value for
    validation -- the full corpus is ~1.3M and meant for an offline GPU build).
    One slogan per theorem is kept (the first row matching ``slogan_model``).

    ``include_paper_ids`` (optional) forces all theorems from those papers to be
    loaded REGARDLESS of ``limit`` -- used by the retrieval eval to guarantee the
    benchmark's target theorems are in the (otherwise small) index, alongside
    ``limit`` distractor theorems.
    """
    _require_pyarrow()
    theo_path = os.path.join(dataset_dir, "theorem.parquet")
    slog_path = os.path.join(dataset_dir, "theorem_slogan.parquet")
    paper_path = os.path.join(dataset_dir, "paper.parquet")
    include_paper_ids = {str(p) for p in (include_paper_ids or set())}

    # 1) theorem bodies (streamed). Always keep theorems from `include_paper_ids`;
    #    additionally keep the first `limit` theorems overall as distractors.
    theorems: Dict[str, dict] = {}
    n_distractor = 0
    done = False
    for batch in pq.ParquetFile(theo_path).iter_batches(batch_size=4096):
        for r in batch.to_pylist():
            tid = str(r["theorem_id"])
            forced = str(r["paper_id"]) in include_paper_ids
            if forced:
                theorems[tid] = r
            elif not done:
                theorems[tid] = r
                n_distractor += 1
                if limit and n_distractor >= limit:
                    done = True
        # stop only once distractor cap is met AND no forced papers remain to find
        if done and not include_paper_ids:
            break
    want = set(theorems)

    # 2) one slogan per wanted theorem.
    slogans: Dict[str, str] = {}
    for batch in pq.ParquetFile(slog_path).iter_batches(batch_size=8192):
        for r in batch.to_pylist():
            tid = str(r["theorem_id"])
            if tid in want and tid not in slogans and r.get("model") == slogan_model:
                slogans[tid] = r["slogan"]
        if len(slogans) >= len(want):
            break

    # 3) paper metadata for the wanted papers only.
    want_papers = {str(t["paper_id"]) for t in theorems.values()}
    papers: Dict[str, dict] = {}
    for batch in pq.ParquetFile(paper_path).iter_batches(batch_size=8192):
        for r in batch.to_pylist():
            pid = str(r["paper_id"])
            if pid in want_papers:
                papers[pid] = r
        if len(papers) >= len(want_papers):
            break

    docs: List[Document] = []
    for tid, t in theorems.items():
        paper = papers.get(str(t["paper_id"]), {})
        docs.append(Document(
            doc_id=tid,
            slogan=slogans.get(tid) or (t.get("body") or "").strip(),
            statement=(t.get("body") or "").strip(),
            name=(t.get("name") or None),
            source=t.get("link") or paper.get("link"),
            title=paper.get("title"),
        ))
    return docs


__all__ = ["Document", "load_documents"]
