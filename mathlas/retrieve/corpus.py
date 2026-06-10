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
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None  # type: ignore[assignment]


# A label is "meaningful" (worth humanising into a query-side signal) only if it
# reads like words, not a Stacks tag ("00IV") or a bare counter ("thm:1"/"eq:3").
# Their dense unit is name+slogan (proven); we add a label to the dense text ONLY
# when it is descriptive, and always to BM25 (where even a tag is a free exact key).
_CRYPTIC_LABEL = re.compile(
    r"^(?:[0-9A-Za-z]{3,5}"                     # Stacks-style tag, e.g. 00IV / 06UU
    r"|(?:thm|theorem|lem|lemma|prop|proposition|cor|corollary|eq|equation|"
    r"def|definition|rmk|remark|sec|section|fig|figure|ex|example|claim|"
    r"conj|conjecture|pf|proof|item|enu|enum|part|step|case|tag|label)"
    r"[-_:.]?\d*)$",
    re.IGNORECASE,
)


def _humanize_label(label: Optional[str]) -> Optional[str]:
    """Turn a descriptive ``\\label{}`` into a short NL phrase, or ``None`` if the
    label is cryptic (a Stacks tag / bare counter the dense channel shouldn't see).

    ``lemma-adequate-finite-presentation`` -> ``adequate finite presentation``
    ``thm:main`` -> ``main``;  ``00IV`` / ``thm:1`` / ``eq-3`` -> ``None``.
    """
    if not label:
        return None
    raw = label.strip()
    if not raw or _CRYPTIC_LABEL.match(raw):
        return None
    # split on label punctuation, drop the leading environment kind, keep words.
    parts = [p for p in re.split(r"[-_:.\s]+", raw) if p]
    kinds = {"thm", "theorem", "lem", "lemma", "prop", "proposition", "cor",
             "corollary", "eq", "equation", "def", "definition", "rmk", "remark",
             "sec", "section", "fig", "figure", "ex", "example", "claim", "conj",
             "conjecture", "pf", "proof", "prop", "lemma", "tag", "label"}
    words = [p for p in parts if p.lower() not in kinds and not p.isdigit()]
    if not words:
        return None
    phrase = " ".join(words).strip()
    # require it to actually look like words (>=2 alpha words, or one long word).
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha_words or (len(alpha_words) < 2 and len(phrase) < 6):
        return None
    return phrase


def _coerce_citations(v) -> Optional[int]:
    """Dataset ``citations`` arrives as float (0.0), int, ``None``, or the string
    'None' (Stacks rows). Coerce to a non-negative int, else ``None``."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            n = int(v)
        except (ValueError, OverflowError):
            return None
        return n if n >= 0 else None
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan", ""):
        return None
    try:
        n = int(float(s))
    except ValueError:
        return None
    return n if n >= 0 else None


def _coerce_category(v) -> Optional[str]:
    """Primary arXiv category string (e.g. ``math.AG``); 'None'/empty -> ``None``."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan"):
        return None
    return s


#: Canonical corpus-source keys (the vocabulary ``source_filter`` /
#: ``source_weights`` accept). Derived from each doc's provenance, NOT stored:
#: the served meta's ``source`` field is a URL/identifier string —
#: ``https://arxiv.org/abs/...`` for the TheoremSearch base, ``https://stacks.
#: math.columbia.edu/tag/...`` for Stacks, a ProofWiki URL, and the literal
#: ``"arXiv (dolma-v1_7)"`` (plus a ``dolma:`` doc_id prefix) for the web-mined
#: Dolma docs. Everything else (textbook PDFs etc.) is ``other``.
KNOWN_SOURCE_KEYS = ("arxiv", "dolma", "stacks", "proofwiki", "other")


def source_key(doc_id: Optional[str], source: Optional[str]) -> str:
    """Map a document's provenance to one of :data:`KNOWN_SOURCE_KEYS`.

    Order matters: the Dolma docs' ``source`` string *contains* "arXiv"
    ("arXiv (dolma-v1_7)"), so dolma is tested before arxiv. The ``dolma:``
    doc_id prefix is the strongest signal (every Dolma row carries it).
    """
    if (doc_id or "").startswith("dolma:"):
        return "dolma"
    s = (source or "").lower()
    if "dolma" in s:
        return "dolma"
    if "stacks.math" in s:
        return "stacks"
    if "proofwiki" in s:
        return "proofwiki"
    if "arxiv" in s:
        return "arxiv"
    return "other"


@dataclass(frozen=True)
class Document:
    """One corpus entry: an existing result, with its NL denotation + source."""
    doc_id: str
    slogan: str                 # NL denotation -> the dense-embedded text
    statement: str              # the real (LaTeX) statement -> shown to MAP/user
    name: Optional[str] = None
    source: Optional[str] = None  # arXiv link / identifier
    title: Optional[str] = None   # paper title
    label: Optional[str] = None   # LaTeX \label{} tag -> BM25 exact key (+ maybe dense)
    citations: Optional[int] = None  # paper citation count -> optional rerank prior
    category: Optional[str] = None   # paper primary arXiv category, e.g. "math.AG"
    #: when True, a *humanised* meaningful label is appended to the dense text.
    #: Off by default: their proven dense unit is name+slogan. Flip on only if a
    #: dev test shows the descriptive label helps (cryptic tags are never added).
    label_in_embed: bool = False

    @property
    def embed_text(self) -> str:
        """What the dense embedder sees: slogan first (the meaning), with the
        name appended so titled lemmas remain findable; never the raw body. A
        humanised *meaningful* label is appended only when ``label_in_embed`` and
        the label is descriptive (cryptic Stacks tags / bare counters excluded)."""
        parts = [p for p in (self.name, self.slogan) if p]
        if self.label_in_embed:
            hl = _humanize_label(self.label)
            if hl and hl.lower() not in (self.slogan or "").lower():
                parts.append(hl)
        return " -- ".join(parts) if parts else self.statement

    @property
    def sparse_text(self) -> str:
        """What BM25 sees: name + slogan + statement + label, so exact symbol/term
        hits land even when the slogan paraphrased them away. The raw ``label`` is
        included verbatim (even a Stacks tag is a free exact key for BM25)."""
        return " ".join(p for p in (self.name, self.slogan, self.statement,
                                    self.label) if p)


def _require_pyarrow() -> None:
    if pq is None:  # pragma: no cover
        raise ImportError("pip install pyarrow to read the dataset parquet files")


def load_documents(dataset_dir: str, limit: Optional[int] = None,
                   slogan_model: str = "DeepSeek-V3.1",
                   include_paper_ids: Optional[set] = None,
                   label_in_embed: bool = False) -> List[Document]:
    """Read the dataset parquets and assemble ``Document`` records.

    ``limit`` caps how many theorems are loaded (use a SMALL value for
    validation -- the full corpus is ~1.3M and meant for an offline GPU build).
    One slogan per theorem is kept (the first row matching ``slogan_model``).

    ``include_paper_ids`` (optional) forces all theorems from those papers to be
    loaded REGARDLESS of ``limit`` -- used by the retrieval eval to guarantee the
    benchmark's target theorems are in the (otherwise small) index, alongside
    ``limit`` distractor theorems.

    ``label_in_embed`` (default False) appends a *humanised meaningful* label to
    each document's dense ``embed_text``. Their proven dense unit is name+slogan;
    enable this only if a dev test shows the descriptive label helps. The raw
    label always feeds the BM25 channel regardless. Paper ``citations`` and
    ``primary_category`` are populated for the optional citation-weighted rerank
    and category metadata.
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
            label=(t.get("label") or None),
            citations=_coerce_citations(paper.get("citations")),
            category=_coerce_category(paper.get("primary_category")),
            label_in_embed=label_in_embed,
        ))
    return docs


__all__ = ["Document", "load_documents", "doc_from_meta"]


def doc_from_meta(m: dict, label_in_embed: bool = False) -> Document:
    """Rebuild a ``Document`` from a serialized index-meta dict (the jsonl/npz
    rows written by ``scripts/build_index_mp.py``). Tolerant of older metas that
    lack the newer ``label``/``citations``/``category`` keys."""
    return Document(
        doc_id=str(m.get("doc_id")),
        slogan=m.get("slogan") or "",
        statement=m.get("statement") or "",
        name=m.get("name"),
        source=m.get("source"),
        title=m.get("title"),
        label=m.get("label"),
        citations=_coerce_citations(m.get("citations")),
        category=_coerce_category(m.get("category")),
        label_in_embed=label_in_embed,
    )
