"""WEB-AUGMENTED RETRIEVAL — the local corpus is finite; the AI has the web.

mathlas's own index is finite (and offline). The CALLING AI, however, can search
the live web. This module is the bridge — and like everything in mathlas it makes
**NO web call and NO LLM call itself**; it (1) tells the AI exactly WHAT to search,
and (2) ingests what the AI brings back into the LIVE corpus so later queries
find it.

  * ``search_directive(problem)`` — analyses the problem (reusing the
    needs<->guarantees signature + light domain heuristics) and returns STRUCTURED
    search instructions: arXiv-style query strings, candidate sub-fields, named
    methods/inequalities/theorems to look for, and which OTHER mathlas tools to
    also run (OEIS / identify_constant / conjecture_relation / search_existing_math).
    mathlas does not browse — it hands the AI a search plan.

  * ``add_finding(statement, slogan, source, name?)`` — the AI feeds a web-found
    result back; mathlas APPENDS it to the live corpus through the BM25 / sparse
    channel + metadata, with **NO embedding-model load** (so it works on any
    machine — the 8B is never loaded per finding). Provenance is ``web_added``.
    THE KEY CONSTRAINT: growing the index must not require loading the embedder.
    A finding is persisted to a JSONL sidecar and indexed by pure-Python BM25;
    ``search_existing_math`` fuses these in. If (and only if) a dense Qwen3 index
    is ALREADY loaded in-process, its query encoder is reused to also give the
    finding a dense vector; otherwise dense is skipped (BM25 covers it), and a
    separate batch ``reindex`` (documented) embeds the backlog later on a GPU box.

NO LLM. NO network. NO API key. Findings are honestly labelled ``web_added`` —
an AI-sourced, NOT-yet-independently-verified pointer the AI must still check
(via verify_numeric / applicability_checklist / a primary source).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .retrieve.bm25 import BM25Index, tokenize


# --------------------------------------------------------------------------- #
# Persistence: the live corpus is a JSONL sidecar (one finding per line). It is
# gitignored (reference/ is ignored wholesale) and append-only. No DB, no model.
# --------------------------------------------------------------------------- #
def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def findings_path() -> str:
    """Path to the live-corpus JSONL. Overridable via ``MATHLAS_FINDINGS`` (tests
    / per-project corpora). Lives under reference/downloads (gitignored)."""
    p = os.environ.get("MATHLAS_FINDINGS")
    if p:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        return p
    d = os.path.join(_repo_root(), "reference", "downloads")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "findings.jsonl")


def load_findings(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read all live-corpus findings (possibly empty). Tolerant of partial/blank
    lines so a half-written append never breaks reads."""
    path = path or findings_path()
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _sparse_text(rec: Dict[str, Any]) -> str:
    """What BM25 sees for a finding — mirrors Document.sparse_text (name + slogan
    + statement + source) so a finding ranks on the same exact-term signal."""
    return " ".join(str(rec.get(k) or "") for k in
                    ("name", "slogan", "statement", "source"))


@dataclass(frozen=True)
class AddFindingResult:
    ok: bool
    statement: str
    name: Optional[str]
    source: str
    slogan: str
    provenance: str                # "web_added"
    dense_added: bool              # True only if a loaded Qwen3 index got a vector
    n_findings: int                # total live-corpus size after the add
    path: str
    note: str


def add_finding(statement: str, slogan: str, source: str,
                name: Optional[str] = None,
                path: Optional[str] = None) -> AddFindingResult:
    """Append a web-found result to the LIVE corpus — NO embedding-model load.

    The finding is persisted to the JSONL sidecar and becomes retrievable through
    the BM25 / sparse channel immediately (``search_existing_math`` fuses live
    findings in). Provenance is ``web_added`` (AI-sourced, NOT yet independently
    verified — the AI must still check it). If a dense Qwen3 index is ALREADY
    loaded in this process, its query encoder is reused to attach a dense vector
    too; otherwise dense is skipped and the documented batch ``reindex`` will
    embed the backlog later (the 8B is never loaded here).
    """
    statement = (statement or "").strip()
    slogan = (slogan or "").strip() or statement
    source = (source or "").strip()
    if not statement:
        raise ValueError("add_finding requires a non-empty `statement`")
    if not source:
        raise ValueError("add_finding requires a `source` (URL / arXiv id / cite)")
    p = path or findings_path()
    rec: Dict[str, Any] = {
        "doc_id": f"web::{int(time.time()*1000)}::{abs(hash(statement)) % 10**8}",
        "name": (name or "").strip() or None,
        "slogan": slogan,
        "statement": statement,
        "source": source,
        "provenance": "web_added",
        "added_ts": time.time(),
        "dense": False,
    }

    # OPTIONAL dense: only if a Qwen3 index is ALREADY loaded in-process. We never
    # construct an embedder here (that would load the model). We sniff the server's
    # retriever cache for an already-served Qwen3 index and reuse its query encoder.
    dense_added = False
    vec = _maybe_dense_vector(rec["slogan"], rec["name"])
    if vec is not None:
        rec["dense"] = True
        rec["dense_vec"] = [float(x) for x in vec]
        dense_added = True

    tmp = p + ".tmp"
    # append atomically: write the new line to a tmp and concatenate is overkill;
    # a plain append with flush is fine for JSONL (line-atomic on POSIX for small
    # lines). We still write+fsync to be safe across crashes.
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(tmp):
        os.remove(tmp)

    n = len(load_findings(p))
    note = ("Appended to the live corpus via BM25/sparse — NO embedding model "
            "loaded. Retrievable now through search_existing_math (RRF-fused). "
            "Provenance web_added: AI-sourced, NOT yet independently verified — "
            "still check it (verify_numeric / applicability_checklist / source). "
            + ("A dense vector was added (a Qwen3 index was already loaded)."
               if dense_added else
               "Dense skipped (no embedder loaded); run the batch `reindex` "
               "(docs/04_build.md) to add dense vectors later on a GPU box."))
    return AddFindingResult(
        ok=True, statement=statement, name=rec["name"], source=source,
        slogan=slogan, provenance="web_added", dense_added=dense_added,
        n_findings=n, path=p, note=note)


def _maybe_dense_vector(slogan: str, name: Optional[str]):
    """Return a dense query-vector for the finding ONLY if a Qwen3-backed index is
    already loaded in this process (so no model gets loaded here); else ``None``.

    We look into ``server._RETRIEVER_CACHE`` for a served index whose embedder is
    a real ``Qwen3Embedder`` already in memory, and reuse it. A HashingEmbedder
    (the zero-download fallback) does NOT count as "a model is loaded" — but it is
    cheap and deterministic, so if the served index uses it we still reuse it
    (consistency with that index's space) without any download.
    """
    try:
        from . import server as _server
        from .embed import Qwen3Embedder, HashingEmbedder
        import numpy as np
    except Exception:
        return None
    cache = getattr(_server, "_RETRIEVER_CACHE", {})
    text = " -- ".join(p for p in (name, slogan) if p) or slogan
    for retr in cache.values():
        emb = getattr(retr, "embedder", None)
        if emb is None:
            continue
        # Reuse an already-loaded embedder. Qwen3 = a real model already resident;
        # Hashing = no model at all (safe to use, matches a hashing-built index).
        if isinstance(emb, (Qwen3Embedder, HashingEmbedder)):
            try:
                v = emb.encode([text], is_query=True)[0]
                return np.asarray(v, dtype=np.float32)
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# Live-corpus BM25 retrieval — pure Python, NO model. Cached per (path, mtime) so
# we rebuild only when the sidecar actually changes (data-flow discipline).
# --------------------------------------------------------------------------- #
_LIVE_CACHE: Dict[str, Any] = {}


def _live_index(path: str):
    """Return a cached ``(records, BM25Index)`` for the findings file, rebuilt
    only when the file's mtime changes."""
    if not os.path.exists(path):
        return [], None
    mtime = os.path.getmtime(path)
    key = f"{path}::{mtime}"
    hit = _LIVE_CACHE.get(key)
    if hit is not None:
        return hit
    recs = load_findings(path)
    bm = BM25Index([_sparse_text(r) for r in recs]) if recs else None
    _LIVE_CACHE.clear()              # only keep the latest mtime's index
    _LIVE_CACHE[key] = (recs, bm)
    return recs, bm


def search_findings(query: str, k: int = 10,
                    path: Optional[str] = None) -> List[Dict[str, Any]]:
    """BM25 search over the live (web-added) corpus only — NO model. Returns the
    matching findings best-first as plain dicts (the calling AI / the
    search_existing_math fuser consumes them)."""
    path = path or findings_path()
    recs, bm = _live_index(path)
    if not recs or bm is None:
        return []
    hits = bm.search(query, k=k)
    out: List[Dict[str, Any]] = []
    for idx, score in hits:
        r = dict(recs[idx])
        r["bm25_score"] = float(score)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# search_directive — tell the AI WHAT to search (mathlas makes NO web call).
# --------------------------------------------------------------------------- #
# Light domain heuristics: keyword -> (sub-fields, named methods/results to look
# for). Deliberately shallow and transparent; the AI refines. These are CUES, not
# claims — they bias the AI's web search toward the right corner of the literature.
_DOMAIN_CUES: List[tuple] = [
    (("fixed point", "contraction", "iterate", "converge to a point"),
     ["analysis", "metric geometry", "dynamical systems"],
     ["Banach fixed-point theorem", "Brouwer fixed-point theorem",
      "Kakutani fixed-point theorem", "contraction mapping principle"]),
    (("prime", "divisor", "modulo", "congruence", "integer", "gcd", "coprime"),
     ["number theory", "analytic number theory"],
     ["Chinese remainder theorem", "Dirichlet's theorem", "quadratic reciprocity",
      "Bertrand's postulate", "sieve methods"]),
    (("graph", "vertex", "edge", "coloring", "clique", "independent set", "matching"),
     ["graph theory", "combinatorics", "extremal combinatorics"],
     ["Ramsey's theorem", "Turán's theorem", "Hall's marriage theorem",
      "probabilistic method", "Szemerédi regularity lemma"]),
    (("inequality", "bound", "at most", "at least", "maximize", "minimize"),
     ["analysis", "convex optimization", "extremal combinatorics"],
     ["Cauchy-Schwarz", "Jensen's inequality", "AM-GM", "Hölder's inequality",
      "rearrangement inequality", "Chebyshev's inequality"]),
    (("matrix", "eigenvalue", "linear map", "determinant", "rank", "vector space"),
     ["linear algebra", "matrix analysis"],
     ["spectral theorem", "Cayley-Hamilton", "Perron-Frobenius theorem",
      "singular value decomposition", "Gershgorin circle theorem"]),
    (("probability", "random", "expected", "variance", "almost surely", "martingale"),
     ["probability theory", "stochastic processes"],
     ["law of large numbers", "central limit theorem", "Chernoff bound",
      "Azuma's inequality", "Borel-Cantelli lemma"]),
    (("continuous", "compact", "open set", "closed set", "limit", "sequence", "metric"),
     ["real analysis", "topology"],
     ["Heine-Borel theorem", "Bolzano-Weierstrass", "Arzelà-Ascoli",
      "Stone-Weierstrass theorem", "Baire category theorem"]),
    (("series", "sum", "converges", "zeta", "closed form", "constant", "evaluate"),
     ["analysis", "number theory", "special functions"],
     ["Euler-Maclaurin", "Abel summation", "Parseval's theorem",
      "Ramanujan summation", "PSLQ / integer-relation detection"]),
    (("group", "ring", "field", "homomorphism", "ideal", "subgroup", "galois"),
     ["abstract algebra", "Galois theory"],
     ["Lagrange's theorem", "Sylow theorems", "fundamental theorem of Galois theory",
      "structure theorem for finitely generated abelian groups"]),
    (("differential equation", "pde", "ode", "boundary value", "heat", "wave"),
     ["differential equations", "functional analysis"],
     ["Picard-Lindelöf theorem", "Lax-Milgram theorem", "maximum principle",
      "Sobolev embedding", "method of characteristics"]),
]

# arXiv math subject classes, for the AI to filter a search.
_ARXIV_CATS = {
    "number theory": "math.NT", "analytic number theory": "math.NT",
    "graph theory": "math.CO", "combinatorics": "math.CO",
    "extremal combinatorics": "math.CO", "analysis": "math.CA",
    "real analysis": "math.CA", "topology": "math.GN", "metric geometry": "math.MG",
    "linear algebra": "math.RA", "matrix analysis": "math.RA",
    "abstract algebra": "math.RA", "Galois theory": "math.NT",
    "probability theory": "math.PR", "stochastic processes": "math.PR",
    "convex optimization": "math.OC", "functional analysis": "math.FA",
    "differential equations": "math.AP", "dynamical systems": "math.DS",
    "special functions": "math.CA", "matrix geometry": "math.MG",
}

_NUMERIC_RE = re.compile(r"\d+\.\d{6,}")          # a high-precision decimal in text
_SEQUENCE_RE = re.compile(r"(?:-?\d+\s*,\s*){3,}-?\d+")  # >=4 comma-separated ints


@dataclass(frozen=True)
class SearchDirective:
    problem: str
    signature: Dict[str, Any]
    arxiv_queries: List[str]
    subfields: List[str]
    arxiv_categories: List[str]
    named_results: List[str]
    mathlas_tools: List[Dict[str, str]]
    instructions: str
    note: str


def _keywords(problem: str) -> List[str]:
    """Content words from the problem (lowercased, stop-words dropped) for arXiv
    query construction."""
    stop = {"the", "a", "an", "of", "to", "in", "is", "are", "that", "this",
            "for", "and", "or", "if", "then", "with", "we", "it", "be", "on",
            "show", "prove", "find", "let", "such", "there", "exists", "all",
            "any", "every", "which", "has", "have", "can", "does", "do", "by",
            "as", "at", "from", "into", "where", "when", "whether", "given"}
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", problem.lower())
             if w not in stop and len(w) > 2]
    # keep order, de-dup
    seen = set()
    out = [w for w in words if not (w in seen or seen.add(w))]
    return out


def search_directive(problem: str) -> SearchDirective:
    """Analyse ``problem`` and return STRUCTURED web-search instructions for the
    calling AI — mathlas makes NO web call, it tells the AI WHAT to search.

    Reuses the needs<->guarantees signature (objects/need/given) plus transparent
    domain cues to propose: arXiv-style query strings, candidate sub-fields +
    their arXiv categories, named methods/inequalities/theorems to look for, and
    which OTHER mathlas tools to also run. The AI does the actual searching and
    feeds results back via ``add_finding``.
    """
    from .map import _heuristic_signature           # reuse the NO-LLM signature
    text = (problem or "").strip()
    sig = _heuristic_signature(text)
    low = text.lower()

    subfields: List[str] = []
    named: List[str] = []
    for cues, fields, results in _DOMAIN_CUES:
        if any(c in low for c in cues):
            subfields.extend(fields)
            named.extend(results)
    # de-dup, preserve order
    subfields = list(dict.fromkeys(subfields))
    named = list(dict.fromkeys(named))
    cats = list(dict.fromkeys(_ARXIV_CATS.get(s) for s in subfields
                              if _ARXIV_CATS.get(s)))

    kws = _keywords(text)
    need = (sig.get("need") or text)[:120]
    # a couple of arXiv-style query strings: a need-phrase query and a keyword query.
    arxiv_queries: List[str] = []
    if kws:
        arxiv_queries.append(" ".join(kws[:6]))
    if need:
        arxiv_queries.append(need)
    if named:
        arxiv_queries.append(" OR ".join(f'"{n}"' for n in named[:3]))
    arxiv_queries = list(dict.fromkeys(q for q in arxiv_queries if q.strip()))

    # which OTHER mathlas tools to run, by detected content.
    tools: List[Dict[str, str]] = [
        {"tool": "search_existing_math",
         "why": "first check mathlas's OWN index (it also fuses in web findings "
                "you add via add_finding)."}]
    if _NUMERIC_RE.search(text):
        tools.append({"tool": "identify_constant",
                      "why": "the problem contains a high-precision decimal — try "
                             "an airtight closed form."})
        tools.append({"tool": "conjecture_relation",
                      "why": "go beyond the flat basis: PSLQ over products/powers "
                             "+ a Ramanujan-Machine continued-fraction conjecture."})
    if _SEQUENCE_RE.search(text):
        tools.append({"tool": "identify_sequence",
                      "why": "the problem lists an integer sequence — exact OEIS "
                             "term-match."})
    if "constant" in low or "series" in low or "sum" in low or "product" in low:
        tools.append({"tool": "conjecture_relation",
                      "why": "a constant/series value may match a known relation "
                             "or continued fraction (verified, not proved)."})
    # de-dup tools by name (keep first reason)
    seen_t = set()
    tools = [t for t in tools if not (t["tool"] in seen_t or seen_t.add(t["tool"]))]

    instructions = (
        "1) Run the suggested mathlas tools first (they are airtight / local). "
        "2) Search the web (arXiv, Google Scholar, MathOverflow, Wikipedia, the "
        "DLMF for special functions) using the arxiv_queries and named_results, "
        "filtered to the arxiv_categories. 3) For each promising result, feed it "
        "back with add_finding(statement, slogan, source) so it enters the live "
        "corpus and future search_existing_math calls surface it. 4) Then VERIFY: "
        "applicability_checklist + mapping_scaffold for theorems, verify_numeric "
        "for numeric claims — a web hit is a lead, not a proof.")
    note = ("mathlas made NO web call and used NO LLM — this is a search PLAN of "
            "DATA. The calling AI does the searching and the judging; feed results "
            "back via add_finding (provenance web_added).")
    return SearchDirective(
        problem=text, signature=sig, arxiv_queries=arxiv_queries,
        subfields=subfields, arxiv_categories=cats, named_results=named,
        mathlas_tools=tools, instructions=instructions, note=note)


__all__ = [
    "search_directive", "SearchDirective",
    "add_finding", "AddFindingResult",
    "load_findings", "search_findings", "findings_path",
]
