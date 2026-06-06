"""mathlas MCP server — the AI-callable toolbox. NO LLM, NO API key.

mathlas is a tool that an AI *uses*, not a tool that uses an AI. This server
exposes mathlas's capabilities over the Model Context Protocol so any MCP client
(Claude Code, Cursor, or any agent) can call them. Every tool runs with **NO LLM
and NO API key** and returns DATA for the calling AI to reason over — the AI is
the brain; mathlas provides search over existing math, airtight numeric/formal
verification, structured needs<->guarantees scaffolds, and provenance.

Tools exposed
-------------
  identify_constant(value, basis?)          airtight closed-form + provenance
  identify_sequence(terms, max_results?)    airtight OEIS exact term-match
  search_existing_math(query, k, corpus_dir?) ranked candidate existing results
  verify_numeric(value, closed_form)        airtight digit-agreement verdict
  verify_formal(statement, lean?)           REAL Lean kernel typecheck (or honest UNDETERMINED)
  applicability_checklist(candidate_statement) the result's preconditions, structured
  mapping_scaffold(problem, candidate_statement) the needs<->guarantees questions
  -- discovery + web-augmentation layer (NO LLM; each returns DATA) --
  conjecture_relation(value, max_terms?, cf_depth?)  Ramanujan-Machine: PSLQ-richer-basis
                                            + continued-fraction conjectures (VERIFIED, not proved)
  funsearch_evaluate(program_src, problem_id)  sandbox-score an AI-written program
  funsearch_register(program_src, score, problem_id)  store it in a MAP-Elites DB
  funsearch_status(problem_id)              best program(s) + few-shot for the next variant
  search_directive(problem)                 a STRUCTURED web-search plan for the AI (no web call)
  add_finding(statement, slogan, source, name?)  ingest a web result into the live corpus (no model load)

Register in Claude Code (no API key needed):

    claude mcp add mathlas -- python -m mathlas.server

or run directly over stdio:  ``python -m mathlas.server``.

Implementation: uses the official ``mcp`` Python SDK (FastMCP) if installed;
otherwise falls back to a dependency-free stdio JSON-RPC MCP server implementing
the same tools and the same wire protocol.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Tool implementations — plain functions, NO LLM, returning JSON-able DATA.
# These are the single source of truth; both the FastMCP and the fallback server
# call them. Each returns a dict the calling AI consumes.
# --------------------------------------------------------------------------- #

#: A tiny, self-contained seed corpus of well-known results so that
#: ``search_existing_math`` works with ZERO downloads / GPU / corpus when no
#: ``corpus_dir`` is supplied. For a real index, point ``corpus_dir`` at the open
#: theorem dataset (parquets) — see scripts/build_index.py / docs/04_build.md.
SEED_CORPUS: List[Dict[str, str]] = [
    {"name": "Mean Value Theorem",
     "statement": "If f is continuous on [a, b] and differentiable on (a, b), "
                  "then there exists c in (a, b) such that f'(c) = (f(b) - f(a)) / (b - a).",
     "source": "calculus"},
    {"name": "Intermediate Value Theorem",
     "statement": "If f is continuous on [a, b] and y lies between f(a) and f(b), "
                  "then there exists c in [a, b] such that f(c) = y.",
     "source": "calculus"},
    {"name": "Banach Fixed-Point Theorem",
     "statement": "Let (X, d) be a complete metric space and T: X -> X a "
                  "contraction. Then T has a unique fixed point, and the iteration "
                  "x_{n+1} = T(x_n) converges to it from any start.",
     "source": "analysis"},
    {"name": "Cauchy-Schwarz Inequality",
     "statement": "For vectors u, v in an inner product space, "
                  "|<u, v>| <= ||u|| * ||v||, with equality iff u and v are linearly dependent.",
     "source": "linear-algebra"},
    {"name": "Bolzano-Weierstrass Theorem",
     "statement": "Every bounded sequence in R^n has a convergent subsequence.",
     "source": "analysis"},
    {"name": "Pigeonhole Principle",
     "statement": "If n items are placed into m boxes with n > m, then at least "
                  "one box contains more than one item.",
     "source": "combinatorics"},
    {"name": "Fundamental Theorem of Arithmetic",
     "statement": "Every integer greater than 1 is either prime or a product of "
                  "primes, unique up to the order of the factors.",
     "source": "number-theory"},
    {"name": "Lagrange's Theorem (group theory)",
     "statement": "If G is a finite group and H is a subgroup of G, then the "
                  "order of H divides the order of G.",
     "source": "algebra"},
    {"name": "Cayley-Hamilton Theorem",
     "statement": "Every square matrix over a commutative ring satisfies its own "
                  "characteristic equation.",
     "source": "linear-algebra"},
    {"name": "Heine-Borel Theorem",
     "statement": "A subset of R^n is compact if and only if it is closed and bounded.",
     "source": "topology"},
    {"name": "Brouwer Fixed-Point Theorem",
     "statement": "Every continuous function from a convex compact subset of R^n "
                  "to itself has a fixed point.",
     "source": "topology"},
    {"name": "Basel problem (Euler)",
     "statement": "The sum over n >= 1 of 1/n^2 equals pi^2 / 6.",
     "source": "analysis"},
]

# Cache built retrievers by corpus key so we do not re-index per call (data-flow
# discipline: every per-call build must be cached).
_RETRIEVER_CACHE: Dict[str, Any] = {}

#: Default location of a prebuilt index (the offline Qwen3-Embedding-8B build).
#: Overridable via the ``MATHLAS_INDEX`` env var. If present, ``search_existing_math``
#: serves it (precomputed dense matrix + BM25) instead of the tiny seed corpus.
_DEFAULT_INDEX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference", "downloads", "index.npz")


def _resolve_index_path() -> Optional[str]:
    """The prebuilt-index path to serve, or ``None``. ``MATHLAS_INDEX`` wins; else
    the default build location is used IF it exists. ``MATHLAS_INDEX`` set but
    missing is an explicit opt-in error (don't silently fall back and confuse)."""
    env = os.environ.get("MATHLAS_INDEX")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(
                f"MATHLAS_INDEX={env} does not exist (set it to a built "
                f"index.npz, or unset it to use the seed corpus).")
        return env
    return _DEFAULT_INDEX if os.path.exists(_DEFAULT_INDEX) else None


def _embedder_for_index(index_path: str):
    """Pick a query-time embedder matching the index's matrix.

    Reads the index's stored ``model``/``embedder``/``dim`` and constructs the
    matching embedder so query and document vectors share a space. A "qwen3"
    index uses ``Qwen3Embedder`` (CPU at query time, no GPU needed for a single
    query); a "hashing" index (CPU dev-smoke) uses ``HashingEmbedder`` at the
    stored dim."""
    import numpy as np
    from .embed import HashingEmbedder, Qwen3Embedder
    head = np.load(index_path, allow_pickle=True)
    kind = str(head["embedder"]) if "embedder" in head else "qwen3"
    model = str(head["model"]) if "model" in head else "Qwen/Qwen3-Embedding-8B"
    dim = int(head["dim"]) if "dim" in head else None
    if kind == "hashing":
        return HashingEmbedder(dim=dim or 256)
    # production: same Qwen3 model the matrix was built with (CPU query encode).
    return Qwen3Embedder(model=model, dim=dim, device="cpu")


def _build_retriever(corpus_dir: Optional[str], limit: int):
    """Return a cached HybridRetriever. Resolution order:

      1. explicit ``corpus_dir`` -> build from those dataset parquets (hashing);
      2. no ``corpus_dir`` but a prebuilt index (``MATHLAS_INDEX`` / default
         location) -> SERVE it (precomputed dense matrix + BM25, query-time embed);
      3. otherwise -> the built-in seed corpus (zero downloads, hashing).
    """
    from .retrieve.corpus import Document
    from .retrieve.hybrid import HybridRetriever

    if corpus_dir:
        key = f"dir::{corpus_dir}::{limit}"
        if key in _RETRIEVER_CACHE:
            return _RETRIEVER_CACHE[key]
        from .retrieve.corpus import load_documents
        docs = load_documents(corpus_dir, limit=limit)
        retr = HybridRetriever(docs)  # hashing embedder -> no model download
        _RETRIEVER_CACHE[key] = retr
        return retr

    index_path = _resolve_index_path()
    if index_path:
        key = f"index::{index_path}"
        if key in _RETRIEVER_CACHE:
            return _RETRIEVER_CACHE[key]
        retr = HybridRetriever.from_index(
            index_path, embedder=_embedder_for_index(index_path))
        _RETRIEVER_CACHE[key] = retr
        return retr

    key = "<seed>"
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]
    docs = [Document(doc_id=str(i), slogan=d["statement"],
                     statement=d["statement"], name=d["name"], source=d["source"])
            for i, d in enumerate(SEED_CORPUS)]
    retr = HybridRetriever(docs)  # default HashingEmbedder -> no model download
    _RETRIEVER_CACHE[key] = retr
    return retr


def tool_identify_constant(value: str, basis: Optional[List[str]] = None) -> Dict[str, Any]:
    """Identify a real value's closed form, airtight (NO LLM). Wraps engine.identify."""
    import mpmath
    from .engine import identify
    from .identify import DEFAULT_BASIS
    b = tuple(basis) if basis else DEFAULT_BASIS
    with mpmath.workdps(60):
        v = mpmath.mpf(str(value))  # str preserves any extra precision given
        res = identify(v, basis=b)
    out: Dict[str, Any] = {
        "query": res.query,
        "identified": res.identified,
        "basis": list(b),
        "candidates": [
            {"expr": c.expr, "display": c.display,
             "digits_agreed": c.verify.digits_agreed,
             "provenance": c.provenance.novelty.value}
            for c in res.candidates
        ],
        "note": ("Airtight: search-low / verify-high / independent library "
                 "(sympy re-eval). Pass many digits (PSLQ needs >16). NO LLM."),
    }
    if res.best is not None:
        out["best"] = {"expr": res.best.expr, "display": res.best.display,
                       "digits_agreed": res.best.verify.digits_agreed,
                       "provenance": res.best.provenance.novelty.value}
    else:
        out["best"] = None
        out["unidentified_reason"] = (
            "No closed form in the basis verified to the required digits "
            "(honest UNIDENTIFIED, not a guess).")
    return out


def tool_identify_sequence(terms: List[int], max_results: int = 5,
                           data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Identify an integer sequence against a LOCAL copy of OEIS — airtight, NO LLM.

    Hand it a list of integers; it returns the matching OEIS entries (A-number,
    name, OEIS URL) by EXACT term match against the local OEIS data — the terms
    occur (as a contiguous run, offset/subsequence-tolerant) in a stored sequence
    or they do not. No fuzzy scoring, no model, no API key. If the OEIS data files
    are not present, returns an honest UNDETERMINED note (never a fake match).
    Results rank by A-number (OEIS's canonical ordering); each match reports the
    offset where the run was found (offset 0 == your terms are a leading prefix)."""
    from .sequence import identify_sequence
    res = identify_sequence(terms, max_results=int(max_results), data_dir=data_dir)
    return {
        "query": res.query,
        "identified": res.identified,
        "matches": [
            {"a_number": m.a_number, "name": m.name, "url": m.url,
             "offset": m.offset, "exact_prefix": m.exact_prefix}
            for m in res.matches
        ],
        "data_dir": res.data_dir,
        "note": res.note,
    }


def tool_search_existing_math(query: str, k: int = 10,
                              corpus_dir: Optional[str] = None,
                              corpus_limit: int = 5000) -> Dict[str, Any]:
    """Search EXISTING math for candidate results (NO LLM). Wraps HybridRetriever.

    Resolution: an explicit ``corpus_dir`` (dataset parquets) wins; else a
    prebuilt index (``MATHLAS_INDEX`` env or the default build location) is
    served if present; else a small built-in seed corpus (zero GPU/downloads).
    Returns ranked candidates for the calling AI to reason over."""
    retr = _build_retriever(corpus_dir, corpus_limit)
    cands = retr.retrieve(query, k=int(k))
    served_index = getattr(retr, "index_path", None)
    if corpus_dir:
        corpus_label = corpus_dir
    elif served_index:
        corpus_label = f"<prebuilt index: {served_index}>"
    else:
        corpus_label = "<built-in seed corpus>"

    base = [
        {"rank": i + 1, "name": c.name, "statement": c.statement,
         "source": c.source, "score": c.score,
         "slogan": (c.meta or {}).get("slogan"),
         "title": (c.meta or {}).get("title"),
         "citations": (c.meta or {}).get("citations"),
         "category": (c.meta or {}).get("category"),
         "provenance": (c.meta or {}).get("provenance")}
        for i, c in enumerate(cands)
    ]

    # Fuse in the LIVE (web-added) corpus via its pure-BM25 channel — NO model
    # load. Findings interleave by rank-fusion so a web_added result can surface
    # above weak corpus hits. Counted/labelled so the AI knows it is AI-sourced.
    n_findings_used = _merge_live_findings(query, base, int(k))

    return {
        "query": query,
        "corpus": corpus_label,
        "k": int(k),
        "live_findings_merged": n_findings_used,
        "candidates": base[:int(k)],
        "next": ("For a promising candidate, call mapping_scaffold(problem, "
                 "candidate.statement) and applicability_checklist(candidate."
                 "statement); YOU (the AI) judge whether it applies. A candidate "
                 "with provenance 'web_added' is AI-sourced — verify it."),
        "note": ("Hybrid dense+BM25+RRF over OUR OWN index, plus any web_added "
                 "live-corpus findings (BM25, no model load). NO LLM. " +
                 ("Serving a prebuilt index (precomputed dense matrix + BM25)."
                  if served_index else
                  "Default embedder is the zero-download hashing fallback; the "
                  "production Qwen3 index is an offline-GPU build (point "
                  "MATHLAS_INDEX at index.npz to serve it).")),
    }


def _merge_live_findings(query: str, base: List[Dict[str, Any]], k: int) -> int:
    """RRF-merge live (web_added) findings into ``base`` IN PLACE. Returns how many
    distinct findings were merged. Pure BM25 over the findings sidecar — loads NO
    embedding model (the whole point of add_finding). De-dups by statement so a
    finding already present in the corpus is not double-counted."""
    try:
        from .webaug import search_findings
    except Exception:
        return 0
    finds = search_findings(query, k=max(k, 10))
    if not finds:
        return 0
    # RRF: combine the corpus ranking (base order) with the findings ranking.
    rrf_k = 60
    scored: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def _key(stmt: str) -> str:
        return (stmt or "").strip().lower()[:200]

    for rank, c in enumerate(base):
        kk = _key(c.get("statement", ""))
        scored[kk] = c
        c["_rrf"] = c.get("_rrf", 0.0) + 1.0 / (rrf_k + rank + 1)
        order.append(kk)
    merged = 0
    for rank, f in enumerate(finds):
        kk = _key(f.get("statement", ""))
        if kk in scored:                       # already in corpus -> just boost
            scored[kk]["_rrf"] += 1.0 / (rrf_k + rank + 1)
            continue
        entry = {"rank": None, "name": f.get("name"),
                 "statement": f.get("statement"), "source": f.get("source"),
                 "score": None, "slogan": f.get("slogan"), "title": None,
                 "citations": None, "category": None,
                 "provenance": f.get("provenance", "web_added"),
                 "_rrf": 1.0 / (rrf_k + rank + 1)}
        scored[kk] = entry
        order.append(kk)
        merged += 1
    # re-rank everything by fused score, rewrite base in place.
    ranked = sorted((scored[kk] for kk in dict.fromkeys(order)),
                    key=lambda c: c.get("_rrf", 0.0), reverse=True)
    base.clear()
    for i, c in enumerate(ranked):
        c.pop("_rrf", None)
        c["rank"] = i + 1
        base.append(c)
    return merged


def tool_verify_numeric(value: str, closed_form: str,
                        dps_verify: int = 50, min_digits: int = 20) -> Dict[str, Any]:
    """Airtight digit-agreement check (NO LLM). Wraps verify.verify_closed_form.

    Re-evaluates ``closed_form`` with sympy at high precision and compares to
    ``value``. This is the airtight tier: a real independent re-evaluation, not
    an opinion."""
    import mpmath
    from .verify import verify_closed_form
    with mpmath.workdps(max(dps_verify + 10, 60)):
        v = mpmath.mpf(str(value))
        vr = verify_closed_form(v, closed_form, dps_verify=dps_verify,
                                min_digits=min_digits)
    return {
        "value": mpmath.nstr(v, 15),
        "closed_form": closed_form,
        "verified": vr.ok,
        "digits_agreed": vr.digits_agreed,
        "min_digits_required": min_digits,
        "reeval": vr.reeval,
        "error": vr.error,
        "note": ("Airtight independent re-evaluation (sympy, higher precision "
                 "than any search). NO LLM. 'verified' true iff digit agreement "
                 ">= min_digits."),
    }


def tool_verify_formal(statement: str, lean: Optional[str] = None) -> Dict[str, Any]:
    """Formal (Lean) verify: REALLY runs the Lean kernel on a snippet (NO LLM).

    If a Lean toolchain is installed and a ``lean`` snippet is given, this runs the
    actual Lean type-checker and reports whether it TYPECHECKS (a real kernel
    check). With no snippet, or no Lean toolchain, it returns an HONEST
    UNDETERMINED verdict — never a fake pass. Honest caveat: a typecheck proves the
    snippet is well-typed and its proof term checks, NOT that the stated theorem is
    the right applicability claim for ``statement`` (typecheck != proves-it-applies;
    that mapping is the calling AI's job)."""
    from .verify_apply import verify_formal, find_lean
    lean_exe = find_lean()
    verdict = verify_formal(lean, lean_exe=lean_exe)
    cond = verdict.conditions[0] if verdict.conditions else None
    # "checked" == we actually ran Lean and got a definite True/False typecheck.
    really_checked = bool(lean) and bool(lean_exe) and (
        cond is not None and cond.satisfied is not None)
    return {
        "statement": statement,
        "lean_provided": bool(lean),
        "lean_available": bool(lean_exe),
        "tier": verdict.tier.value,
        "typechecks": (cond.satisfied if cond is not None else None),
        "applies": verdict.applies,
        "confidence": verdict.confidence,
        "checked": really_checked,
        "stub": not really_checked,
        "detail": (cond.evidence if cond is not None else ""),
        "failure": verdict.failure,
        "note": verdict.note,
    }


def tool_applicability_checklist(candidate_statement: str) -> Dict[str, Any]:
    """Decompose a candidate result into an applicability checklist (NO LLM).

    Returns the result's atomic preconditions + its conclusion, structured for
    the calling AI to check against its own problem. mathlas supplies the
    scaffold; the AI does the judging."""
    from .verify_apply import applicability_checklist
    cl = applicability_checklist(candidate_statement)
    return {
        "statement": cl.statement,
        "preconditions": [
            {"text": p, "satisfied": None, "evidence": ""} for p in cl.preconditions
        ],
        "conclusion": cl.conclusion,
        "instructions": cl.instructions,
        "note": cl.note,
    }


def tool_mapping_scaffold(problem: str, candidate_statement: str) -> Dict[str, Any]:
    """Build the needs<->guarantees scaffold for the AI (NO LLM).

    Returns the structured questions + a fill-in template the calling AI uses to
    decide whether the candidate applies to the problem and how. The analogy
    reasoning is the AI's job; mathlas provides the structure."""
    from .map import mapping_scaffold
    sc = mapping_scaffold(problem, candidate_statement)
    return {
        "problem": sc.problem,
        "candidate_statement": sc.candidate_statement,
        "signature": sc.signature,
        "checklist": sc.checklist,
        "questions": sc.questions,
        "answer_template": sc.answer_template,
        "note": sc.note,
    }


# --------------------------------------------------------------------------- #
# DISCOVERY + WEB-AUGMENTATION tools (NEW). All NO-LLM; each returns DATA for the
# calling AI. ramanujan.py (conjecture), funsearch.py (program search harness),
# webaug.py (search_directive + add_finding).
# --------------------------------------------------------------------------- #
def tool_conjecture_relation(value: str, max_terms: int = 16,
                             cf_depth: int = 200,
                             min_digits: int = 25) -> Dict[str, Any]:
    """RAMANUJAN MACHINE: conjecture EXISTING relations for a real constant (NO LLM).

    Beyond identify_constant's flat basis: (a) PSLQ over a RICHER basis (powers,
    pairwise products of known constants, log/exp/zeta values) -> integer-relation
    closed forms; (b) a Ramanujan-Machine continued-fraction / polynomial-recurrence
    conjecture (search small integer polys p(n),q(n) whose generalized CF matches
    the constant); plus the SIMPLE continued fraction + any recognised pattern.
    EVERY candidate is numerically VERIFIED to high precision before it is
    reported. Provenance is 'conjectured_relation' — a numerically-verified
    CONJECTURE, NOT a proof (take it to verify_formal / a human / the literature).
    Pass MANY digits of the constant (PSLQ/CF need >16). Cites the Ramanujan
    Machine (Raayoni et al., Nature 2021) + PSLQ (Ferguson-Bailey-Arno)."""
    import mpmath
    from .ramanujan import conjecture
    with mpmath.workdps(max(int(min_digits) + 40, 80)):
        res = conjecture(str(value), max_terms=int(max_terms),
                         cf_depth=int(cf_depth), min_digits=int(min_digits))
    relations = [
        {"kind": r.kind, "closed_form": r.expr,
         "integer_relation_coeffs": list(r.coeffs), "basis": list(r.basis),
         "digits_verified": r.verify.digits_agreed,
         "reeval": r.verify.reeval, "provenance": r.provenance.novelty.value,
         "method": r.provenance.method}
        for r in res.relations
    ]
    cfs = [
        {"kind": c.kind,
         "a_n_poly_coeffs": list(c.poly_a), "b_n_poly_coeffs": list(c.poly_b),
         "cf_equals": c.image, "cf_value": c.cf_value,
         "digits_verified": c.digits_agreed, "depth": c.depth,
         "form": "a0 + b1/(a1 + b2/(a2 + ...)), a_n=poly_a(n), b_n=poly_b(n)",
         "provenance": c.provenance.novelty.value, "method": c.provenance.method}
        for c in res.continued_fractions
    ]
    scf = None
    if res.simple_cf is not None:
        s = res.simple_cf
        scf = {"kind": s.kind, "terms": list(s.terms), "pattern": s.pattern,
               "convergent": s.convergent, "digits_verified": s.digits_agreed,
               "provenance": s.provenance.novelty.value, "method": s.provenance.method}
    return {
        "query": res.query,
        "found": res.found,
        "integer_relations": relations,
        "continued_fractions": cfs,
        "simple_continued_fraction": scf,
        "note": ("All candidates are numerically VERIFIED conjectures (provenance "
                 "'conjectured_relation'), NOT proofs — verify_formal / a human / "
                 "the literature for a proof. NO LLM. Ramanujan Machine (Raayoni "
                 "et al., Nature 2021) + PSLQ (Ferguson-Bailey-Arno). Honest "
                 "UNIDENTIFIED if nothing verified."),
    }


def tool_funsearch_evaluate(program_src: str, problem_id: str,
                            timeout_s: float = 10.0) -> Dict[str, Any]:
    """FUNSEARCH HARNESS — score an AI-written program in a SANDBOX (NO LLM).

    The AI is the program GENERATOR; mathlas is the deterministic HARNESS. Runs
    ``program_src`` in a sandboxed subprocess (hard wall-clock timeout, network
    stubbed out, POSIX CPU/memory rlimits, throwaway cwd) against the registered
    scorer for ``problem_id`` and returns the numeric score or the error. Ship
    problems: 'cap_set' (cap set in Z_3^n), 'online_bin_packing'. FunSearch
    (Romera-Paredes et al., Nature 2024); OpenEvolve is the open prior art."""
    from .funsearch import evaluate
    r = evaluate(program_src, problem_id, timeout_s=float(timeout_s))
    return {
        "problem_id": r.problem_id,
        "ok": r.ok,
        "score": r.score,
        "behavior": list(r.behavior),
        "error": r.error,
        "timed_out": r.timed_out,
        "seconds": round(r.seconds, 3),
        "note": ("Deterministic sandboxed run (subprocess + timeout + no network "
                 "+ rlimits). NO LLM — YOU write the program; mathlas scores it. "
                 "Higher score is better. Register a good one with funsearch_register, "
                 "then funsearch_status for the few-shot to write a better variant."),
    }


def tool_funsearch_register(program_src: str, score: float,
                            problem_id: str,
                            behavior: Optional[List[Any]] = None) -> Dict[str, Any]:
    """FUNSEARCH — store a scored program in the on-disk MAP-Elites DB (NO LLM).

    Persists ``program_src`` (with its ``score``) into the island / MAP-Elites
    program database under reference/downloads/funsearch/ (gitignored) and reports
    whether it is a new best — globally and in its behaviour cell. Pass the
    ``behavior`` returned by funsearch_evaluate to land it in the right cell."""
    from .funsearch import register
    beh = tuple(behavior) if behavior else None
    r = register(program_src, float(score), problem_id, behavior=beh)
    return {
        "problem_id": r.problem_id,
        "accepted": r.accepted,
        "new_global_best": r.new_global_best,
        "score": r.score,
        "cell": r.cell,
        "global_best_score": r.global_best_score,
        "n_cells": r.n_cells,
        "n_registered": r.n_registered,
        "note": ("Stored in the MAP-Elites program DB (one elite per behaviour "
                 "cell + the global best). NO LLM. Call funsearch_status to get "
                 "the few-shot context for the next, better variant."),
    }


def tool_funsearch_status(problem_id: str, top_k: int = 3) -> Dict[str, Any]:
    """FUNSEARCH — current best program(s) + score + the FEW-SHOT the AI writes
    the next variant from (NO LLM).

    Returns the problem spec, the best program + score, the per-cell MAP-Elites
    elites, and ``few_shot_context`` — the best-shot prompt assembled as DATA for
    the calling AI to write a strictly-better program. mathlas never sends it to
    a model; the AI is the generator."""
    from .funsearch import status
    r = status(problem_id, top_k=int(top_k))
    return {
        "problem_id": r.problem_id,
        "description": r.description,
        "entry_point": r.entry_point,
        "best_score": r.best_score,
        "best_program": r.best_program,
        "n_cells": r.n_cells,
        "n_registered": r.n_registered,
        "elites": r.elites,
        "few_shot_context": r.few_shot_context,
        "starter_program": r.starter_program,
        "note": ("few_shot_context is DATA for YOU to write the next, better "
                 "program (FunSearch's best-shot prompt) — mathlas calls no LLM. "
                 "Write it, funsearch_evaluate it, funsearch_register it, repeat."),
    }


def tool_search_directive(problem: str) -> Dict[str, Any]:
    """WEB-AUGMENTED RETRIEVAL — tell the AI WHAT to search (mathlas makes NO web
    call, NO LLM).

    The local corpus is finite; the calling AI has the web. mathlas analyses the
    problem (the needs<->guarantees signature + domain heuristics) and returns
    STRUCTURED search instructions: arXiv query strings, candidate sub-fields +
    arXiv categories, named methods/inequalities/theorems to look for, and which
    OTHER mathlas tools to also run. The AI does the searching, then feeds results
    back via add_finding."""
    from .webaug import search_directive
    d = search_directive(problem)
    return {
        "problem": d.problem,
        "signature": d.signature,
        "arxiv_queries": d.arxiv_queries,
        "subfields": d.subfields,
        "arxiv_categories": d.arxiv_categories,
        "named_results": d.named_results,
        "also_try_mathlas_tools": d.mathlas_tools,
        "instructions": d.instructions,
        "note": d.note,
    }


def tool_add_finding(statement: str, slogan: str, source: str,
                     name: Optional[str] = None) -> Dict[str, Any]:
    """WEB-AUGMENTED RETRIEVAL — the AI feeds a web-found result into the LIVE
    corpus (NO embedding-model load, NO LLM).

    Appends the finding to the live corpus through the BM25 / sparse channel +
    metadata with provenance 'web_added' — it becomes retrievable IMMEDIATELY via
    search_existing_math (RRF-fused), and crucially this requires NO embedding
    model (the 8B is never loaded per finding; works on any machine). If a dense
    Qwen3 index is ALREADY loaded in-process, a dense vector is added too; else
    dense is skipped (BM25 covers it) and a batch reindex embeds the backlog
    later. A web finding is a LEAD, not a proof — still verify it."""
    from .webaug import add_finding
    r = add_finding(statement, slogan, source, name=name)
    return {
        "ok": r.ok,
        "statement": r.statement,
        "name": r.name,
        "slogan": r.slogan,
        "source": r.source,
        "provenance": r.provenance,
        "dense_added": r.dense_added,
        "n_findings": r.n_findings,
        "note": r.note,
    }


# Registry: (name, fn, description, json-schema-ish param spec) — drives both the
# FastMCP registration and the fallback server's tools/list + tools/call.
_TOOLS: List[Dict[str, Any]] = [
    {"name": "identify_constant", "fn": tool_identify_constant,
     "description": tool_identify_constant.__doc__,
     "params": {
         "value": {"type": "string", "description":
                   "the real value as a decimal string (give many digits)"},
         "basis": {"type": "array", "items": {"type": "string"},
                   "description": "optional constant basis, e.g. [\"pi\",\"e\",\"catalan\"]"},
     }, "required": ["value"]},
    {"name": "identify_sequence", "fn": tool_identify_sequence,
     "description": tool_identify_sequence.__doc__,
     "params": {
         "terms": {"type": "array", "items": {"type": "integer"},
                   "description": "the integer sequence to identify, e.g. "
                                  "[1,1,2,3,5,8,13,21] (give >= 4 terms)"},
         "max_results": {"type": "integer", "description":
                         "max OEIS matches to return (default 5)"},
     }, "required": ["terms"]},
    {"name": "search_existing_math", "fn": tool_search_existing_math,
     "description": tool_search_existing_math.__doc__,
     "params": {
         "query": {"type": "string", "description": "a problem / result description"},
         "k": {"type": "integer", "description": "number of candidates (default 10)"},
         "corpus_dir": {"type": "string", "description":
                        "optional dir of open theorem dataset parquets; omit to "
                        "use the built-in seed corpus (zero downloads)"},
     }, "required": ["query"]},
    {"name": "verify_numeric", "fn": tool_verify_numeric,
     "description": tool_verify_numeric.__doc__,
     "params": {
         "value": {"type": "string", "description": "the value as a decimal string"},
         "closed_form": {"type": "string", "description":
                         "a closed-form expression, e.g. \"pi**2/6\" or \"zeta(3)\""},
     }, "required": ["value", "closed_form"]},
    {"name": "verify_formal", "fn": tool_verify_formal,
     "description": tool_verify_formal.__doc__,
     "params": {
         "statement": {"type": "string", "description": "the statement to check"},
         "lean": {"type": "string", "description": "optional Lean snippet (stub)"},
     }, "required": ["statement"]},
    {"name": "applicability_checklist", "fn": tool_applicability_checklist,
     "description": tool_applicability_checklist.__doc__,
     "params": {
         "candidate_statement": {"type": "string", "description":
                                 "the candidate result's statement"},
     }, "required": ["candidate_statement"]},
    {"name": "mapping_scaffold", "fn": tool_mapping_scaffold,
     "description": tool_mapping_scaffold.__doc__,
     "params": {
         "problem": {"type": "string", "description": "the problem to solve"},
         "candidate_statement": {"type": "string", "description":
                                 "a candidate existing result's statement"},
     }, "required": ["problem", "candidate_statement"]},
    # --- DISCOVERY + WEB-AUGMENTATION (NEW) --- #
    {"name": "conjecture_relation", "fn": tool_conjecture_relation,
     "description": tool_conjecture_relation.__doc__,
     "params": {
         "value": {"type": "string", "description":
                   "the real constant as a decimal string (give MANY digits; "
                   "PSLQ/CF search needs >16)"},
         "max_terms": {"type": "integer", "description":
                       "max PSLQ basis vector length (default 16; cost grows fast)"},
         "cf_depth": {"type": "integer", "description":
                      "continued-fraction evaluation depth (default 200)"},
     }, "required": ["value"]},
    {"name": "funsearch_evaluate", "fn": tool_funsearch_evaluate,
     "description": tool_funsearch_evaluate.__doc__,
     "params": {
         "program_src": {"type": "string", "description":
                         "the candidate Python program source (YOU write it; it "
                         "must define the problem's entry point)"},
         "problem_id": {"type": "string", "description":
                        "the problem to score against: 'cap_set' or "
                        "'online_bin_packing'"},
         "timeout_s": {"type": "number", "description":
                       "hard wall-clock timeout in seconds (default 10)"},
     }, "required": ["program_src", "problem_id"]},
    {"name": "funsearch_register", "fn": tool_funsearch_register,
     "description": tool_funsearch_register.__doc__,
     "params": {
         "program_src": {"type": "string", "description": "the program source"},
         "score": {"type": "number", "description":
                   "the score from funsearch_evaluate"},
         "problem_id": {"type": "string", "description": "the problem id"},
         "behavior": {"type": "array", "items": {},
                      "description": "the behaviour descriptor from "
                                     "funsearch_evaluate (MAP-Elites cell)"},
     }, "required": ["program_src", "score", "problem_id"]},
    {"name": "funsearch_status", "fn": tool_funsearch_status,
     "description": tool_funsearch_status.__doc__,
     "params": {
         "problem_id": {"type": "string", "description":
                        "the problem id: 'cap_set' or 'online_bin_packing'"},
         "top_k": {"type": "integer", "description":
                   "how many elite programs to include in the few-shot (default 3)"},
     }, "required": ["problem_id"]},
    {"name": "search_directive", "fn": tool_search_directive,
     "description": tool_search_directive.__doc__,
     "params": {
         "problem": {"type": "string", "description":
                     "a problem / result description to build a web-search plan for"},
     }, "required": ["problem"]},
    {"name": "add_finding", "fn": tool_add_finding,
     "description": tool_add_finding.__doc__,
     "params": {
         "statement": {"type": "string", "description":
                       "the web-found result's statement (the real text)"},
         "slogan": {"type": "string", "description":
                    "a short natural-language denotation of it (what it says)"},
         "source": {"type": "string", "description":
                    "where it came from: a URL / arXiv id / citation"},
         "name": {"type": "string", "description":
                  "optional name/title of the result"},
     }, "required": ["statement", "slogan", "source"]},
]


def tool_names() -> List[str]:
    """Names of the exposed MCP tools (handy for tests / introspection)."""
    return [t["name"] for t in _TOOLS]


# --------------------------------------------------------------------------- #
# FastMCP server (preferred — uses the official mcp SDK if installed).
# --------------------------------------------------------------------------- #
def build_fastmcp():
    """Construct a FastMCP server with all tools registered. Raises ImportError
    if the ``mcp`` SDK is not installed (caller falls back to the stdio server)."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "mathlas",
        instructions=(
            "mathlas is a tool you (the AI) use; it never calls an LLM and needs "
            "no API key. It gives you: identify_constant (airtight closed forms), "
            "identify_sequence (airtight OEIS exact term-match for integer "
            "sequences), search_existing_math (find existing theorems), "
            "verify_numeric (airtight digit check), verify_formal (REAL Lean "
            "kernel typecheck, or honest UNDETERMINED), applicability_checklist "
            "and mapping_scaffold (structured needs<->guarantees scaffolds you "
            "reason over). PLUS a discovery + web-augmentation layer: "
            "conjecture_relation (Ramanujan-Machine PSLQ-richer-basis + "
            "continued-fraction conjectures, VERIFIED not proved), "
            "funsearch_evaluate/register/status (a sandboxed program-search "
            "harness where YOU write the programs), and search_directive + "
            "add_finding (mathlas tells you what to web-search and ingests your "
            "findings into the live corpus with no model load). Typical flow: "
            "search_existing_math -> mapping_scaffold + applicability_checklist -> "
            "you judge applicability -> verify_numeric for any numeric claim."),
    )
    # Register each tool. FastMCP introspects the wrapped fn's signature/types.
    for spec in _TOOLS:
        mcp.tool(name=spec["name"], description=(spec["description"] or "").strip())(
            spec["fn"])
    return mcp


# --------------------------------------------------------------------------- #
# Dependency-free fallback: a minimal stdio JSON-RPC MCP server.
# Implements just enough of the MCP wire protocol (initialize, tools/list,
# tools/call) to be usable with no third-party deps when ``mcp`` is absent.
# --------------------------------------------------------------------------- #
PROTOCOL_VERSION = "2025-06-18"


def _input_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": spec["params"],
            "required": spec.get("required", [])}


def _dispatch(method: str, params: Dict[str, Any]) -> Any:
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "mathlas", "version": "0.1.0"},
            "instructions": ("mathlas: a tool you use; it never calls an LLM and "
                             "needs no API key. Search existing math + airtight "
                             "verification + needs<->guarantees scaffolds."),
        }
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [
            {"name": s["name"],
             "description": (s["description"] or "").strip(),
             "inputSchema": _input_schema(s)}
            for s in _TOOLS]}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        spec = next((s for s in _TOOLS if s["name"] == name), None)
        if spec is None:
            raise ValueError(f"unknown tool: {name}")
        result = spec["fn"](**args)
        return {"content": [{"type": "text",
                             "text": json.dumps(result, indent=2, default=str)}],
                "structuredContent": result, "isError": False}
    raise ValueError(f"unknown method: {method}")


def serve_stdio() -> None:
    """Minimal line-delimited JSON-RPC MCP loop over stdio (no deps).

    Reads one JSON-RPC message per line from stdin, writes one JSON response per
    line to stdout. Notifications (no ``id``) get no reply. Used only when the
    ``mcp`` SDK is unavailable; the wire shape matches MCP stdio transport."""
    import sys
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {}) or {}
        try:
            result = _dispatch(method, params)
        except Exception as e:  # surface as a JSON-RPC error (only if a request)
            if mid is not None:
                out.write(json.dumps({"jsonrpc": "2.0", "id": mid,
                                      "error": {"code": -32603, "message": str(e)}}) + "\n")
                out.flush()
            continue
        if mid is None:
            continue  # notification: no response
        out.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\n")
        out.flush()


def main() -> int:
    """Run the mathlas MCP server over stdio. Prefers the official SDK; falls back
    to the dependency-free stdio server if ``mcp`` is not installed."""
    try:
        mcp = build_fastmcp()
    except ImportError:
        serve_stdio()
        return 0
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
