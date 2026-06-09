"""mathlas MCP server — the AI-callable toolbox. NO LLM, NO API key.

mathlas is a tool that an AI *uses*, not a tool that uses an AI. This server
exposes mathlas's capabilities over the Model Context Protocol so any MCP client
(Claude Code, Cursor, or any agent) can call them. Every tool runs with **NO LLM
and NO API key** and returns DATA for the calling AI to reason over — the AI is
the brain; mathlas provides search over existing math, airtight numeric/formal
verification, structured needs<->guarantees scaffolds, and provenance.

Tools exposed (12)
------------------
  identify_constant(value, basis?)          airtight closed-form + provenance
  identify_sequence(terms, max_results?)    airtight OEIS exact term-match
  search_existing_math(query, k, corpus_dir?) ranked candidate existing results
  search_formal_math(query, k?, backend?)   mathlib declarations via public Loogle/LeanSearch
  verify_numeric(value, closed_form)        airtight digit-agreement verdict
  verify_formal(statement, lean?)           REAL Lean kernel typecheck (or honest UNDETERMINED)
  applicability_checklist(candidate_statement) the result's preconditions, structured
  mapping_scaffold(problem, candidate_statement) the needs<->guarantees questions
  -- discovery + web-augmentation layer (NO LLM; each returns DATA) --
  conjecture_relation(value, max_terms?, cf_depth?)  Ramanujan-Machine: PSLQ-richer-basis
                                            + continued-fraction conjectures (VERIFIED, not proved)
  funsearch(action, problem_id, ...)        one tool, action = evaluate | register | status
                                            (sandbox-score / MAP-Elites store / best + few-shot;
                                            the old funsearch_evaluate/_register/_status names
                                            still dispatch on the fallback server, unlisted)
  search_directive(problem)                 a STRUCTURED web-search plan for the AI (no web call)
  add_finding(statement, slogan, source, name?)  ingest a web result into the live corpus (no model load)

Register in Claude Code (no API key needed):

    claude mcp add mathlas -- python -m mathlas.server

or run directly over stdio:  ``python -m mathlas.server``.

Index selection (environment)
-----------------------------
``search_existing_math`` serves a prebuilt index if one exists on disk; otherwise
a tiny built-in seed corpus (zero GPU/downloads). Two env vars override:

  * ``MATHLAS_SEED=1``  force the LIGHTWEIGHT seed corpus and NEVER load the
                        multi-GB prebuilt index — a fast cold start on any box
                        (wins over ``MATHLAS_INDEX``). Accepts 1/true/yes/on.
  * ``MATHLAS_INDEX=/path/index.npz``  serve this specific prebuilt index
                        (errors if missing). Ignored when ``MATHLAS_SEED`` is set.

Implementation: uses the official ``mcp`` Python SDK (FastMCP) if installed;
otherwise falls back to a dependency-free stdio JSON-RPC MCP server implementing
the same tools and the same wire protocol.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional

# --------------------------------------------------------------------------- #
# Tool implementations — plain functions, NO LLM, returning JSON-able DATA.
# These are the single source of truth; both the FastMCP and the fallback server
# call them. Each returns a dict the calling AI consumes.
# --------------------------------------------------------------------------- #

#: A tiny, self-contained seed corpus of well-known results so that
#: ``search_existing_math`` works with ZERO downloads / GPU / corpus when no
#: ``corpus_dir`` is supplied. For a real index, point ``corpus_dir`` at the open
#: theorem dataset (parquets) — see scripts/build_index.py / docs/methods.md.
SEED_CORPUS: List[Dict[str, str]] = [
    {
        "name": "Mean Value Theorem",
        "statement": "If f is continuous on [a, b] and differentiable on (a, b), "
        "then there exists c in (a, b) such that f'(c) = (f(b) - f(a)) / (b - a).",
        "source": "calculus",
    },
    {
        "name": "Intermediate Value Theorem",
        "statement": "If f is continuous on [a, b] and y lies between f(a) and f(b), "
        "then there exists c in [a, b] such that f(c) = y.",
        "source": "calculus",
    },
    {
        "name": "Banach Fixed-Point Theorem",
        "statement": "Let (X, d) be a complete metric space and T: X -> X a "
        "contraction. Then T has a unique fixed point, and the iteration "
        "x_{n+1} = T(x_n) converges to it from any start.",
        "source": "analysis",
    },
    {
        "name": "Cauchy-Schwarz Inequality",
        "statement": "For vectors u, v in an inner product space, "
        "|<u, v>| <= ||u|| * ||v||, with equality iff u and v are linearly dependent.",
        "source": "linear-algebra",
    },
    {
        "name": "Bolzano-Weierstrass Theorem",
        "statement": "Every bounded sequence in R^n has a convergent subsequence.",
        "source": "analysis",
    },
    {
        "name": "Pigeonhole Principle",
        "statement": "If n items are placed into m boxes with n > m, then at least "
        "one box contains more than one item.",
        "source": "combinatorics",
    },
    {
        "name": "Fundamental Theorem of Arithmetic",
        "statement": "Every integer greater than 1 is either prime or a product of "
        "primes, unique up to the order of the factors.",
        "source": "number-theory",
    },
    {
        "name": "Lagrange's Theorem (group theory)",
        "statement": "If G is a finite group and H is a subgroup of G, then the "
        "order of H divides the order of G.",
        "source": "algebra",
    },
    {
        "name": "Cayley-Hamilton Theorem",
        "statement": "Every square matrix over a commutative ring satisfies its own "
        "characteristic equation.",
        "source": "linear-algebra",
    },
    {
        "name": "Heine-Borel Theorem",
        "statement": "A subset of R^n is compact if and only if it is closed and bounded.",
        "source": "topology",
    },
    {
        "name": "Brouwer Fixed-Point Theorem",
        "statement": "Every continuous function from a convex compact subset of R^n "
        "to itself has a fixed point.",
        "source": "topology",
    },
    {
        "name": "Basel problem (Euler)",
        "statement": "The sum over n >= 1 of 1/n^2 equals pi^2 / 6.",
        "source": "analysis",
    },
]

# Cache built retrievers by corpus key so we do not re-index per call (data-flow
# discipline: every per-call build must be cached).
_RETRIEVER_CACHE: Dict[str, Any] = {}


#: Default location of a prebuilt index (the offline Qwen3-Embedding-8B build).
#: Overridable via the ``MATHLAS_INDEX`` env var. If present, ``search_existing_math``
#: serves it (precomputed dense matrix + BM25) instead of the tiny seed corpus.
def _default_index() -> str:
    """Prefer the merged 1,635,233-doc exact dense union index (base 1.341M +
    dolma 294K + Stacks + ProofWiki) if built; else fall back to ``index.npz``."""
    d = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reference",
        "downloads",
    )
    full = os.path.join(d, "index_full_dense.npz")
    return full if os.path.exists(full) else os.path.join(d, "index.npz")


_DEFAULT_INDEX = _default_index()


def _seed_forced() -> bool:
    """``MATHLAS_SEED`` truthy => force the tiny built-in seed corpus and SKIP the
    heavy prebuilt index entirely (a fast, lightweight cold start on any box).
    Accepts 1/true/yes/on (case-insensitive); 0/empty/unset = normal resolution."""
    return os.environ.get("MATHLAS_SEED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_index_path() -> Optional[str]:
    """The prebuilt-index path to serve, or ``None`` (=> built-in seed corpus).

    ``MATHLAS_SEED`` wins over everything: if set it forces the seed corpus and
    NEVER touches the (multi-GB) prebuilt index — use it for a lightweight cold
    start. Otherwise ``MATHLAS_INDEX`` wins; else the default build location is
    used IF it exists. ``MATHLAS_INDEX`` set but missing is an explicit opt-in
    error (don't silently fall back and confuse)."""
    if _seed_forced():
        return None
    env = os.environ.get("MATHLAS_INDEX")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(
                f"MATHLAS_INDEX={env} does not exist (set it to a built "
                f"index.npz, or unset it to use the seed corpus)."
            )
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
            index_path, embedder=_embedder_for_index(index_path)
        )
        _RETRIEVER_CACHE[key] = retr
        return retr

    key = "<seed>"
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]
    docs = [
        Document(
            doc_id=str(i),
            slogan=d["statement"],
            statement=d["statement"],
            name=d["name"],
            source=d["source"],
        )
        for i, d in enumerate(SEED_CORPUS)
    ]
    retr = HybridRetriever(docs)  # default HashingEmbedder -> no model download
    _RETRIEVER_CACHE[key] = retr
    return retr


def tool_identify_constant(
    value: str, basis: Optional[List[str]] = None
) -> Dict[str, Any]:
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
            {
                "expr": c.expr,
                "display": c.display,
                "digits_agreed": c.verify.digits_agreed,
                "provenance": c.provenance.novelty.value,
            }
            for c in res.candidates
        ],
        "note": (
            "Airtight: search-low / verify-high / independent library "
            "(sympy re-eval). Pass many digits (PSLQ needs >16). NO LLM."
        ),
    }
    if res.best is not None:
        out["best"] = {
            "expr": res.best.expr,
            "display": res.best.display,
            "digits_agreed": res.best.verify.digits_agreed,
            "provenance": res.best.provenance.novelty.value,
        }
    else:
        out["best"] = None
        out["unidentified_reason"] = (
            "No closed form in the basis verified to the required digits "
            "(honest UNIDENTIFIED, not a guess)."
        )
    return out


def tool_identify_sequence(
    terms: List[int], max_results: int = 5, data_dir: Optional[str] = None
) -> Dict[str, Any]:
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
    out = {
        "query": res.query,
        "identified": res.identified,
        "matches": [
            {
                "a_number": m.a_number,
                "name": m.name,
                "url": m.url,
                "offset": m.offset,
                "exact_prefix": m.exact_prefix,
            }
            for m in res.matches
        ],
        "data_dir": res.data_dir,
        "note": res.note,
    }
    if not res.identified and "not available" in (res.note or ""):
        out["remediation"] = (
            "Local OEIS data is missing — download https://oeis.org/stripped.gz "
            "and https://oeis.org/names.gz into reference/downloads/oeis/ (or pass "
            "data_dir pointing at a directory containing them), then retry."
        )
    return out


def tool_search_existing_math(
    query: str, k: int = 10, corpus_dir: Optional[str] = None, corpus_limit: int = 5000
) -> Dict[str, Any]:
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
        {
            "rank": i + 1,
            "name": c.name,
            "statement": c.statement,
            "source": c.source,
            "score": c.score,
            "slogan": (c.meta or {}).get("slogan"),
            "title": (c.meta or {}).get("title"),
            "citations": (c.meta or {}).get("citations"),
            "category": (c.meta or {}).get("category"),
            "provenance": (c.meta or {}).get("provenance"),
        }
        for i, c in enumerate(cands)
    ]

    # Fuse in the LIVE (web-added) corpus via its pure-BM25 channel — NO model
    # load. Findings interleave by rank-fusion so a web_added result can surface
    # above weak corpus hits. Counted/labelled so the AI knows it is AI-sourced.
    n_findings_used = _merge_live_findings(query, base, int(k), retr=retr)

    return {
        "query": query,
        "corpus": corpus_label,
        "k": int(k),
        "live_findings_merged": n_findings_used,
        "candidates": base[: int(k)],
        "next": (
            "For a promising candidate, call mapping_scaffold(problem, "
            "candidate.statement) and applicability_checklist(candidate."
            "statement); YOU (the AI) judge whether it applies. A candidate "
            "with provenance 'web_added' is AI-sourced — verify it."
        ),
        "note": (
            "Hybrid dense+BM25+RRF over OUR OWN index, plus any web_added "
            "live-corpus findings (BM25, no model load). NO LLM. "
            + (
                "Serving a prebuilt index (precomputed dense matrix + BM25)."
                if served_index
                else "Default embedder is the zero-download hashing fallback; the "
                "production Qwen3 index is an offline-GPU build (point "
                "MATHLAS_INDEX at index.npz to serve it)."
            )
        ),
    }


def _dense_finding_ranking(query: str, retr) -> List[Dict[str, Any]]:
    """Rank live findings that carry a stored ``dense_vec`` by COSINE against the
    query vector, best-first. Reuses the SERVED retriever's already-loaded embedder
    to embed the query (no new model load) and dot-products the stored unit-norm
    finding vectors. Returns ``[]`` unless there is a real dense space to score in
    (a finding's dense_vec must match the served embedder dim — same space).

    This is the channel that makes a caller-supplied dense finding reachable when
    its wording differs from the query and BM25 misses it (dense's whole purpose).
    """
    emb = getattr(retr, "embedder", None)
    if emb is None or not getattr(emb, "dim", None):
        return []
    try:
        from .webaug import dense_findings
        import numpy as np
    except Exception:
        return []
    dfs = dense_findings()
    if not dfs:
        return []
    dim = int(emb.dim)
    rows, recs = [], []
    for r in dfs:
        v = r.get("dense_vec")
        if v is not None and len(v) == dim:  # only same-space vectors
            rows.append(np.asarray(v, dtype=np.float32))
            recs.append(r)
    if not recs:
        return []
    mat = np.vstack(rows)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    mat = mat / n  # cosine via dot on unit rows
    q = emb.encode([query], is_query=True)[0].astype(np.float32)
    qn = np.linalg.norm(q)
    if qn:
        q = q / qn
    sims = mat @ q
    order = np.argsort(-sims)
    return [recs[i] for i in order]


def _merge_live_findings(
    query: str, base: List[Dict[str, Any]], k: int, retr=None
) -> int:
    """RRF-merge live (web_added) findings into ``base`` IN PLACE. Returns how many
    distinct findings were merged. Loads NO embedding model: BM25 over the findings
    sidecar ALWAYS, plus — when the served index has a real dense space — a DENSE
    channel that scores each finding's caller-supplied ``dense_vec`` against the
    query vector by cosine, RRF-fused alongside the BM25 finding-rank (the SAME
    dense+BM25 treatment native docs get). De-dups by statement so a finding
    already in the corpus is not double-counted."""
    try:
        from .webaug import search_findings
    except Exception:
        return 0
    finds = search_findings(query, k=max(k, 10))
    dense_ranked = _dense_finding_ranking(query, retr) if retr is not None else []
    if not finds and not dense_ranked:
        return 0
    # RRF: combine the corpus ranking (base order) with the findings BM25 ranking
    # AND the findings dense ranking — three rankings, one fused score.
    rrf_k = 60
    scored: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def _key(stmt: str) -> str:
        return (stmt or "").strip().lower()[:200]

    def _entry(f: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rank": None,
            "name": f.get("name"),
            "statement": f.get("statement"),
            "source": f.get("source"),
            "score": None,
            "slogan": f.get("slogan"),
            "title": None,
            "citations": None,
            "category": None,
            "provenance": f.get("provenance", "web_added"),
            "_rrf": 0.0,
        }

    for rank, c in enumerate(base):
        kk = _key(c.get("statement", ""))
        scored[kk] = c
        c["_rrf"] = c.get("_rrf", 0.0) + 1.0 / (rrf_k + rank + 1)
        order.append(kk)
    merged = 0
    for rank, f in enumerate(finds):
        kk = _key(f.get("statement", ""))
        if kk not in scored:
            scored[kk] = _entry(f)
            order.append(kk)
            merged += 1
        scored[kk]["_rrf"] += 1.0 / (rrf_k + rank + 1)  # BM25 finding-rank
    for rank, f in enumerate(dense_ranked):
        kk = _key(f.get("statement", ""))
        if kk not in scored:
            scored[kk] = _entry(f)
            order.append(kk)
            merged += 1
        scored[kk]["_rrf"] += 1.0 / (rrf_k + rank + 1)  # DENSE finding-rank
    # re-rank everything by fused score, rewrite base in place.
    ranked = sorted(
        (scored[kk] for kk in dict.fromkeys(order)),
        key=lambda c: c.get("_rrf", 0.0),
        reverse=True,
    )
    base.clear()
    for i, c in enumerate(ranked):
        c.pop("_rrf", None)
        c["rank"] = i + 1
        base.append(c)
    return merged


def tool_search_formal_math(
    query: str, k: int = 10, backend: str = "auto"
) -> Dict[str, Any]:
    """Search FORMAL math — mathlib declarations via the public Loogle and
    LeanSearch services (NO LLM, no API key).

    The one tool that makes a web call itself. ``backend='loogle'`` for
    pattern/type queries (``?a * ?b = ?b * ?a``), ``'leansearch'`` for
    natural-language queries, ``'auto'`` for both (interleaved + deduped).
    Hits are provenance-labeled ``external:<service>``; if a service is down the
    tool says so honestly (``backends[<name>].available: false``) instead of
    fabricating hits. Composes with verify_formal: find the declaration here,
    then kernel-check the snippet you write."""
    from .formal_search import search_formal_math

    return search_formal_math(query, k=int(k), backend=backend)


def tool_verify_numeric(
    value: str, closed_form: str, dps_verify: int = 50, min_digits: int = 20
) -> Dict[str, Any]:
    """Airtight digit-agreement check (NO LLM). Wraps verify.verify_closed_form.

    Re-evaluates ``closed_form`` with sympy at high precision and compares to
    ``value``. This is the airtight tier: a real independent re-evaluation, not
    an opinion."""
    import mpmath
    from .verify import verify_closed_form

    with mpmath.workdps(max(dps_verify + 10, 60)):
        v = mpmath.mpf(str(value))
        vr = verify_closed_form(
            v, closed_form, dps_verify=dps_verify, min_digits=min_digits
        )
    return {
        "value": mpmath.nstr(v, 15),
        "closed_form": closed_form,
        "verified": vr.ok,
        "digits_agreed": vr.digits_agreed,
        "min_digits_required": min_digits,
        "reeval": vr.reeval,
        "error": vr.error,
        "note": (
            "Airtight independent re-evaluation (sympy, higher precision "
            "than any search). NO LLM. 'verified' true iff digit agreement "
            ">= min_digits."
        ),
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
    really_checked = (
        bool(lean)
        and bool(lean_exe)
        and (cond is not None and cond.satisfied is not None)
    )
    out = {
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
    if not really_checked:  # agent-actionable: say exactly what unblocks a real check
        if not lean:
            out["remediation"] = (
                "No `lean` snippet was provided — statement text alone cannot be "
                "kernel-checked. Write a Lean 4 snippet (e.g. "
                "`example : 2 + 2 = 4 := rfl`) and pass it as `lean`. "
                "search_formal_math can find the mathlib declaration names to use."
            )
        elif not lean_exe:
            out["remediation"] = (
                "No Lean toolchain found — install elan "
                "(`curl -sSf https://elan.lean-lang.org/elan-init.sh | sh`) or set "
                "the LEAN env var to a `lean` binary, then retry."
            )
    return out


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
def tool_conjecture_relation(
    value: str, max_terms: int = 16, cf_depth: int = 200, min_digits: int = 25
) -> Dict[str, Any]:
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
        res = conjecture(
            str(value),
            max_terms=int(max_terms),
            cf_depth=int(cf_depth),
            min_digits=int(min_digits),
        )
    relations = [
        {
            "kind": r.kind,
            "closed_form": r.expr,
            "integer_relation_coeffs": list(r.coeffs),
            "basis": list(r.basis),
            "digits_verified": r.verify.digits_agreed,
            "reeval": r.verify.reeval,
            "provenance": r.provenance.novelty.value,
            "method": r.provenance.method,
        }
        for r in res.relations
    ]
    cfs = [
        {
            "kind": c.kind,
            "a_n_poly_coeffs": list(c.poly_a),
            "b_n_poly_coeffs": list(c.poly_b),
            "cf_equals": c.image,
            "cf_value": c.cf_value,
            "digits_verified": c.digits_agreed,
            "depth": c.depth,
            "form": "a0 + b1/(a1 + b2/(a2 + ...)), a_n=poly_a(n), b_n=poly_b(n)",
            "provenance": c.provenance.novelty.value,
            "method": c.provenance.method,
        }
        for c in res.continued_fractions
    ]
    scf = None
    if res.simple_cf is not None:
        s = res.simple_cf
        scf = {
            "kind": s.kind,
            "terms": list(s.terms),
            "pattern": s.pattern,
            "convergent": s.convergent,
            "digits_verified": s.digits_agreed,
            "provenance": s.provenance.novelty.value,
            "method": s.provenance.method,
        }
    return {
        "query": res.query,
        "found": res.found,
        "integer_relations": relations,
        "continued_fractions": cfs,
        "simple_continued_fraction": scf,
        "note": (
            "All candidates are numerically VERIFIED conjectures (provenance "
            "'conjectured_relation'), NOT proofs — verify_formal / a human / "
            "the literature for a proof. NO LLM. Ramanujan Machine (Raayoni "
            "et al., Nature 2021) + PSLQ (Ferguson-Bailey-Arno). Honest "
            "UNIDENTIFIED if nothing verified."
        ),
    }


def tool_funsearch_evaluate(
    program_src: str, problem_id: str, timeout_s: float = 10.0
) -> Dict[str, Any]:
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
        "note": (
            "Deterministic sandboxed run (subprocess + timeout + no network "
            "+ rlimits). NO LLM — YOU write the program; mathlas scores it. "
            "Higher score is better. Register a good one with funsearch_register, "
            "then funsearch_status for the few-shot to write a better variant."
        ),
    }


def tool_funsearch_register(
    program_src: str,
    score: float,
    problem_id: str,
    behavior: Optional[List[Any]] = None,
) -> Dict[str, Any]:
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
        "note": (
            "Stored in the MAP-Elites program DB (one elite per behaviour "
            "cell + the global best). NO LLM. Call funsearch_status to get "
            "the few-shot context for the next, better variant."
        ),
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
        "note": (
            "few_shot_context is DATA for YOU to write the next, better "
            "program (FunSearch's best-shot prompt) — mathlas calls no LLM. "
            "Write it, funsearch_evaluate it, funsearch_register it, repeat."
        ),
    }


def tool_funsearch(
    action: Literal["evaluate", "register", "status"],
    problem_id: str,
    program_src: Optional[str] = None,
    score: Optional[float] = None,
    behavior: Optional[List[Any]] = None,
    timeout_s: float = 10.0,
    top_k: int = 3,
) -> Dict[str, Any]:
    """FUNSEARCH harness — one tool, three actions (NO LLM; YOU write the programs).

    ``action='evaluate'`` sandbox-scores ``program_src`` against ``problem_id``
    ('cap_set' or 'online_bin_packing'); ``action='register'`` stores a scored
    program (pass ``program_src`` + ``score`` + the ``behavior`` evaluate
    returned) in the on-disk MAP-Elites DB; ``action='status'`` returns the best
    program(s) + the few-shot context to write the next, better variant.
    Loop: status -> you write a program -> evaluate -> register -> repeat."""
    act = (action or "").strip().lower()
    if act == "evaluate":
        if not program_src:
            return {
                "ok": False,
                "action": act,
                "error": "action='evaluate' requires program_src (the candidate Python "
                "program defining the problem's entry point). Call "
                "funsearch(action='status', problem_id=...) first to get the "
                "spec + starter program.",
            }
        return dict(
            tool_funsearch_evaluate(
                program_src, problem_id, timeout_s=float(timeout_s)
            ),
            action=act,
        )
    if act == "register":
        if not program_src or score is None:
            return {
                "ok": False,
                "action": act,
                "error": "action='register' requires program_src AND score (use the "
                "score + behavior that funsearch(action='evaluate', ...) "
                "returned).",
            }
        return dict(
            tool_funsearch_register(
                program_src, float(score), problem_id, behavior=behavior
            ),
            action=act,
        )
    if act == "status":
        return dict(tool_funsearch_status(problem_id, top_k=int(top_k)), action=act)
    return {
        "ok": False,
        "action": action,
        "error": f"unknown action {action!r} — use 'evaluate' (sandbox-score a "
        "program), 'register' (store a scored program), or 'status' (best "
        "programs + few-shot context).",
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


def tool_add_finding(
    statement: str,
    slogan: str,
    source: str,
    name: Optional[str] = None,
    dense_vec: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """WEB-AUGMENTED RETRIEVAL — the AI feeds a web-found result into the LIVE
    corpus (NO embedding-model load, NO LLM).

    Appends the finding to the live corpus through the BM25 / sparse channel +
    metadata with provenance 'web_added' — it becomes retrievable IMMEDIATELY via
    search_existing_math (RRF-fused), and crucially this requires NO embedding
    model (the 8B is never loaded per finding; works on any machine).

    DENSE (the self-augmenting corpus): pass ``dense_vec`` — YOU (the AI) embed the
    finding's slogan with the SAME model the served index was built with and hand
    mathlas the vector. mathlas stores it and the finding then gets the SAME
    dense+BM25 retrieval native docs get (cosine vs the query vector, RRF-fused),
    so it's found even when its wording differs from the query — with NO model load
    here. The vector's length must equal the served index dim (else an honest error
    is returned, finding NOT added). Omit dense_vec for BM25-only (backward
    compatible). A web finding is a LEAD, not a proof — still verify it."""
    from .webaug import add_finding

    r = add_finding(statement, slogan, source, name=name, dense_vec=dense_vec)
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


# Registry: (name, title, fn, description, params, output schema) — drives both
# the FastMCP registration and the fallback server's tools/list + tools/call.
# Descriptions are deliberately CRISP (when-to-use + args), not the long
# docstrings: MCP clients show these to the model on every turn.
_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "identify_constant",
        "title": "Identify constant (closed form)",
        "fn": tool_identify_constant,
        "description": (
            "Identify a real number's closed form, airtight: PSLQ + closed-form "
            "search, every candidate independently re-evaluated to 50+ digits, "
            "honest UNIDENTIFIED otherwise. Use when you have a numeric constant "
            "and want to know what it IS. Args: value (decimal string — give MANY "
            "digits, >16), optional basis (constant names like ['pi','e'])."
        ),
        "params": {
            "value": {
                "type": "string",
                "description": "the real value as a decimal string (give many digits, >16)",
            },
            "basis": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'optional constant basis, e.g. ["pi","e","catalan"]',
            },
        },
        "required": ["value"],
        "output": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "identified": {"type": "boolean"},
                "best": {
                    "type": ["object", "null"],
                    "description": "best verified candidate {expr, display, digits_agreed, "
                    "provenance}, or null if honest UNIDENTIFIED",
                },
                "candidates": {"type": "array", "items": {"type": "object"}},
                "basis": {"type": "array", "items": {"type": "string"}},
                "unidentified_reason": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["query", "identified", "candidates", "best"],
        },
    },
    {
        "name": "identify_sequence",
        "title": "Identify integer sequence (OEIS)",
        "fn": tool_identify_sequence,
        "description": (
            "Match an integer sequence against a LOCAL OEIS copy by EXACT "
            "contiguous term-match (no fuzzy scoring; honest UNDETERMINED if the "
            "data files are absent). Use when you have >= 4 integer terms and want "
            "the named sequence. Args: terms (list of integers), max_results "
            "(default 5)."
        ),
        "params": {
            "terms": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "the integer sequence to identify, e.g. "
                "[1,1,2,3,5,8,13,21] (give >= 4 terms)",
            },
            "max_results": {
                "type": "integer",
                "description": "max OEIS matches to return (default 5)",
            },
        },
        "required": ["terms"],
        "output": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "identified": {"type": "boolean"},
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "a_number": {"type": "string"},
                            "name": {"type": "string"},
                            "url": {"type": "string"},
                            "offset": {"type": "integer"},
                            "exact_prefix": {"type": "boolean"},
                        },
                    },
                },
                "data_dir": {"type": ["string", "null"]},
                "note": {"type": "string"},
                "remediation": {
                    "type": "string",
                    "description": "present iff local OEIS data is missing — how to get it",
                },
            },
            "required": ["query", "identified", "matches"],
        },
    },
    {
        "name": "search_existing_math",
        "title": "Search existing math (mathlas index)",
        "fn": tool_search_existing_math,
        "description": (
            "Find existing theorems/results for a problem from the mathlas "
            "3.68M-doc index (dense + BM25 + RRF, fused with any live web_added "
            "findings). Use FIRST for any 'does known math solve this?' question; "
            "follow up with applicability_checklist on promising candidates. Args: "
            "query (problem/result description), k (default 10), optional "
            "corpus_dir (dataset parquets; omit to serve the prebuilt index or "
            "seed corpus)."
        ),
        "params": {
            "query": {
                "type": "string",
                "description": "a problem / result description",
            },
            "k": {
                "type": "integer",
                "description": "number of candidates (default 10)",
            },
            "corpus_dir": {
                "type": "string",
                "description": "optional dir of open theorem dataset parquets; omit to "
                "use the served index / built-in seed corpus",
            },
        },
        "required": ["query"],
        "output": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "corpus": {"type": "string", "description": "what was actually served"},
                "k": {"type": "integer"},
                "live_findings_merged": {"type": "integer"},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer"},
                            "name": {"type": ["string", "null"]},
                            "statement": {"type": ["string", "null"]},
                            "source": {"type": ["string", "null"]},
                            "provenance": {
                                "type": ["string", "null"],
                                "description": "'web_added' = AI-sourced live finding — verify it",
                            },
                        },
                    },
                },
                "next": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["query", "candidates"],
        },
    },
    {
        "name": "search_formal_math",
        "title": "Search formal math (Loogle/LeanSearch)",
        "fn": tool_search_formal_math,
        "description": (
            "Find mathlib DECLARATIONS (name + type) via the public Loogle "
            "(pattern/type queries like '?a * ?b = ?b * ?a') and LeanSearch "
            "(natural-language queries) services — the ONE tool that itself calls "
            "the web; honest 'service unavailable' if down. Use when you need the "
            "formal Lean name/type of a result, e.g. before writing a "
            "verify_formal snippet. Args: query, k (default 10), backend "
            "('auto'|'loogle'|'leansearch')."
        ),
        "params": {
            "query": {
                "type": "string",
                "description": "natural language (leansearch) or a Loogle pattern/type query",
            },
            "k": {"type": "integer", "description": "max merged hits (default 10)"},
            "backend": {
                "type": "string",
                "enum": ["auto", "loogle", "leansearch"],
                "description": "'loogle' = pattern/type, 'leansearch' = "
                "natural language, 'auto' = both (default)",
            },
        },
        "required": ["query"],
        "output": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "backend": {"type": "string"},
                "backends": {
                    "type": "object",
                    "description": "per-service block {available, hits, error}",
                },
                "hits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": ["string", "null"]},
                            "type": {"type": ["string", "null"]},
                            "module": {"type": ["string", "null"]},
                            "doc": {"type": ["string", "null"]},
                            "url": {"type": ["string", "null"]},
                            "source": {"type": "string"},
                            "provenance": {"type": "string"},
                        },
                    },
                },
                "note": {"type": "string"},
            },
            "required": ["query", "hits"],
        },
    },
    {
        "name": "verify_numeric",
        "title": "Verify numeric claim (airtight)",
        "fn": tool_verify_numeric,
        "description": (
            "Airtight check that a closed-form expression equals a numeric value: "
            "independent sympy re-evaluation at higher precision, verified only on "
            ">= 20 agreeing digits. Use BEFORE asserting any numeric identity. "
            "Args: value (decimal string), closed_form (e.g. 'pi**2/6', "
            "'zeta(3)')."
        ),
        "params": {
            "value": {"type": "string", "description": "the value as a decimal string"},
            "closed_form": {
                "type": "string",
                "description": 'a closed-form expression, e.g. "pi**2/6" or "zeta(3)"',
            },
        },
        "required": ["value", "closed_form"],
        "output": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "closed_form": {"type": "string"},
                "verified": {"type": "boolean"},
                "digits_agreed": {"type": "integer"},
                "min_digits_required": {"type": "integer"},
                "reeval": {"type": ["string", "null"]},
                "error": {"type": ["string", "null"]},
                "note": {"type": "string"},
            },
            "required": ["verified", "digits_agreed"],
        },
    },
    {
        "name": "verify_formal",
        "title": "Verify formal (real Lean kernel)",
        "fn": tool_verify_formal,
        "description": (
            "Run the REAL Lean 4 kernel on a Lean snippet and report whether it "
            "typechecks; honest UNDETERMINED (with a remediation) when no snippet "
            "or no toolchain. Use to kernel-check a formal claim you wrote — find "
            "declaration names first with search_formal_math. Args: statement "
            "(what is being claimed), lean (the Lean 4 snippet — REQUIRED for a "
            "real check, e.g. 'example : 2 + 2 = 4 := rfl')."
        ),
        "params": {
            "statement": {"type": "string", "description": "the claim being checked"},
            "lean": {
                "type": "string",
                "description": "Lean 4 snippet to kernel-check, e.g. "
                '"example : 2 + 2 = 4 := rfl" (omit it and the verdict is an '
                "honest UNDETERMINED — statement text alone is not checkable)",
            },
        },
        "required": ["statement"],
        "output": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "lean_provided": {"type": "boolean"},
                "lean_available": {"type": "boolean"},
                "tier": {"type": "string"},
                "typechecks": {"type": ["boolean", "null"]},
                "applies": {"type": ["boolean", "null"]},
                "checked": {
                    "type": "boolean",
                    "description": "true iff the Lean kernel actually ran and gave a verdict",
                },
                "detail": {"type": "string"},
                "remediation": {
                    "type": "string",
                    "description": "present iff not checked — exactly what unblocks a real "
                    "kernel check",
                },
                "note": {"type": "string"},
            },
            "required": ["checked", "typechecks", "lean_available"],
        },
    },
    {
        "name": "applicability_checklist",
        "title": "Applicability checklist",
        "fn": tool_applicability_checklist,
        "description": (
            "Decompose a candidate theorem's statement into atomic preconditions + "
            "conclusion for YOU to verify one by one against your problem (catches "
            "misapplications like using a closed-interval theorem on an open "
            "interval). Use after search, before relying on any candidate. Args: "
            "candidate_statement (the result's statement text)."
        ),
        "params": {
            "candidate_statement": {
                "type": "string",
                "description": "the candidate result's statement",
            },
        },
        "required": ["candidate_statement"],
        "output": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "preconditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "satisfied": {
                                "type": ["boolean", "null"],
                                "description": "null — YOU mark each one against your problem",
                            },
                            "evidence": {"type": "string"},
                        },
                    },
                },
                "conclusion": {"type": "string"},
                "instructions": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["preconditions", "conclusion"],
        },
    },
    {
        "name": "mapping_scaffold",
        "title": "Needs-guarantees mapping scaffold",
        "fn": tool_mapping_scaffold,
        "description": (
            "Build the needs<->guarantees scaffold (structured questions + fill-in "
            "template) between your problem and a candidate result. Use when "
            "applicability is non-obvious and you want structure for the judgment "
            "(the judging is yours). Args: problem, candidate_statement."
        ),
        "params": {
            "problem": {"type": "string", "description": "the problem to solve"},
            "candidate_statement": {
                "type": "string",
                "description": "a candidate existing result's statement",
            },
        },
        "required": ["problem", "candidate_statement"],
        "output": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "candidate_statement": {"type": "string"},
                "signature": {"type": "object"},
                "checklist": {"type": "object"},
                "questions": {"type": "array"},
                "answer_template": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["questions", "answer_template"],
        },
    },
    # --- DISCOVERY + WEB-AUGMENTATION --- #
    {
        "name": "conjecture_relation",
        "title": "Conjecture relations (Ramanujan Machine)",
        "fn": tool_conjecture_relation,
        "description": (
            "Conjecture relations for a real constant — Ramanujan-Machine style: "
            "PSLQ over a rich basis + continued-fraction/recurrence search; every "
            "candidate numerically VERIFIED to >= 25 digits but NOT proved "
            "(provenance 'conjectured_relation'). Use when identify_constant "
            "returns UNIDENTIFIED. Args: value (decimal string, MANY digits), "
            "max_terms (default 16), cf_depth (default 200)."
        ),
        "params": {
            "value": {
                "type": "string",
                "description": "the real constant as a decimal string (give MANY digits; "
                "PSLQ/CF search needs >16)",
            },
            "max_terms": {
                "type": "integer",
                "description": "max PSLQ basis vector length (default 16; cost grows fast)",
            },
            "cf_depth": {
                "type": "integer",
                "description": "continued-fraction evaluation depth (default 200)",
            },
        },
        "required": ["value"],
        "output": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "found": {"type": "boolean"},
                "integer_relations": {"type": "array", "items": {"type": "object"}},
                "continued_fractions": {"type": "array", "items": {"type": "object"}},
                "simple_continued_fraction": {"type": ["object", "null"]},
                "note": {"type": "string"},
            },
            "required": ["found", "integer_relations", "continued_fractions"],
        },
    },
    {
        "name": "funsearch",
        "title": "FunSearch harness (evaluate/register/status)",
        "fn": tool_funsearch,
        "description": (
            "Sandboxed program-search harness (FunSearch): action='evaluate' "
            "scores YOUR Python program for problem_id ('cap_set' or "
            "'online_bin_packing') in a no-network/timeout/rlimit sandbox; "
            "action='register' stores a scored program in the MAP-Elites DB; "
            "action='status' returns the best programs + few-shot context for "
            "writing the next variant. Use to iteratively evolve programs — YOU "
            "are the generator, mathlas is the deterministic scorer. Args: action, "
            "problem_id, then program_src (evaluate/register), score + behavior "
            "(register), timeout_s (evaluate), top_k (status)."
        ),
        "params": {
            "action": {
                "type": "string",
                "enum": ["evaluate", "register", "status"],
                "description": "'evaluate' = sandbox-score program_src; "
                "'register' = store a scored program; "
                "'status' = best programs + few-shot context",
            },
            "problem_id": {
                "type": "string",
                "description": "the problem: 'cap_set' or 'online_bin_packing'",
            },
            "program_src": {
                "type": "string",
                "description": "(evaluate/register) the candidate Python program "
                "source — YOU write it; it must define the problem's "
                "entry point",
            },
            "score": {
                "type": "number",
                "description": "(register) the score that action='evaluate' returned",
            },
            "behavior": {
                "type": "array",
                "items": {"type": ["number", "string"]},
                "description": "(register) the behaviour descriptor from "
                "action='evaluate' (selects the MAP-Elites "
                "cell)",
            },
            "timeout_s": {
                "type": "number",
                "description": "(evaluate) hard wall-clock timeout seconds (default 10)",
            },
            "top_k": {
                "type": "integer",
                "description": "(status) elite programs in the few-shot (default 3)",
            },
        },
        "required": ["action", "problem_id"],
        "output": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "problem_id": {"type": "string"},
                "ok": {
                    "type": "boolean",
                    "description": "(evaluate) program ran + scored",
                },
                "score": {"type": ["number", "null"]},
                "behavior": {"type": "array"},
                "error": {
                    "type": ["string", "null"],
                    "description": "agent-actionable: what failed and which args to fix",
                },
                "accepted": {"type": "boolean", "description": "(register)"},
                "best_score": {"type": ["number", "null"], "description": "(status)"},
                "best_program": {"type": ["string", "null"], "description": "(status)"},
                "few_shot_context": {
                    "type": ["string", "null"],
                    "description": "(status) DATA for you to write the next program",
                },
                "note": {"type": "string"},
            },
            "required": ["problem_id"],
        },
    },
    {
        "name": "search_directive",
        "title": "Web-search directive (plan only)",
        "fn": tool_search_directive,
        "description": (
            "Get a STRUCTURED web-search plan for a problem — arXiv query strings, "
            "sub-fields/categories, named results to look for, and which other "
            "mathlas tools to run; mathlas makes NO web call (YOU search, then "
            "feed results back via add_finding). Use when the local index missed. "
            "Args: problem (description)."
        ),
        "params": {
            "problem": {
                "type": "string",
                "description": "a problem / result description to build a web-search plan for",
            },
        },
        "required": ["problem"],
        "output": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "signature": {"type": "object"},
                "arxiv_queries": {"type": "array", "items": {"type": "string"}},
                "subfields": {"type": "array", "items": {"type": "string"}},
                "arxiv_categories": {"type": "array", "items": {"type": "string"}},
                "named_results": {"type": "array", "items": {"type": "string"}},
                "also_try_mathlas_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "instructions": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["arxiv_queries", "instructions"],
        },
    },
    {
        "name": "add_finding",
        "title": "Add web finding to live corpus",
        "fn": tool_add_finding,
        "description": (
            "Ingest a web-found result into the live mathlas corpus so "
            "search_existing_math returns it immediately (provenance 'web_added'; "
            "BM25 always — no model load; full dense retrieval too if you pass "
            "dense_vec embedded in the served index's space). Use after "
            "web-searching per search_directive. Args: statement, slogan, source, "
            "optional name, optional dense_vec."
        ),
        "params": {
            "statement": {
                "type": "string",
                "description": "the web-found result's statement (the real text)",
            },
            "slogan": {
                "type": "string",
                "description": "a short natural-language denotation of it (what it says)",
            },
            "source": {
                "type": "string",
                "description": "where it came from: a URL / arXiv id / citation",
            },
            "name": {
                "type": "string",
                "description": "optional name/title of the result",
            },
            "dense_vec": {
                "type": "array",
                "items": {"type": "number"},
                "description": "OPTIONAL dense embedding of the slogan, computed BY YOU "
                "(the AI) with the SAME model the served index uses, length "
                "== the served index dim. Storing it gives the finding full "
                "dense+BM25 retrieval (found even when wording differs from "
                "the query). NO model is loaded by mathlas. Omit for "
                "BM25-only.",
            },
        },
        "required": ["statement", "slogan", "source"],
        "output": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "statement": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
                "slogan": {"type": ["string", "null"]},
                "source": {"type": ["string", "null"]},
                "provenance": {"type": ["string", "null"]},
                "dense_added": {"type": "boolean"},
                "n_findings": {"type": "integer"},
                "note": {
                    "type": "string",
                    "description": "on failure (e.g. dense_vec dim mismatch) says exactly what to "
                    "fix; the finding is NOT added",
                },
            },
            "required": ["ok", "note"],
        },
    },
]

#: Back-compat: the pre-1.1 funsearch trio still dispatches on the fallback
#: server (mapped onto funsearch(action=...)). NOT listed in tools/list — new
#: clients should call the single `funsearch` tool.
_FUNSEARCH_ALIASES: Dict[str, str] = {
    "funsearch_evaluate": "evaluate",
    "funsearch_register": "register",
    "funsearch_status": "status",
}


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
            "reason over), search_formal_math (mathlib declarations via the "
            "public Loogle/LeanSearch services). PLUS a discovery + "
            "web-augmentation layer: conjecture_relation (Ramanujan-Machine "
            "PSLQ-richer-basis + continued-fraction conjectures, VERIFIED not "
            "proved), funsearch (action=evaluate/register/status — a sandboxed "
            "program-search harness where YOU write the programs), and "
            "search_directive + add_finding (mathlas tells you what to "
            "web-search and ingests your findings into the live corpus with no "
            "model load). Typical flow: search_existing_math -> "
            "mapping_scaffold + applicability_checklist -> you judge "
            "applicability -> verify_numeric for any numeric claim."
        ),
    )
    # Register each tool. FastMCP introspects the wrapped fn's signature/types
    # and (SDK >= 1.10) derives outputSchema + structuredContent from the
    # Dict[str, Any] return annotation per MCP spec 2025-06-18. `title` is
    # passed when the installed SDK supports it; otherwise degrade gracefully.
    for spec in _TOOLS:
        kwargs = {
            "name": spec["name"],
            "description": (spec["description"] or "").strip(),
        }
        try:
            mcp.tool(title=spec.get("title"), **kwargs)(spec["fn"])
        except TypeError:  # older SDK without `title`
            mcp.tool(**kwargs)(spec["fn"])
    return mcp


# --------------------------------------------------------------------------- #
# Dependency-free fallback: a minimal stdio JSON-RPC MCP server.
# Implements just enough of the MCP wire protocol (initialize, tools/list,
# tools/call) to be usable with no third-party deps when ``mcp`` is absent.
# --------------------------------------------------------------------------- #
PROTOCOL_VERSION = "2025-06-18"


def _input_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": spec["params"],
        "required": spec.get("required", []),
    }


def _server_version() -> str:
    """The installed package version (for serverInfo), best-effort."""
    try:
        from importlib.metadata import version

        return version("mathlas-mcp")
    except Exception:
        return "1.1.0"


def _tool_error(message: str) -> Dict[str, Any]:
    """An in-band MCP tool-execution error (spec: isError result, not a protocol
    error) with an agent-actionable message."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _dispatch(method: str, params: Dict[str, Any]) -> Any:
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "mathlas",
                "title": "mathlas",
                "version": _server_version(),
            },
            "instructions": (
                "mathlas: a tool you use; it never calls an LLM and "
                "needs no API key. Search existing math + airtight "
                "verification + needs<->guarantees scaffolds."
            ),
        }
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return {}
    if method == "tools/list":
        tools = []
        for s in _TOOLS:
            t = {
                "name": s["name"],
                "title": s.get("title", s["name"]),
                "description": (s["description"] or "").strip(),
                "inputSchema": _input_schema(s),
            }
            if s.get("output"):  # MCP spec 2025-06-18 structured output
                t["outputSchema"] = s["output"]
            tools.append(t)
        return {"tools": tools}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name in _FUNSEARCH_ALIASES:  # pre-1.1 back-compat (unlisted)
            spec = next(s for s in _TOOLS if s["name"] == "funsearch")
            args = dict(args, action=_FUNSEARCH_ALIASES[name])
        else:
            spec = next((s for s in _TOOLS if s["name"] == name), None)
        if spec is None:
            raise ValueError(f"unknown tool: {name}")
        try:
            result = spec["fn"](**args)
        except TypeError as e:  # bad/missing arguments — point at the schema
            return _tool_error(
                f"Invalid arguments for {name}: {e}. Expected inputSchema: "
                f"{json.dumps(_input_schema(spec), default=str)}"
            )
        except FileNotFoundError as e:  # missing local data — message says what/where
            return _tool_error(f"{name}: required local data missing — {e}")
        except Exception as e:  # tool execution error, in-band per MCP spec
            return _tool_error(f"{name} failed: {e.__class__.__name__}: {e}")
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, indent=2, default=str)}
            ],
            "structuredContent": result,
            "isError": False,
        }
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
                out.write(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": mid,
                            "error": {"code": -32603, "message": str(e)},
                        }
                    )
                    + "\n"
                )
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
