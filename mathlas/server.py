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

Register in Claude Code (no API key needed):

    claude mcp add mathlas -- python -m mathlas.server

or run directly over stdio:  ``python -m mathlas.server``.

Implementation: uses the official ``mcp`` Python SDK (FastMCP) if installed;
otherwise falls back to a dependency-free stdio JSON-RPC MCP server implementing
the same tools and the same wire protocol.
"""
from __future__ import annotations

import json
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


def _build_retriever(corpus_dir: Optional[str], limit: int):
    """Return a cached HybridRetriever for the seed corpus (default) or a corpus
    dir (open theorem dataset parquets, if pyarrow is available). NO GPU: always
    the zero-download HashingEmbedder."""
    from .retrieve.corpus import Document
    from .retrieve.hybrid import HybridRetriever

    key = f"{corpus_dir or '<seed>'}::{limit}"
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]

    if corpus_dir:
        from .retrieve.corpus import load_documents
        docs = load_documents(corpus_dir, limit=limit)
    else:
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

    Defaults to a small built-in seed corpus so it works with no GPU/downloads;
    pass ``corpus_dir`` (open theorem dataset parquets) for the real index.
    Returns ranked candidates for the calling AI to reason over."""
    retr = _build_retriever(corpus_dir, corpus_limit)
    cands = retr.retrieve(query, k=int(k))
    return {
        "query": query,
        "corpus": corpus_dir or "<built-in seed corpus>",
        "k": int(k),
        "candidates": [
            {"rank": i + 1, "name": c.name, "statement": c.statement,
             "source": c.source, "score": c.score,
             "slogan": (c.meta or {}).get("slogan")}
            for i, c in enumerate(cands)
        ],
        "next": ("For a promising candidate, call mapping_scaffold(problem, "
                 "candidate.statement) and applicability_checklist(candidate."
                 "statement); YOU (the AI) judge whether it applies."),
        "note": ("Hybrid dense(hashing)+BM25+RRF over OUR OWN index. NO LLM. "
                 "Default embedder is the zero-download fallback; the production "
                 "Qwen3 index is an offline-GPU build."),
    }


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
            "reason over). Typical flow: search_existing_math -> mapping_scaffold "
            "+ applicability_checklist -> you judge applicability -> verify_numeric "
            "for any numeric claim."),
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
