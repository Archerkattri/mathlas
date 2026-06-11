"""mathlas — a tool FOR an AI, not a tool that uses an AI. No API key. Free.

mathlas gives a CALLING AI (Claude Code, Cursor, any MCP client / agent) the
capabilities it lacks: search over EXISTING math, AIRTIGHT numeric/formal
verification, structured needs<->guarantees scaffolds, and honest provenance
(never "novel"). **mathlas itself NEVER calls an LLM and needs NO API key** — the
AI is the brain; mathlas is the toolbox. Plug it in over MCP (``mathlas.server``)
or call the library functions directly.

What mathlas provides (all with NO LLM, returning DATA for the AI to reason over)
---------------------------------------------------------------------------------
  IDENTIFY  a real value/constant -> a known closed form, verified by independent
            high-precision re-evaluation. ``identify`` / ``engine.py``. Airtight.
  SEARCH    a query -> ranked candidate EXISTING results, via OUR OWN hybrid
            (dense+BM25+RRF) index. ``retrieve`` / ``HybridRetriever``.
  VERIFY    numeric (airtight digit agreement) + formal (Lean, stubbed) tiers,
            plus an ``applicability_checklist`` -- the candidate's atomic
            preconditions for the AI to check. ``verify`` / ``verify_apply``.
  SCAFFOLD  the needs<->guarantees questions as data (``mapping_scaffold``) for
            the AI to answer -- the analogy reasoning is the AI's job. ``map``.
  PROVENANCE every result is tied to an existing source or labelled UNIDENTIFIED.

A small bring-your-own-LLM ``solve()`` helper exists as a SECONDARY standalone
convenience (you supply the LLM; the default is a no-op stub). mathlas ships no
vendor SDK and no default model.

    >>> import mpmath
    >>> from mathlas import identify
    >>> print(identify(mpmath.zeta(2)))      # doctest: +SKIP
    1.64493406684823 -> pi**2/6  [known_form, verified 48 digits]
"""
# Numeric domain (airtight, no LLM, no network).
from .engine import identify, Result, Candidate
from .provenance import Provenance, Novelty
from .verify import verify_closed_form, VerifyResult

# Integer-sequence domain (airtight EXACT term-match vs a local copy of OEIS;
# no LLM, no network at call time). Heavy data load stays lazy inside the module.
from .sequence import (identify_sequence, SequenceResult, SequenceMatch,
                       OEISIndex)

# Retrieval + scaffolds + verification tiers (NO LLM). ``solve`` pulls in
# numpy/scipy (declared deps); the heavier retrieval corpus/embedder imports
# stay lazy inside their modules.
from .map import (mapping_scaffold, MappingScaffold,
                  map_candidates, extract_signature, Mapping, Signature)
from .verify_apply import (applicability_checklist, Checklist,
                           verify_numeric_claim, verify_formal, verify_informal,
                           ApplyVerdict, Tier, Condition)
from .retrieve import Retriever, Candidate as RetrievedCandidate

# DISCOVERY + WEB-AUGMENTATION layer (NO LLM, no network, no API key).
#   ramanujan  -- PSLQ-over-richer-basis + Ramanujan-Machine continued-fraction
#                 conjectures, each numerically VERIFIED (provenance = conjecture).
#   funsearch  -- the deterministic HARNESS for AI-generated program search
#                 (sandboxed evaluate + on-disk MAP-Elites DB + few-shot status).
#   webaug     -- search_directive (tell the AI what to web-search) + add_finding
#                 (ingest a web result into the live corpus with NO model load).
from .ramanujan import (conjecture, ConjectureResult, integer_relations,
                        continued_fractions, simple_continued_fraction)
from .webaug import (search_directive, SearchDirective, add_finding,
                     AddFindingResult, search_findings, load_findings)

# OPTIONAL bring-your-own-LLM standalone helper (secondary; no vendor SDK).
from .solve import solve, Solution, AppliedResult
from .llm import LLM, EchoLLM

try:  # single source of truth: the installed package metadata (pyproject)
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("mathlas-mcp")
    except PackageNotFoundError:  # not installed (e.g. raw source tree)
        __version__ = "1.4.0"
except ImportError:  # pragma: no cover - importlib.metadata is stdlib >=3.8
    __version__ = "1.3.0"
__all__ = [
    # numeric (airtight)
    "identify", "Result", "Candidate",
    "verify_closed_form", "VerifyResult",
    "verify_numeric_claim",
    # integer sequences (airtight OEIS exact term-match)
    "identify_sequence", "SequenceResult", "SequenceMatch", "OEISIndex",
    # provenance
    "Provenance", "Novelty",
    # search (no LLM)
    "Retriever", "RetrievedCandidate",
    # scaffolds + verification tiers (no LLM)
    "mapping_scaffold", "MappingScaffold",
    "applicability_checklist", "Checklist",
    "verify_formal", "ApplyVerdict", "Tier", "Condition",
    # discovery + web-augmentation (no LLM, no network)
    "conjecture", "ConjectureResult", "integer_relations",
    "continued_fractions", "simple_continued_fraction",
    "search_directive", "SearchDirective", "add_finding", "AddFindingResult",
    "search_findings", "load_findings",
    # optional bring-your-own-LLM standalone path (secondary)
    "solve", "Solution", "AppliedResult",
    "map_candidates", "extract_signature", "Mapping", "Signature",
    "verify_informal",
    "LLM", "EchoLLM",
]
