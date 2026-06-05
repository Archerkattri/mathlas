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

# Retrieval + scaffolds + verification tiers (NO LLM). ``solve`` pulls in
# numpy/scipy (declared deps); the heavier retrieval corpus/embedder imports
# stay lazy inside their modules.
from .map import (mapping_scaffold, MappingScaffold,
                  map_candidates, extract_signature, Mapping, Signature)
from .verify_apply import (applicability_checklist, Checklist,
                           verify_numeric_claim, verify_formal, verify_informal,
                           ApplyVerdict, Tier, Condition)
from .retrieve import Retriever, Candidate as RetrievedCandidate

# OPTIONAL bring-your-own-LLM standalone helper (secondary; no vendor SDK).
from .solve import solve, Solution, AppliedResult
from .llm import LLM, EchoLLM

__version__ = "0.1.0"
__all__ = [
    # numeric (airtight)
    "identify", "Result", "Candidate",
    "verify_closed_form", "VerifyResult",
    "verify_numeric_claim",
    # provenance
    "Provenance", "Novelty",
    # search (no LLM)
    "Retriever", "RetrievedCandidate",
    # scaffolds + verification tiers (no LLM)
    "mapping_scaffold", "MappingScaffold",
    "applicability_checklist", "Checklist",
    "verify_formal", "ApplyVerdict", "Tier", "Condition",
    # optional bring-your-own-LLM standalone path (secondary)
    "solve", "Solution", "AppliedResult",
    "map_candidates", "extract_signature", "Mapping", "Signature",
    "verify_informal",
    "LLM", "EchoLLM",
]
