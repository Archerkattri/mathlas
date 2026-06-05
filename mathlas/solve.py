"""solve() -- OPTIONAL standalone bring-your-own-LLM convenience.

IMPORTANT: this is the SECONDARY interface. mathlas is a tool an AI *uses*, not
one that uses an AI. The PRIMARY way to drive the informal/retrieval domain is
the MCP tools (``mathlas/server.py``) or the plain library functions
(``search_existing_math`` -> ``mapping_scaffold`` -> ``applicability_checklist``
-> ``verify_numeric``) -- all with NO LLM and NO API key, where the CALLING AI
does the reasoning.

``solve()`` exists only as a one-shot library helper for callers who want to
*bring their own* LLM and have mathlas run the needs<->guarantees reasoning loop
for them. mathlas ships NO vendor SDK and NO default model: ``llm`` defaults to
the ``EchoLLM`` no-op stub (so the package runs with zero API key), which returns
no applicable mapping -- supply your own ``LLM`` subclass to get real reasoning.

Pipeline (when a real LLM is supplied)
--------------------------------------
  retrieve  -> our own hybrid (dense+BM25+RRF) index returns candidates
  MAP       -> two-stage (abduction signature -> per-candidate deduction)
  VERIFY    -> structured-adversarial informal applicability check;
               numeric/formal tiers are available via verify_apply for claims
               that reduce to those forms
  PROVENANCE-> RETRIEVED_APPLIES / RETRIEVED_UNVERIFIED / UNIDENTIFIED -- never
               "novel" (we FIND existing math and say so)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LLM, EchoLLM
from .map import Mapping, extract_signature, map_candidates
from .provenance import Provenance, Novelty
from .retrieve import Retriever
from .verify_apply import ApplyVerdict, verify_informal


@dataclass(frozen=True)
class AppliedResult:
    """One existing result the engine connected to the problem, with its
    mapping, applicability verdict, and provenance."""
    mapping: Mapping
    verdict: Optional[ApplyVerdict]
    provenance: Provenance

    @property
    def verified(self) -> bool:
        return self.verdict is not None and self.verdict.applies


@dataclass(frozen=True)
class Solution:
    problem: str
    results: List[AppliedResult] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return any(r.verified for r in self.results)

    @property
    def best(self) -> Optional[AppliedResult]:
        verified = [r for r in self.results if r.verified]
        if verified:
            return max(verified, key=lambda r: r.verdict.confidence)
        return self.results[0] if self.results else None

    def __str__(self) -> str:
        if not self.results:
            return f"{self.problem!r} -> UNIDENTIFIED (no applicable existing result found)"
        lines = [f"{self.problem!r} -> {len(self.results)} candidate result(s):"]
        for r in self.results:
            m, v = r.mapping, r.verdict
            tag = r.provenance.novelty.value
            conf = v.confidence if v else m.confidence
            lines.append(f"  - [{conf:.2f} | {tag}] {m.candidate.name or m.guarantee}")
            lines.append(f"      connection: {m.connection}")
            if v and not v.applies and v.failure:
                lines.append(f"      VERIFY FAILED: {v.failure}")
            lines.append(f"      source: {m.candidate.source or '(unknown)'}")
        return "\n".join(lines)


def solve(problem: str, retriever: Retriever, llm: Optional[LLM] = None, *,
          k: int = 10, min_confidence: float = 0.5,
          verify: bool = True, verify_passes: int = 1) -> Solution:
    """OPTIONAL bring-your-own-LLM helper: find, map, and verify existing results.

    ``llm`` defaults to ``EchoLLM`` (a no-op stub) so the package runs with NO
    API key -- but then no mapping is produced. Supply your OWN ``LLM`` subclass
    for real needs<->guarantees reasoning. For the no-LLM, AI-driven path, use the
    MCP tools / library functions (``search_existing_math`` -> ``mapping_scaffold``
    -> ``applicability_checklist`` -> ``verify_numeric``) instead.

    ``verify=False`` returns mapped-but-unverified candidates (faster). With
    ``verify=True`` every mapped candidate is run through the structured
    informal check, and provenance reflects whether it survived.
    """
    llm = llm or EchoLLM()
    candidates = retriever.retrieve(problem, k=k)
    sig = extract_signature(problem, llm)
    mappings = map_candidates(problem, candidates, llm,
                              min_confidence=min_confidence, signature=sig)

    results: List[AppliedResult] = []
    for m in mappings:
        verdict = verify_informal(m, llm, passes=verify_passes) if verify else None
        if verdict is None:
            novelty = Novelty.RETRIEVED_UNVERIFIED
        elif verdict.applies:
            novelty = Novelty.RETRIEVED_APPLIES
        else:
            novelty = Novelty.RETRIEVED_REJECTED
        prov = Provenance(
            novelty=novelty,
            method="hybrid-retrieve + 2stage-map"
                   + ("+adversarial-verify" if verify else ""),
            source=m.candidate.source,
        )
        results.append(AppliedResult(mapping=m, verdict=verdict, provenance=prov))

    # Verified-applicable first, then by confidence.
    results.sort(key=lambda r: (r.verified,
                                r.verdict.confidence if r.verdict else r.mapping.confidence),
                 reverse=True)
    return Solution(problem=problem, results=results)


__all__ = ["solve", "Solution", "AppliedResult"]
