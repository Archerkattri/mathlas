"""MAP — the needs<->guarantees scaffold (the genuinely valuable step).

Given a PROBLEM and a CANDIDATE existing result, decide whether the candidate
applies and HOW: not keyword similarity, but matching what the problem NEEDS to
what a known result GUARANTEES -- directly, or after a reduction/analogy. This is
the step every retrieval tool (TheoremSearch included) punts to the human, and
the step the May-2026 unit-distance disproof showed is where the value lives.

THE DESIGN CORRECTION: mathlas is a tool an AI uses, not one that uses an AI.
----------------------------------------------------------------------------
The needs<->guarantees MAPPING is informal mathematical reasoning -- exactly what
the CALLING AI (Claude Code / Cursor / any agent) is for. So the PRIMARY mapping
interface, ``mapping_scaffold(problem, candidate_statement)``, builds the
needs<->guarantees structure as DATA -- the objects/need/given of the problem and
the explicit questions the AI must answer -- with **NO LLM and NO API key**. The
AI consumes the scaffold and does the reasoning; mathlas provides the structure.

An OPTIONAL bring-your-own-LLM path (``extract_signature`` / ``map_candidates``)
remains below for the standalone ``solve()`` convenience, mirroring the two-stage
abduction->deduction scheme the analogy literature favours (Webb et al. on LLM
analogy; structural-mapping work, e.g. arXiv:2603.29997). mathlas NEVER supplies
the LLM for it -- a caller brings their own. The engine NEVER claims novelty; it
only connects a problem to existing results.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .retrieve import Candidate

_SYSTEM = (
    "You are a mathematical reasoning engine. Your ONE job: decide whether a "
    "KNOWN result can be APPLIED to a given problem, by matching what the problem "
    "NEEDS to what the result GUARANTEES. You never invent new mathematics; you "
    "only connect the problem to existing results. Be skeptical -- if the result "
    "does not actually apply, say applies=false. Respond with ONLY the JSON asked for."
)

# --- Stage 1: ABDUCTION -- the problem's structure-level requirement signature.
_SIG_SYS = (
    "You are a mathematical reasoning engine. Extract the STRUCTURE of a problem: "
    "what objects it involves, the property or conclusion it seeks, and the "
    "hypotheses it already provides. Stay at the level of mathematical structure, "
    "not surface keywords. Respond with ONLY the JSON asked for."
)
_SIG_PROMPT = """\
PROBLEM:
{problem}

Extract the problem's requirement signature -- the abstract structure a solving
result must match, independent of any specific theorem. Respond with ONLY:
{{
  "objects": ["<the mathematical objects in play>"],
  "need": "<the property/object/conclusion the problem seeks, one line>",
  "given": ["<hypotheses the problem already provides>"],
  "field_hints": ["<sub-fields whose results might apply>"]
}}"""

# --- Stage 2: DEDUCTION -- match one candidate's guarantee to the signature.
_PROMPT = """\
PROBLEM (verbatim):
{problem}

PROBLEM REQUIREMENT SIGNATURE (extracted):
  objects: {objects}
  need:    {sig_need}
  given:   {given}

CANDIDATE KNOWN RESULT:
  name: {name}
  statement: {statement}
  source: {source}

Decide whether this known result can be applied, matching the SIGNATURE's need
to the result's GUARANTEE:
- What does the RESULT guarantee (its conclusion, under its hypotheses)?
- Does that guarantee supply the signature's need -- directly, or after a
  reduction/transform? Match requirement-to-guarantee, not keyword similarity.
- What hypotheses/preconditions must hold for the result to apply HERE, and does
  the problem's `given` establish them?

Respond with ONLY this JSON object:
{{
  "applies": true|false,
  "confidence": 0.0-1.0,
  "need": "<what the problem needs, one line>",
  "guarantee": "<what the result guarantees, one line>",
  "connection": "<how the result supplies the need, or why it does not>",
  "preconditions": ["<hypotheses that must hold here>"],
  "reduction": "<the transform/reduction linking them, or null>"
}}"""


@dataclass(frozen=True)
class Signature:
    """The problem's structure-level requirement signature (abduction stage)."""
    objects: List[str]
    need: str
    given: List[str]
    field_hints: List[str]


@dataclass(frozen=True)
class Mapping:
    candidate: Candidate
    applies: bool
    confidence: float
    need: str
    guarantee: str
    connection: str
    preconditions: List[str]
    reduction: Optional[str]
    problem: str = ""              # original problem text (threaded to VERIFY)


# --------------------------------------------------------------------------- #
# PRIMARY mapping interface (NO LLM): the needs<->guarantees scaffold as DATA.
#
# mathlas does not perform the analogy reasoning; it structures the problem and
# poses the exact needs<->guarantees questions for the CALLING AI to answer.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MappingScaffold:
    """The needs<->guarantees structure as text/data for the CALLING AI -- NO LLM.

    Fields the AI consumes to decide if the candidate applies and how:
      * ``signature`` -- a lightly-parsed view of the problem (objects / need /
        given) to anchor the AI's abduction step;
      * ``checklist``  -- the candidate's atomic preconditions + conclusion
        (from verify_apply.applicability_checklist);
      * ``questions``  -- the explicit needs<->guarantees questions to answer;
      * ``answer_template`` -- the JSON shape the AI should fill in, so its
        verdict is auditable (the same shape the optional LLM path emits).
    """
    problem: str
    candidate_statement: str
    signature: Dict[str, object]
    checklist: Dict[str, object]
    questions: List[str]
    answer_template: Dict[str, object]
    note: str = ("mathlas built this scaffold with NO LLM. The CALLING AI does "
                 "the needs<->guarantees reasoning and fills in answer_template; "
                 "then verify each precondition (applicability_checklist) and, "
                 "where the claim reduces to a numeric identity, call "
                 "verify_numeric for an airtight check.")


# Cue words that often introduce the problem's GOAL / what it needs.
_NEED_CUES = ("show that", "prove that", "prove", "show", "find", "determine",
              "compute", "decide whether", "decide", "establish", "is it true",
              "does", "verify that", "evaluate")
# Cue words that introduce a GIVEN hypothesis in the problem.
_GIVEN_CUES = ("let ", "suppose ", "assume ", "given ", "with ", "where ",
               "for all ", "for every ", "if ")


def _heuristic_signature(problem: str) -> Dict[str, object]:
    """A NO-LLM, lightly-parsed requirement signature for the AI to refine.

    Pulls a best-effort ``need`` (the goal clause) and ``given`` (hypothesis
    clauses) out of the problem prose. Deliberately shallow -- the AI does the
    real abduction; this just seeds the structure so the scaffold is concrete.
    """
    text = (problem or "").strip()
    low = text.lower()
    need = text
    for cue in _NEED_CUES:
        idx = low.find(cue)
        if idx != -1:
            need = text[idx:].strip().rstrip(".")
            break
    givens: List[str] = []
    # sentence-ish split; collect clauses opening with a 'given' cue.
    for chunk in re.split(r"(?<=[.;])\s+|,\s+", text):
        c = chunk.strip()
        cl = c.lower()
        if any(cl.startswith(g) for g in _GIVEN_CUES):
            givens.append(c.rstrip(".").strip())
    # crude object guess: capitalised tokens + math-y words.
    objects = sorted(set(re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text)))[:8]
    return {"objects": objects, "need": need, "given": givens}


def mapping_scaffold(problem: str, candidate_statement: str) -> MappingScaffold:
    """Build the needs<->guarantees scaffold for ``problem`` vs a candidate -- NO LLM.

    Returns the structured questions and a fill-in template the CALLING AI uses to
    decide whether the candidate applies and how (direct or via a reduction).
    mathlas supplies the structure; the AI does the reasoning. Pair the AI's
    answers with ``verify_apply.applicability_checklist`` (precondition marking)
    and, when the claim reduces to a numeric identity, ``verify_numeric``.
    """
    from .verify_apply import applicability_checklist  # avoid import cycle
    sig = _heuristic_signature(problem)
    cl = applicability_checklist(candidate_statement)
    checklist = {
        "preconditions": list(cl.preconditions),
        "conclusion": cl.conclusion,
        "instructions": cl.instructions,
    }
    questions = [
        "NEED: In one line, what does the problem actually require (its goal)?",
        "GUARANTEE: In one line, what does the candidate result guarantee "
        "(its conclusion, under its hypotheses)?",
        "CONNECTION: Does that guarantee supply the need -- directly, or after a "
        "reduction/transform? Match requirement-to-guarantee, NOT keyword overlap.",
        "REDUCTION: If not direct, what exact transform/reduction links them "
        "(or is there none, so the result does not apply)?",
        "PRECONDITIONS: For each precondition in the checklist, does the "
        "problem's `given` establish it? Name any one that fails.",
        "VERDICT: applies = true only if every precondition holds AND the "
        "guarantee meets the need.",
    ]
    answer_template = {
        "applies": "true|false",
        "confidence": "0.0-1.0",
        "need": "<what the problem needs, one line>",
        "guarantee": "<what the result guarantees, one line>",
        "connection": "<how the result supplies the need, or why it does not>",
        "preconditions": [
            {"text": "<precondition>", "satisfied": "true|false|unknown",
             "evidence": "<why>"}
        ],
        "reduction": "<the transform/reduction linking them, or null>",
        "failure": "<the single condition or mismatch that breaks it, or null>",
    }
    return MappingScaffold(
        problem=(problem or "").strip(),
        candidate_statement=(candidate_statement or "").strip(),
        signature=sig, checklist=checklist, questions=questions,
        answer_template=answer_template,
    )


# --------------------------------------------------------------------------- #
# OPTIONAL bring-your-own-LLM path (standalone solve()): two-stage abduction ->
# deduction. mathlas NEVER supplies the LLM; the caller brings their own.
# --------------------------------------------------------------------------- #
def extract_signature(problem: str, llm: "LLM") -> Signature:
    """ABDUCTION: extract the problem's requirement signature once, up front.
    On a parse failure, fall back to a minimal signature (problem as its own
    need) so the deduction stage still runs."""
    raw = llm.complete(_SIG_PROMPT.format(problem=problem),
                       system=_SIG_SYS, temperature=0.0)
    try:
        obj = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return Signature(objects=[], need=problem, given=[], field_hints=[])
    return Signature(
        objects=list(obj.get("objects", []) or []),
        need=str(obj.get("need", "") or problem),
        given=list(obj.get("given", []) or []),
        field_hints=list(obj.get("field_hints", []) or []),
    )


def map_candidate(problem: str, candidate: Candidate, llm: LLM,
                  signature: Optional[Signature] = None) -> Optional[Mapping]:
    """DEDUCTION: needs<->guarantees mapping for one candidate against the
    (precomputed) signature. Returns None if the reply can't be parsed."""
    sig = signature or extract_signature(problem, llm)
    prompt = _PROMPT.format(
        problem=problem,
        objects=", ".join(sig.objects) or "(unspecified)",
        sig_need=sig.need,
        given=", ".join(sig.given) or "(none stated)",
        name=candidate.name or "(unnamed)",
        statement=candidate.statement,
        source=candidate.source or "(unknown)",
    )
    raw = llm.complete(prompt, system=_SYSTEM, temperature=0.0)
    try:
        obj = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    return Mapping(
        candidate=candidate,
        applies=bool(obj.get("applies", False)),
        confidence=float(obj.get("confidence", 0.0) or 0.0),
        need=str(obj.get("need", "")),
        guarantee=str(obj.get("guarantee", "")),
        connection=str(obj.get("connection", "")),
        preconditions=list(obj.get("preconditions", []) or []),
        reduction=obj.get("reduction") or None,
        problem=problem,
    )


def map_candidates(problem: str, candidates: List[Candidate], llm: LLM,
                   min_confidence: float = 0.5,
                   signature: Optional[Signature] = None) -> List[Mapping]:
    """Map each candidate against ONE shared signature (extracted once), keeping
    those that apply at/above the confidence threshold, best first."""
    sig = signature or extract_signature(problem, llm)
    mapped = (map_candidate(problem, c, llm, signature=sig) for c in candidates)
    keep = [m for m in mapped if m and m.applies and m.confidence >= min_confidence]
    keep.sort(key=lambda m: m.confidence, reverse=True)
    return keep


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of an LLM reply (handles ``` fences + prose)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return s[start:end + 1]
