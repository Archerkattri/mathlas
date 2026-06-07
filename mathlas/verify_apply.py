"""Tiered applicability VERIFICATION -- does a retrieved result actually apply?

Retrieval finds a *candidate* existing result; this module separates a real hit
from a confident hallucination -- the piece TheoremSearch (and every
retrieval-only tool) lacks. Three tiers, cheapest-first (the AlphaEvolve
cheap->expensive cascade):

  NUMERIC  (airtight)  -- when the claim reduces to a numeric identity, re-check
                          it independently at high precision. Reuses verify.py.
                          mathlas does this itself; NO LLM, NO API key.
  FORMAL   (Lean)      -- when a Lean term/check is available, kernel-check it.
                          Correct-by-construction but narrow; stubbed pending a
                          Lean toolchain (interface fixed so it slots in).
  INFORMAL             -- the general natural-language case. Here mathlas does
                          NOT pretend to judge: it builds a structured,
                          adversarial **applicability checklist** (the candidate's
                          atomic preconditions + need/guarantee questions) and
                          hands that to the CALLING AI to check against its own
                          problem. mathlas provides the scaffold; the AI reasons.

THE DESIGN CORRECTION (mathlas is a tool an AI uses, not one that uses an AI)
----------------------------------------------------------------------------
mathlas itself NEVER calls an LLM and needs NO API key. The informal-tier
applicability JUDGMENT is the calling AI's job, so the primary informal entry
point is ``applicability_checklist(candidate_statement)`` -- a pure-Python,
heuristic decomposition of the result into atomic, individually-falsifiable
preconditions for the AI to mark. (An OPTIONAL bring-your-own-LLM helper
``verify_informal(mapping, llm)`` remains for the standalone ``solve()`` path,
but it is secondary and mathlas never supplies the LLM.)

Why a checklist (grounded in the latest verification research)
--------------------------------------------------------------
The honest finding from 2025-2026 NL-proof grading is that a single LLM-as-judge
is unreliable -- sensitive to model/prompt, and a bare rubric does NOT reliably
help (Petrov et al., "ProofGrader," arXiv:2510.13888, ICLR 2026; "Scaling
Generative Verifiers," arXiv:2511.13027). What *does* work is **separating
generation from verification and decomposing the check into atomic, individually
falsifiable conditions** -- the generator-verifier split of DeepSeekMath-V2
(arXiv:2511.22570). mathlas operationalises exactly the decomposition half (the
part a tool CAN do deterministically) and leaves the falsification to the AI:

  1. EXTRACT the candidate's hypotheses as a checklist of atomic preconditions
     (from the result's own statement -- a problem-specific marking scheme, not a
     generic rubric). This is heuristic parsing, NO LLM.
  2. The CALLING AI adversarially checks each against ITS problem: find a
     precondition the problem does NOT establish, or a need/guarantee mismatch.
     Any single unmet precondition => does-not-apply.
  3. The AI aggregates: applies only if every precondition is satisfied AND
     need-meets-guarantee, with the failing condition surfaced when not.

We also honour the two hard-won lessons: (a) do NOT prepend retrieved math
blindly -- verification is a *gated* step, run only on the specific candidate;
(b) "typechecks / sounds plausible" != "solves the problem" -- the checklist
forces need-vs-guarantee fit, not mere coherence.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:  # optional bring-your-own-LLM path only; never imported at runtime
    from .llm import LLM
    from .map import Mapping


class Tier(str, Enum):
    NUMERIC = "numeric"
    FORMAL = "formal"
    INFORMAL = "informal"


@dataclass(frozen=True)
class Condition:
    text: str
    satisfied: Optional[bool]      # None = undetermined
    evidence: str = ""


@dataclass(frozen=True)
class ApplyVerdict:
    tier: Tier
    applies: bool                  # verifier's decision (may downgrade MAP's)
    confidence: float
    conditions: List[Condition] = field(default_factory=list)
    failure: Optional[str] = None  # the single condition that sank it, if any
    note: str = ""

    @property
    def unmet(self) -> List[Condition]:
        return [c for c in self.conditions if c.satisfied is False]


# --------------------------------------------------------------------------- #
# NUMERIC tier -- airtight, reuses the existing independent re-evaluation.
# --------------------------------------------------------------------------- #
def verify_numeric_claim(value, candidate_expr: str, *,
                         dps_verify: int = 50, min_digits: int = 20) -> ApplyVerdict:
    """Applicability == an exact numeric identity. Delegates to the airtight
    independent re-evaluation in verify.py (search-low / verify-high / different
    library). Use when MAP's connection reduces to 'these two quantities are
    equal to high precision'."""
    import mpmath
    from .verify import verify_closed_form
    v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)
    vr = verify_closed_form(v, candidate_expr, dps_verify=dps_verify,
                            min_digits=min_digits)
    cond = Condition(
        text=f"{candidate_expr} == query to >= {min_digits} digits",
        satisfied=vr.ok,
        evidence=f"agreed {vr.digits_agreed} digits"
                 + (f"; {vr.error}" if vr.error else ""),
    )
    return ApplyVerdict(
        tier=Tier.NUMERIC, applies=vr.ok,
        confidence=1.0 if vr.ok else 0.0, conditions=[cond],
        failure=None if vr.ok else "numeric identity not confirmed",
        note="independent high-precision re-evaluation",
    )


# --------------------------------------------------------------------------- #
# FORMAL tier -- REAL Lean kernel check (no LLM). Runs the Lean type-checker on a
# provided snippet and reports whether it TYPECHECKS. The honest caveat stands:
# typechecks != proves-it-applies (the snippet's *statement* must be the
# applicability claim; that mapping is the AI's job, not Lean's).
# --------------------------------------------------------------------------- #
#: Caveat threaded through every real verdict -- a kernel typecheck proves the
#: snippet is well-typed and its proof term checks, NOT that the theorem stated is
#: the right applicability claim. (typecheck != proves-it-applies.)
_TYPECHECK_CAVEAT = ("Caveat: a Lean typecheck proves the snippet is well-typed "
                     "and its proof term passes the kernel -- it does NOT prove "
                     "the stated theorem is the right applicability claim "
                     "(typecheck != proves-it-applies; the AI owns that mapping).")

#: How long to let Lean run before giving up (seconds). A bare snippet (no
#: mathlib) typechecks in well under this; the cap guards against a hang.
LEAN_TIMEOUT_S = 120

# Cache the resolved Lean executable path (or False if none) so discovery -- which
# may shell out to ``elan which`` -- runs at most once per process.
_LEAN_EXE_CACHE: Union[str, bool, None] = None


def _candidate_lean_dirs() -> List[str]:
    """Directories under which a directly-runnable (real, non-proxy) ``lean`` may
    live. The elan install used by mathlas keeps toolchains under
    ``reference/downloads/elan/toolchains/<tc>/bin``."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    roots = [
        os.environ.get("ELAN_HOME", ""),
        os.path.join(repo_root, "reference", "downloads", "elan"),
        os.path.join(os.getcwd(), "reference", "downloads", "elan"),
    ]
    dirs: List[str] = []
    for root in roots:
        if not root:
            continue
        tc = os.path.join(root, "toolchains")
        if os.path.isdir(tc):
            for name in sorted(os.listdir(tc)):
                b = os.path.join(tc, name, "bin")
                if os.path.isfile(os.path.join(b, "lean")):
                    dirs.append(b)
    return dirs


def find_lean() -> Optional[str]:
    """Locate a runnable Lean executable, or None. NO LLM, NO network.

    Order: ``LEAN`` env var -> a real toolchain ``lean`` under the mathlas elan
    install (preferred: invoking it directly avoids the elan proxy's cwd
    warnings) -> ``elan which lean`` -> ``lean`` on PATH. Cached per process."""
    global _LEAN_EXE_CACHE
    if _LEAN_EXE_CACHE is not None:
        return _LEAN_EXE_CACHE if isinstance(_LEAN_EXE_CACHE, str) else None  # False -> None

    found: Optional[str] = None
    env_lean = os.environ.get("LEAN")
    if env_lean and os.path.isfile(env_lean) and os.access(env_lean, os.X_OK):
        found = env_lean
    if not found:
        for b in _candidate_lean_dirs():
            cand = os.path.join(b, "lean")
            if os.access(cand, os.X_OK):
                found = cand
                break
    if not found:
        # Ask elan (env or PATH) to resolve the active toolchain's lean.
        elan = None
        elan_home = os.environ.get("ELAN_HOME", "")
        if elan_home:
            c = os.path.join(elan_home, "bin", "elan")
            if os.access(c, os.X_OK):
                elan = c
        elan = elan or shutil.which("elan")
        if elan:
            try:
                out = subprocess.run([elan, "which", "lean"], capture_output=True,
                                     text=True, timeout=30)
                p = out.stdout.strip()
                if out.returncode == 0 and p and os.path.isfile(p):
                    found = p
            except (OSError, subprocess.SubprocessError):
                pass
    if not found:
        found = shutil.which("lean")

    _LEAN_EXE_CACHE = found or False
    return found


def lean_typecheck(lean_snippet: str, *, lean_exe: Optional[str] = None,
                   timeout_s: int = LEAN_TIMEOUT_S) -> Tuple[Optional[bool], str]:
    """Run Lean on ``lean_snippet`` and report whether it TYPECHECKS. NO LLM.

    Returns ``(ok, detail)`` where ``ok`` is True (kernel-checked clean), False
    (Lean reported errors), or None (could not run Lean -> undetermined). The real
    Lean kernel verifies any proof term in the snippet, so a clean exit is a
    genuine type/proof check, not a guess."""
    exe = lean_exe or find_lean()
    if not exe:
        return None, "no Lean toolchain found"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(lean_snippet)
            if not lean_snippet.endswith("\n"):
                fh.write("\n")
            tmp_path = fh.name
        try:
            proc = subprocess.run([exe, tmp_path], capture_output=True, text=True,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None, f"Lean timed out after {timeout_s}s"
        out = (proc.stdout + proc.stderr).strip()
        # Lean exits 0 and is silent on a clean typecheck; nonzero + an 'error:'
        # diagnostic on failure. Treat any 'error:' as a failure even if (rarely)
        # the exit code is 0, and surface warnings without failing.
        has_error = bool(re.search(r"\berror:", out))
        if proc.returncode == 0 and not has_error:
            return True, (out or "kernel typecheck OK (no diagnostics)")
        return False, (out or f"lean exited {proc.returncode}")
    except OSError as e:
        return None, f"could not run Lean ({e})"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def verify_formal(lean_snippet: Optional[str], *,
                  lean_exe: Optional[str] = None) -> ApplyVerdict:
    """Kernel-check a Lean snippet (REAL check when Lean is installed; NO LLM).

    If a Lean toolchain is available and a snippet is given, this actually runs the
    Lean type-checker and returns a real verdict (typechecks => applies True with
    full confidence; Lean errors => applies False). If no snippet is given, or no
    Lean toolchain is found, it returns an HONEST undetermined verdict -- never a
    fake pass. Remember: a typecheck proves the snippet is well-typed and its proof
    term checks, NOT that the stated theorem is the right applicability claim
    (typecheck != proves-it-applies; the AI owns that mapping)."""
    if not lean_snippet:
        exe = lean_exe or find_lean()
        where = f" (Lean available at {exe})" if exe else " (no Lean toolchain found)"
        return ApplyVerdict(
            tier=Tier.FORMAL, applies=False, confidence=0.0,
            note="no Lean snippet provided; formal tier skipped" + where)

    ok, detail = lean_typecheck(lean_snippet, lean_exe=lean_exe)
    if ok is None:
        # Could not run Lean -> honest UNDETERMINED (the stub behaviour, but only
        # when Lean genuinely is not runnable; never a fake pass).
        return ApplyVerdict(
            tier=Tier.FORMAL, applies=False, confidence=0.0,
            conditions=[Condition(text="Lean snippet typechecks",
                                  satisfied=None, evidence=detail)],
            failure=None,
            note=("UNDETERMINED: could not run the Lean kernel check (" + detail
                  + "). Install the Lean toolchain to enable a real check "
                  "(see docs/methods.md). " + _TYPECHECK_CAVEAT))
    if ok:
        return ApplyVerdict(
            tier=Tier.FORMAL, applies=True, confidence=1.0,
            conditions=[Condition(text="Lean snippet typechecks",
                                  satisfied=True, evidence=detail)],
            note=("REAL Lean kernel check: snippet TYPECHECKS. " + _TYPECHECK_CAVEAT))
    return ApplyVerdict(
        tier=Tier.FORMAL, applies=False, confidence=0.0,
        conditions=[Condition(text="Lean snippet typechecks",
                              satisfied=False, evidence=detail)],
        failure="Lean reported type/elaboration errors",
        note=("REAL Lean kernel check: snippet does NOT typecheck. " + _TYPECHECK_CAVEAT))


# --------------------------------------------------------------------------- #
# INFORMAL tier (PRIMARY, NO LLM) -- the applicability CHECKLIST for the AI.
#
# mathlas does not judge applicability; it decomposes the candidate into atomic,
# individually-checkable preconditions and hands them to the calling AI. This is
# pure-Python heuristic parsing of mathematical prose -- no model, no API key.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Checklist:
    """A structured, NO-LLM applicability scaffold for the CALLING AI.

    mathlas extracts the candidate result's hypotheses as atomic preconditions
    and the conclusion it guarantees; the AI fills in ``satisfied`` for each
    precondition against its own problem, then decides applicability. This is the
    decomposition half of the generator/verifier split -- the part a tool can do
    deterministically -- with the adversarial falsification left to the AI.
    """
    statement: str
    preconditions: List[str]        # atomic hypotheses the AI must check
    conclusion: str                 # what the result GUARANTEES (its 'then')
    instructions: str               # how the AI should use this checklist
    note: str = ("Heuristic extraction (no LLM). The CALLING AI verifies each "
                 "precondition against its problem; a single unmet one => the "
                 "result does NOT apply.")

    def as_conditions(self) -> List[Condition]:
        """Preconditions as undetermined Conditions for the AI to mark."""
        return [Condition(text=p, satisfied=None, evidence="") for p in self.preconditions]


# Hypothesis-introducing cue words in mathematical statements. A clause opening
# with one of these typically states a PRECONDITION the result requires.
_HYP_CUES = (
    "let ", "suppose ", "assume ", "assuming ", "given ", "for all ", "for every ",
    "for any ", "for each ", "if ", "whenever ", "provided ", "where ",
    "such that ", "with ", "consider ", "fix ",
)
# Conclusion-introducing cues -- what the result GUARANTEES.
_CONCL_CUES = ("then ", "it follows that ", "we have ", "implies ", "yields ",
               "we conclude ", "hence ", "therefore ", "so that ")
# Strip a leading enumeration label so the statement body parses cleanly.
_LABEL = re.compile(
    r"^\s*(theorem|lemma|proposition|corollary|claim|fact|remark|definition)"
    r"\s*[\w.]*\s*[.:)]?\s*", re.IGNORECASE)


def _depth_aware_split(text: str, seps: tuple) -> List[str]:
    """Split ``text`` on any separator in ``seps``, but ONLY at bracket depth 0
    so a comma inside ``(X, d)`` or ``{a, b}`` never splits a tuple/set. ``seps``
    are matched case-insensitively as whole lowercase substrings between spaces
    (for words like 'and') or as literal punctuation (',', ';')."""
    out: List[str] = []
    depth = 0
    i = 0
    n = len(text)
    buf = []
    low = text.lower()
    while i < n:
        ch = text[i]
        if ch in "([{":
            depth += 1
            buf.append(ch); i += 1; continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch); i += 1; continue
        cut = None
        if depth == 0:
            for sep in seps:
                if sep[0].isalpha():               # word separator, need word bounds
                    cand = f" {sep} "
                    if low.startswith(cand, i):
                        cut = len(cand); break
                elif low.startswith(sep, i):        # punctuation separator
                    cut = len(sep); break
        if cut is not None:
            out.append("".join(buf)); buf = []; i += cut; continue
        buf.append(ch); i += 1
    out.append("".join(buf))
    return [p.strip(" .,") for p in out if p.strip(" .,")]


def _sentences(text: str) -> List[str]:
    """Split statement prose into sentence-ish / clause-ish units, conservatively.
    Splits on sentence punctuation and on the ' then ' boundary so an
    'If A and B, then C' statement yields the A,B hypotheses and the C conclusion
    as separate units. Bracketed commas are protected (see _depth_aware_split)."""
    s = _LABEL.sub("", (text or "").strip())
    # Turn the conclusion marker into a hard sentence break: '. then ' so the
    # sentence splitter (which needs whitespace after the period) cuts here, and
    # the resulting unit opens with the 'then ' conclusion cue.
    s = re.sub(r",?\s*\bthen\b", ". then", s, flags=re.IGNORECASE)
    parts: List[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", s):
        sent = sent.strip()
        if not sent:
            continue
        # break on ';' at depth 0 too (independent clauses)
        parts.extend(_depth_aware_split(sent, (";",)) or [sent])
    return parts


def _split_hyp_clauses(clause: str) -> List[str]:
    """Split a hypothesis clause that bundles several conditions (', and ', ' and ',
    'with') into atomic pieces, keeping each independently checkable. Commas/words
    inside brackets (e.g. the tuple in 'Let (X, d) be...') are NOT split."""
    out = _depth_aware_split(clause, ("and", "with", ","))
    return out or [clause.strip(" .,")]


def applicability_checklist(candidate_statement: str) -> Checklist:
    """Decompose a candidate result into an applicability checklist -- NO LLM.

    Parses the statement's prose heuristically into (a) atomic PRECONDITIONS the
    result requires (hypothesis-introducing clauses: 'Let', 'Suppose', 'If',
    'for all', 'where', ...) and (b) the CONCLUSION it guarantees ('then ...').
    The calling AI then checks each precondition against its own problem. mathlas
    supplies the scaffold; the AI does the judging.
    """
    units = _sentences(candidate_statement)
    preconds: List[str] = []
    concls: List[str] = []
    for u in units:
        low = u.lower()
        is_concl = low.startswith(".then") or any(low.startswith(c) for c in _CONCL_CUES)
        if is_concl:
            concls.append(re.sub(r"^\.?then\s+", "", u, flags=re.IGNORECASE).strip(" .,"))
            continue
        # a hypothesis clause if it opens with a cue word
        body = u.strip()
        matched = next((c for c in _HYP_CUES if low.startswith(c)), None)
        if matched:
            # drop a leading 'if '/'then '-style cue but KEEP 'let/suppose' (they
            # name the object the condition is about) for readability.
            payload = body
            if matched in ("if ", "whenever ", "provided ", "such that ", "where "):
                payload = body[len(matched):].strip()
            preconds.extend(_split_hyp_clauses(payload))
        elif not concls:
            # No cue and we have not hit a conclusion yet: treat as a setup/given.
            preconds.append(body.strip(" .,"))
        else:
            concls.append(body.strip(" .,"))

    # De-dupe preserving order; drop empties.
    seen = set()
    preconds = [p for p in preconds if p and not (p.lower() in seen or seen.add(p.lower()))]  # type: ignore[func-returns-value]
    if not preconds:
        preconds = ["(no explicit hypotheses parsed -- AI: infer the result's "
                    "preconditions from its statement and check each)"]
    conclusion = "; ".join(concls) or (
        "(no explicit conclusion parsed -- AI: state what this result guarantees)")

    instructions = (
        "For EACH precondition, decide whether YOUR problem establishes it "
        "(satisfied true/false/unknown) with a one-line reason. The result "
        "APPLIES only if every precondition is satisfied AND its conclusion "
        "supplies what your problem needs. A single unmet precondition, or a "
        "need/guarantee mismatch, means it does NOT apply -- name that one "
        "failing condition. Be adversarial: try to falsify applicability.")
    return Checklist(statement=(candidate_statement or "").strip(),
                     preconditions=preconds, conclusion=conclusion,
                     instructions=instructions)


# --------------------------------------------------------------------------- #
# INFORMAL tier (OPTIONAL, bring-your-own-LLM) -- standalone solve() helper.
#
# Secondary path: mathlas NEVER supplies the LLM. Used only by solve() when a
# caller brings their own. The PRIMARY informal interface is the checklist above.
# --------------------------------------------------------------------------- #
_EXTRACT_SYS = (
    "You are a meticulous mathematical referee. Extract the EXACT hypotheses / "
    "preconditions a stated result requires before it can be applied. List them "
    "atomically -- one checkable condition each. Do not judge the problem yet. "
    "Respond with ONLY the JSON asked for."
)
_EXTRACT_PROMPT = """\
KNOWN RESULT:
  name: {name}
  statement: {statement}

List the result's hypotheses as atomic, individually-checkable preconditions
(the things that must hold for it to apply). Respond with ONLY:
{{"preconditions": ["<condition 1>", "<condition 2>", ...]}}"""

_CHECK_SYS = (
    "You are an ADVERSARIAL mathematical verifier. A colleague claims a known "
    "result applies to a problem. Your job is to FALSIFY that claim if you can: "
    "find ONE precondition the problem fails to establish, or a gap between what "
    "the problem NEEDS and what the result DELIVERS. Be skeptical and concrete. "
    "Soundness over politeness. A result applies ONLY if every precondition is "
    "met AND its guarantee supplies the need. Respond with ONLY the JSON asked for."
)
_CHECK_PROMPT = """\
PROBLEM:
{problem}

KNOWN RESULT:
  name: {name}
  statement: {statement}

CLAIMED CONNECTION (from the mapping step):
  need:        {need}
  guarantee:   {guarantee}
  reduction:   {reduction}

PRECONDITIONS the result requires:
{precond_block}

For EACH precondition, decide whether the PROBLEM establishes it. Then judge
overall applicability. A single unmet precondition, or a need/guarantee mismatch,
means it does NOT apply. Respond with ONLY:
{{
  "conditions": [
    {{"text": "<precondition>", "satisfied": true|false|null, "evidence": "<why>"}}
  ],
  "need_met_by_guarantee": true|false,
  "applies": true|false,
  "confidence": 0.0-1.0,
  "failure": "<the single condition or mismatch that breaks it, or null>"
}}"""


def _extract_preconditions(mapping: "Mapping", llm: "LLM") -> List[str]:
    from .map import _extract_json
    c = mapping.candidate
    raw = llm.complete(
        _EXTRACT_PROMPT.format(name=c.name or "(unnamed)", statement=c.statement),
        system=_EXTRACT_SYS, temperature=0.0,
    )
    try:
        obj = json.loads(_extract_json(raw))
        pre = [str(p) for p in obj.get("preconditions", []) if str(p).strip()]
    except (json.JSONDecodeError, ValueError):
        pre = []
    # Fall back to the heuristic (NO-LLM) checklist, then to MAP's preconditions.
    if not pre:
        pre = applicability_checklist(c.statement).preconditions
    return pre or list(mapping.preconditions)


def verify_informal(mapping: "Mapping", llm: "LLM", *,
                    passes: int = 1) -> ApplyVerdict:
    """OPTIONAL bring-your-own-LLM applicability check (standalone solve() path).

    mathlas NEVER supplies the LLM; the PRIMARY informal interface is the NO-LLM
    ``applicability_checklist``. This helper exists for the standalone ``solve()``
    convenience when a caller brings their own LLM. It decomposes the candidate
    into atomic preconditions, then runs an adversarial skeptic that must either
    confirm every one against the problem or name the one that fails. ``passes``
    > 1 runs the skeptic independently several times and takes the WORST verdict
    (any pass finding a failure wins) -- cheap consistency-for-rejection.
    """
    preconds = _extract_preconditions(mapping, llm)
    block = "\n".join(f"  - {p}" for p in preconds) or "  (none stated; infer them)"
    c = mapping.candidate

    verdicts: List[ApplyVerdict] = []
    for _ in range(max(1, passes)):
        raw = llm.complete(
            _CHECK_PROMPT.format(
                problem=_problem_text(mapping),
                name=c.name or "(unnamed)", statement=c.statement,
                need=mapping.need or "(unspecified)",
                guarantee=mapping.guarantee or "(unspecified)",
                reduction=mapping.reduction or "(none)",
                precond_block=block,
            ),
            system=_CHECK_SYS, temperature=0.0,
        )
        verdicts.append(_parse_informal(raw))

    # Worst-case aggregation: a single pass finding non-application rejects.
    rejecting = [v for v in verdicts if not v.applies]
    chosen = (min(rejecting, key=lambda v: v.confidence) if rejecting
              else max(verdicts, key=lambda v: v.confidence))
    return chosen


def _problem_text(mapping: "Mapping") -> str:
    """The informal verifier needs the original problem. MAP threads it onto the
    Mapping (``problem`` field); fall back to need+connection if absent."""
    return mapping.problem or (
        f"(need) {mapping.need}\n(context) {mapping.connection}")


def _parse_informal(raw: str) -> ApplyVerdict:
    from .map import _extract_json
    try:
        obj = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return ApplyVerdict(tier=Tier.INFORMAL, applies=False, confidence=0.0,
                            note="verifier reply unparseable -> reject")
    conds = []
    for c in obj.get("conditions", []) or []:
        sat = c.get("satisfied")
        conds.append(Condition(
            text=str(c.get("text", "")),
            satisfied=(None if sat is None else bool(sat)),
            evidence=str(c.get("evidence", "")),
        ))
    applies = bool(obj.get("applies", False)) and bool(
        obj.get("need_met_by_guarantee", True))
    return ApplyVerdict(
        tier=Tier.INFORMAL,
        applies=applies,
        confidence=float(obj.get("confidence", 0.0) or 0.0),
        conditions=conds,
        failure=(obj.get("failure") or None) if not applies else None,
        note="structured-adversarial precondition check",
    )


__all__ = [
    "Tier", "Condition", "ApplyVerdict", "Checklist",
    "verify_numeric_claim", "verify_formal", "applicability_checklist",
    "verify_informal", "find_lean", "lean_typecheck",
]
