"""Provenance labelling for math_engine.

Core discipline: this engine FINDS EXISTING math. It never claims novelty.
Every result is either tied to a known source (a closed form in known
constants, or a catalogued OEIS sequence) or labelled UNIDENTIFIED. This is
the gate the Oct-2025 GPT-5 episode lacked -- it reported retrieved results
as new. Here, "retrieved" is the whole point, and we always say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class Novelty(str, Enum):
    # --- numeric domain (engine.py) ---
    KNOWN_FORM = "known_form"          # closed form in known constants (PSLQ/identify)
    SEQUENCE_MATCH = "sequence_match"  # matched a catalogued OEIS sequence
    # --- discovery domain (ramanujan.py) ---
    CONJECTURED_RELATION = "conjectured_relation"  # numerically-VERIFIED conjecture, NOT a proof
    # --- retrieval/informal domain (solve.py) ---
    RETRIEVED_APPLIES = "retrieved_applies"        # found + verified to apply
    RETRIEVED_REJECTED = "retrieved_rejected"      # found, mapping claimed apply, VERIFY rejected
    RETRIEVED_UNVERIFIED = "retrieved_unverified"  # found + mapped, not (yet) verified
    WEB_ADDED = "web_added"            # AI fed a web-found result into the live corpus
    # --- shared ---
    UNIDENTIFIED = "unidentified"      # nothing found/verified -- no claim made

    @property
    def is_retrieved(self) -> bool:
        return self in (Novelty.RETRIEVED_APPLIES, Novelty.RETRIEVED_REJECTED,
                        Novelty.RETRIEVED_UNVERIFIED)

    @property
    def is_conjecture(self) -> bool:
        """A numerically-verified CONJECTURE -- airtight to N digits, NOT a proof.
        Honest middle ground between KNOWN_FORM (a relation in a closed basis) and
        UNIDENTIFIED (nothing found)."""
        return self is Novelty.CONJECTURED_RELATION


@dataclass(frozen=True)
class Provenance:
    novelty: Novelty
    method: str                         # e.g. "mpmath.identify+sympy-verify"
    source: Optional[str] = None        # e.g. "OEIS:A000045" or the basis relation
    basis: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_known(self) -> bool:
        """True when the engine tied the result to an existing source AND (where
        applicable) verified it -- i.e. a defensible 'this is existing math' claim."""
        return self.novelty in (Novelty.KNOWN_FORM, Novelty.SEQUENCE_MATCH,
                                Novelty.RETRIEVED_APPLIES)
