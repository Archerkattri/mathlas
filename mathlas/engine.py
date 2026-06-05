"""math_engine v0 facade -- the numeric beachhead.

``identify(value)`` runs the full loop:

    route (implicit: real constant)
      -> identify   (find an existing closed form)
      -> verify     (independent high-precision re-evaluation)
      -> provenance (label the source; never "novel")

and returns a ``Result`` you can inspect or print.

Precision caveat: pass high-precision inputs as ``mpmath.mpf`` or as strings.
A Python ``float`` carries only ~15 digits, which limits PSLQ to the simplest
relations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mpmath

from .identify import identify_constant, DEFAULT_BASIS
from .verify import verify_closed_form, VerifyResult
from .provenance import Provenance, Novelty


def _pretty(expr: str) -> str:
    """A human-readable form of a verified PSLQ expression for DISPLAY only.
    Runs after verification, so it can never weaken the airtight check; on any
    failure it falls back to the raw expression. ``pi**2*(sqrt(24)/12)**2`` ->
    ``pi**2/6``; ``(1*catalan)`` -> ``Catalan``."""
    try:
        import sympy
        from .verify import _to_sympy_expr
        return str(sympy.simplify(_to_sympy_expr(expr)))
    except Exception:
        return expr


@dataclass(frozen=True)
class Candidate:
    expr: str                      # the verified PSLQ expression (raw)
    verify: VerifyResult
    provenance: Provenance

    @property
    def display(self) -> str:
        """Simplified, human-readable form (verification used ``expr``)."""
        return _pretty(self.expr)


@dataclass(frozen=True)
class Result:
    query: str
    best: Optional[Candidate]
    candidates: List[Candidate]

    @property
    def identified(self) -> bool:
        return self.best is not None

    def __str__(self) -> str:
        if not self.identified:
            return f"{self.query} -> UNIDENTIFIED (no verified closed form)"
        b = self.best
        return (f"{self.query} -> {b.display}  "
                f"[{b.provenance.novelty.value}, verified {b.verify.digits_agreed} digits]")


def identify(value, dps_search: int = 30, dps_verify: int = 50,
             min_digits: int = 20, basis=DEFAULT_BASIS) -> Result:
    """Find and verify an existing closed form for a real ``value``."""
    with mpmath.workdps(dps_verify):
        v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)
        query = mpmath.nstr(v, 15)

        proposals = identify_constant(v, dps_search=dps_search, basis=basis)

        cands: List[Candidate] = []
        for expr in proposals:
            vr = verify_closed_form(v, expr, dps_verify=dps_verify,
                                    min_digits=min_digits)
            if vr.ok:
                prov = Provenance(
                    novelty=Novelty.KNOWN_FORM,
                    method="mpmath.identify+sympy-verify",
                    source=expr,
                    basis=tuple(basis),
                )
                cands.append(Candidate(expr=expr, verify=vr, provenance=prov))

        cands.sort(key=lambda c: c.verify.digits_agreed, reverse=True)
        best = cands[0] if cands else None
        return Result(query=query, best=best, candidates=cands)
