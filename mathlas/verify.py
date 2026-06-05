"""Independent verification of a candidate closed form.

The search (mpmath PSLQ / identify) proposes; THIS module disposes -- it
re-evaluates the candidate with a DIFFERENT engine (sympy) at HIGHER precision
than the search used, and accepts only on N-digit agreement. That asymmetry
(search low, verify high, independent library) is what makes spurious
integer-relation hits structurally hard to pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import mpmath
import sympy

# mpmath.identify emits constant names (lowercase, its own vocabulary); sympy
# uses different identifiers. We translate before sympify so the INDEPENDENT
# sympy re-evaluation actually resolves named constants instead of silently
# making a non-numeric Symbol (which would falsely reject e.g. Catalan, zeta(3)).
# Order matters: replace longer / function-like names first.
# Only constants sympy can re-evaluate to ARBITRARY precision are mapped; a
# constant with no exact sympy symbol (khinchin, glaisher) is intentionally left
# unmapped so it fails to verify rather than fake-matching a finite literal.
_NAME_MAP = [
    ("catalan", "Catalan"),
    ("euler", "EulerGamma"),     # mpmath 'euler' = Euler-Mascheroni gamma
    ("apery", "zeta(3)"),        # Apery's constant
]
_WORD = re.compile(r"[A-Za-z_]\w*")


def _to_sympy_expr(candidate: str):
    """Translate an mpmath-identify constant string into a sympy expression,
    resolving constant names sympy knows under different identifiers."""
    def repl(m: "re.Match") -> str:
        w = m.group(0)
        for src, dst in _NAME_MAP:
            if w == src:
                return dst
        return w
    translated = _WORD.sub(repl, candidate)
    return sympy.sympify(translated)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    digits_agreed: int
    reeval: Optional[str]          # candidate re-evaluated, as a decimal string
    error: Optional[str] = None


def verify_closed_form(value: "mpmath.mpf", candidate: str,
                       dps_verify: int = 50, min_digits: int = 20) -> VerifyResult:
    """Re-evaluate ``candidate`` (an mpmath-identify expression string) at
    ``dps_verify`` with sympy (a DIFFERENT engine) and compare it to ``value``.

    Returns the number of agreeing significant digits (relative error) and
    whether that clears ``min_digits``. A parse/eval failure is a clean
    rejection, not an exception. Named constants are name-translated to sympy's
    vocabulary first so the independent re-evaluation is real, not a no-op.
    """
    try:
        expr = _to_sympy_expr(candidate)
        evaluated = expr.evalf(dps_verify + 5)
        if not getattr(evaluated, "is_number", False):
            raise ValueError(f"non-numeric after eval: {candidate}")
        reval = mpmath.mpf(str(evaluated))
    except Exception as e:  # unparseable / non-numeric -> not verifiable
        return VerifyResult(ok=False, digits_agreed=0, reeval=None, error=str(e))

    with mpmath.workdps(dps_verify):
        scale = abs(value) if abs(value) > 1 else mpmath.mpf(1)
        diff = abs(reval - value) / scale
        if diff == 0:
            digits = dps_verify
        elif diff < 1:
            digits = int(-mpmath.log10(diff))
        else:
            digits = 0

    return VerifyResult(
        ok=digits >= min_digits,
        digits_agreed=max(digits, 0),
        reeval=mpmath.nstr(reval, min(dps_verify, 30)),
    )
