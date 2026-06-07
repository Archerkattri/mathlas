"""Map a numeric value to an EXISTING closed form via integer-relation search.

Uses ``mpmath.identify`` (PSLQ under the hood) against a basis of well-known
mathematical constants. This is the "find the existing math" step for the
numeric domain: every hit is, by construction, a combination of known
constants. Proposing is cheap and permissive here -- the independent verify
pass (see verify.py) is what filters spurious low-precision relations.
"""
from __future__ import annotations

from typing import List, Sequence

import mpmath

# Conservative basis of well-known constants. (Golden ratio is reachable via
# sqrt(5).) ``apery`` = zeta(3); ``catalan`` and ``euler`` are named constants
# mpmath.identify recognises and verify.py can independently re-evaluate in sympy
# (see verify._NAME_MAP). Kept conservative -- the high-precision verify gate is
# what guards false positives, so breadth is low-risk but we still err small.
DEFAULT_BASIS: tuple = (
    "pi", "e", "euler", "catalan", "apery",
    "log(2)", "log(3)",
    "sqrt(2)", "sqrt(3)", "sqrt(5)",
)


def identify_constant(value, dps_search: int = 30,
                      basis: Sequence[str] = DEFAULT_BASIS) -> List[str]:
    """Return candidate closed-form strings for ``value`` (possibly empty).

    Searches at ``dps_search``; downstream verification (higher precision)
    filters spurious hits, so we err toward proposing here.
    """
    candidates: List[str] = []
    with mpmath.workdps(dps_search):
        v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)

        # 1) bare identify -- rationals and simple algebraic closed forms.
        bare = mpmath.identify(v)
        if bare:
            candidates.append(bare)

        # 2) identify against the known-constant basis.
        try:
            withconst = mpmath.identify(v, list(basis))
        except Exception:
            withconst = None
        if withconst and withconst not in candidates:
            candidates.append(withconst)

    return candidates
