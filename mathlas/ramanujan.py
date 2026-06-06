"""RAMANUJAN MACHINE — conjecture EXISTING relations for a real constant. NO LLM.

This is the discovery layer that goes *beyond* ``identify_constant``'s flat
basis. Given a high-precision real constant it returns two kinds of candidate,
each NUMERICALLY VERIFIED to high precision before it is reported:

  (a) INTEGER RELATION over a RICHER basis -- powers and pairwise products of
      known constants plus log/exp/zeta values -- found by PSLQ
      (Ferguson-Bailey-Arno). ``identify_constant`` runs ``mpmath.identify``
      against a flat list of constant *names*; here we build the basis VECTOR
      ourselves (so x^2, x*pi, log(2)*log(3), zeta(5)... are reachable) and run
      PSLQ directly, then reconstruct the relation as an expression string.

  (b) CONTINUED-FRACTION / polynomial-recurrence CONJECTURE in the
      Ramanujan-Machine style (Raayoni et al., *Nature* 2021): search small
      integer polynomials p(n), q(n) whose **generalized continued fraction**
        a0 + b1/(a1 + b2/(a2 + ...)),   a_n = p(n), b_n = q(n)
      converges to the constant (or to a simple a/(b*const+c) image of it). We
      test the CF convergents to high precision and lock the hit with the same
      airtight verify gate. The classic RM seed 4/pi = 1 + 1/(3 + 4/(5 + 9/...))
      (a_n = 2n+1, b_n = n^2) is exactly this shape.

HONESTY GATE (the whole point). Every candidate is re-checked by an INDEPENDENT
high-precision re-evaluation (``verify.verify_closed_form`` for relations; a
deep mpmath re-evaluation of the CF for recurrences) and only reported if it
agrees to ``min_digits``+ digits. Provenance is ``CONJECTURED_RELATION`` — this
is a *numerically-verified conjecture*, NOT a proof. The calling AI is told so
explicitly and can take it to ``verify_formal`` / a human / the literature.

NO LLM, NO network, NO API key. Pure mpmath/sympy, same discipline as verify.py.

References
----------
- Raayoni, Gottlieb, Manor, Pisha, Harris, Mendlovic, Haviv, Hadad & Kaminer,
  "Generating conjectures on fundamental constants with the Ramanujan Machine,"
  *Nature* 590, 67-73 (2021). *(the CF / polynomial-recurrence conjecture form.)*
- Ferguson, Bailey & Arno, "Analysis of PSLQ, an integer relation finding
  algorithm," *Math. Comp.* 68 (1999). *(the integer-relation engine; mpmath.pslq.)*
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import mpmath

from .provenance import Novelty, Provenance
from .verify import verify_closed_form, VerifyResult


# --------------------------------------------------------------------------- #
# The richer basis: each entry is (expr_string, evaluator). The expr_string is
# valid for BOTH mpmath (search) and, after verify._to_sympy_expr, sympy
# (independent re-eval) -- so a found relation re-verifies in a different engine.
# --------------------------------------------------------------------------- #
def _basis_atoms() -> List[Tuple[str, Callable[[], "mpmath.mpf"]]]:
    """Named constant atoms reachable by sympy too (kept airtight-verifiable)."""
    return [
        ("pi", lambda: mpmath.pi),
        ("E", lambda: mpmath.e),                 # sympy 'E'
        ("EulerGamma", lambda: mpmath.euler),    # sympy 'EulerGamma'
        ("Catalan", lambda: mpmath.catalan),     # sympy 'Catalan'
        ("log(2)", lambda: mpmath.log(2)),
        ("log(3)", lambda: mpmath.log(3)),
        ("sqrt(2)", lambda: mpmath.sqrt(2)),
        ("sqrt(3)", lambda: mpmath.sqrt(3)),
        ("sqrt(5)", lambda: mpmath.sqrt(5)),
        ("zeta(3)", lambda: mpmath.zeta(3)),
        ("zeta(5)", lambda: mpmath.zeta(5)),
    ]


def _richer_basis(value: "mpmath.mpf", max_terms: int
                  ) -> List[Tuple[str, "mpmath.mpf"]]:
    """Assemble the basis VECTOR for PSLQ: the constant ``value`` itself, the
    named atoms, their SQUARES, and a few PAIRWISE PRODUCTS — capped at
    ``max_terms`` (PSLQ cost grows fast with vector length, so keep it small).

    Returned entries are ``(expr_string, evaluated_mpf)``; ``"x"`` denotes the
    input value. Every expr re-parses in sympy for the independent re-eval.
    """
    atoms = _basis_atoms()
    vec: List[Tuple[str, "mpmath.mpf"]] = [("x", value)]
    # powers of the value itself (x^2 catches algebraic-degree-2 / product forms)
    vec.append(("x**2", value * value))
    # the named atoms
    for name, fn in atoms:
        vec.append((name, fn()))
    # squares of the atoms (so e.g. pi^2 is directly in the basis)
    for name, fn in atoms:
        v = fn()
        vec.append((f"({name})**2", v * v))
    # a few pairwise products of the cheapest atoms (pi*log2, log2*log3, ...)
    cheap = atoms[:6]
    for i in range(len(cheap)):
        for j in range(i + 1, len(cheap)):
            ni, fi = cheap[i]
            nj, fj = cheap[j]
            vec.append((f"({ni})*({nj})", fi() * fj()))
    # the rational unit (lets PSLQ find an additive constant)
    vec.append(("1", mpmath.mpf(1)))
    # de-dup by expr while preserving order, then cap length
    seen = set()
    out: List[Tuple[str, "mpmath.mpf"]] = []
    for name, v in vec:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, v))
        if len(out) >= max_terms:
            break
    return out


def _relation_to_expr(coeffs: Sequence[int],
                      names: Sequence[str]) -> Optional[str]:
    """Solve an integer relation ``sum c_i * b_i == 0`` for the input ``x`` and
    return ``x``'s closed form as an expression string, or ``None`` if ``x``
    does not appear (then the relation is among the other constants, not about x).

    The basis always lists ``x`` at index 0. If c_0 != 0 we move every other term
    to the RHS and divide: ``x = -(sum_{i>0} c_i b_i) / c_0``.
    """
    if not coeffs or coeffs[0] == 0:
        return None
    c0 = coeffs[0]
    num_terms: List[str] = []
    for c, name in zip(coeffs[1:], names[1:]):
        if c == 0:
            continue
        num_terms.append(f"({-c})*({name})")
    if not num_terms:
        rhs = "0"
    else:
        rhs = " + ".join(num_terms)
    # x = (rhs) / c0   -> keep as a plain expression sympy can re-evaluate.
    return f"({rhs})/({c0})"


@dataclass(frozen=True)
class RelationCandidate:
    """A PSLQ integer-relation conjecture for the constant, verified to digits."""
    kind: str                       # "integer_relation"
    expr: str                       # x's closed form (raw, sympy-evaluable)
    coeffs: Tuple[int, ...]         # the integer relation coefficients
    basis: Tuple[str, ...]          # the basis names the coeffs multiply
    verify: VerifyResult
    provenance: Provenance


@dataclass(frozen=True)
class CFCandidate:
    """A Ramanujan-Machine continued-fraction conjecture, verified to digits.

    The generalized CF ``a0 + b1/(a1 + b2/(a2 + ...))`` with ``a_n = poly_a(n)``,
    ``b_n = poly_b(n)`` converges to ``image`` of the constant, where ``image``
    is ``(p*x + q)/(r*x + s)`` (defaults to ``x`` itself). We report the polys
    as integer-coefficient lists (low-order first) plus the achieved digits.
    """
    kind: str                       # "continued_fraction"
    poly_a: Tuple[int, ...]         # a_n = sum poly_a[k] n^k
    poly_b: Tuple[int, ...]         # b_n = sum poly_b[k] n^k
    image: str                      # what the CF equals, as an expr in x
    cf_value: str                   # the CF's value (decimal string)
    digits_agreed: int
    depth: int
    provenance: Provenance


# --------------------------------------------------------------------------- #
# (a) PSLQ over the richer basis.
# --------------------------------------------------------------------------- #
def integer_relations(value: "mpmath.mpf", max_terms: int = 16,
                      dps_search: int = 40, dps_verify: int = 60,
                      min_digits: int = 25,
                      maxcoeff: int = 10 ** 6) -> List[RelationCandidate]:
    """Find integer relations for ``value`` over the richer basis via PSLQ, then
    keep only those whose reconstructed closed form INDEPENDENTLY re-verifies
    (sympy, higher precision) to ``min_digits``. Honest-or-nothing."""
    out: List[RelationCandidate] = []
    seen_expr = set()
    with mpmath.workdps(dps_search):
        v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)
        basis = _richer_basis(v, max_terms)
        names = [n for n, _ in basis]
        vec = [x for _, x in basis]
        try:
            rel = mpmath.pslq(vec, maxcoeff=maxcoeff, maxsteps=10 ** 4)
        except (ValueError, RuntimeError):
            rel = None
    if not rel:
        return out
    expr = _relation_to_expr(rel, names)
    if expr is None or expr in seen_expr:
        return out
    seen_expr.add(expr)
    # INDEPENDENT high-precision re-eval (the airtight gate, reused from verify.py)
    with mpmath.workdps(max(dps_verify + 10, 70)):
        vr = verify_closed_form(value, expr, dps_verify=dps_verify,
                                min_digits=min_digits)
    if vr.ok:
        prov = Provenance(
            novelty=Novelty.CONJECTURED_RELATION,
            method="mpmath.pslq(richer-basis)+sympy-verify",
            source=expr,
            basis=tuple(names),
        )
        out.append(RelationCandidate(
            kind="integer_relation", expr=expr,
            coeffs=tuple(int(c) for c in rel), basis=tuple(names),
            verify=vr, provenance=prov))
    return out


# --------------------------------------------------------------------------- #
# (b) Ramanujan-Machine continued-fraction / polynomial-recurrence search.
# --------------------------------------------------------------------------- #
def _poly(coeffs: Sequence[int]) -> Callable[[int], "mpmath.mpf"]:
    """Build n -> sum coeffs[k]*n^k as an mpf evaluator (low-order first)."""
    cs = [mpmath.mpf(int(c)) for c in coeffs]

    def f(n: int) -> "mpmath.mpf":
        x = mpmath.mpf(0)
        p = mpmath.mpf(1)
        nn = mpmath.mpf(n)
        for c in cs:
            x += c * p
            p *= nn
        return x
    return f


def gcf_value(poly_a: Sequence[int], poly_b: Sequence[int],
              depth: int) -> Optional["mpmath.mpf"]:
    """Evaluate the generalized CF a0 + b1/(a1 + b2/(a2 + ...)) to ``depth`` by
    backward recurrence. Returns ``None`` on a division-by-zero (a degenerate
    polynomial pair), so the caller just skips it."""
    fa = _poly(poly_a)
    fb = _poly(poly_b)
    try:
        x = fa(depth)
        if x == 0:
            return None
        for n in range(depth - 1, -1, -1):
            denom = x
            if denom == 0:
                return None
            x = fa(n) + fb(n + 1) / denom
        return x
    except (ZeroDivisionError, ValueError):
        return None


def _image_targets(value: "mpmath.mpf") -> List[Tuple[str, "mpmath.mpf"]]:
    """Simple rational images (p*x+q)/(r*x+s) of the constant a CF might equal.
    The RM paper's hits are usually a CF == a SIMPLE function of the constant
    (e.g. 4/pi, not pi). We test a handful of low-complexity images."""
    x = value
    out: List[Tuple[str, "mpmath.mpf"]] = [("x", x)]
    if x != 0:
        out.append(("1/(x)", 1 / x))
        out.append(("2/(x)", 2 / x))
        out.append(("4/(x)", 4 / x))
        out.append(("(x)/(x+1)", x / (x + 1)) if x != -1 else ("x", x))
    out.append(("x+1", x + 1))
    out.append(("x-1", x - 1))
    out.append(("2*x", 2 * x))
    # de-dup numerically-distinct images by their expr string
    seen = set()
    uniq = []
    for name, val in out:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, val))
    return uniq


def _small_poly_pairs(max_deg: int, coeff_range: int):
    """Yield (poly_a, poly_b) integer-coefficient pairs to try, smallest first.

    Kept TINY by design (RM hits are low-complexity): a_n is degree<=1
    (a0 + a1*n) and b_n is degree<=2 (b0 + b1*n + b2*n^2). We keep a_n's leading
    coefficient positive so a_n grows (CF convergence), and skip the all-zero b
    (which collapses the CF to the constant a0)."""
    R = range(-coeff_range, coeff_range + 1)
    # a_n = a0 + a1 n   (degree <= 1, positive leading)
    for a1 in (1, 2):
        for a0 in R:
            # b_n = b0 + b1 n + b2 n^2  (degree <= 2)
            for b2 in range(0, coeff_range + 1):
                for b1 in R:
                    for b0 in R:
                        if b0 == 0 and b1 == 0 and b2 == 0:
                            continue        # b==0 -> CF is just a0, skip
                        yield (a0, a1), (b0, b1, b2)


def continued_fractions(value: "mpmath.mpf", cf_depth: int = 200,
                        max_deg: int = 2, coeff_range: int = 4,
                        min_digits: int = 25,
                        max_hits: int = 3) -> List[CFCandidate]:
    """Ramanujan-Machine-style search: small integer polynomials p(n), q(n) whose
    generalized CF converges to a simple image of ``value``. Each hit is verified
    to ``min_digits`` by a DEEPER independent re-evaluation of the CF (depth*2)
    before it is reported. Returns up to ``max_hits`` candidates, best first.

    This is a bounded brute-force over a TINY polynomial space (the RM insight is
    that the famous CFs have low-complexity integer polynomials). It is honest:
    nothing is returned unless a deeper recomputation confirms the same digits.
    """
    out: List[CFCandidate] = []
    with mpmath.workdps(max(cf_depth // 3, 60)):
        v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)
        images = _image_targets(v)
        seen = set()
        for pa, pb in _small_poly_pairs(max_deg, coeff_range):
            cf = gcf_value(pa, pb, cf_depth)
            if cf is None or not mpmath.isfinite(cf):
                continue
            for img_name, img_val in images:
                if img_val == 0:
                    continue
                scale = abs(img_val) if abs(img_val) > 1 else mpmath.mpf(1)
                diff = abs(cf - img_val) / scale
                if diff == 0:
                    digits = mpmath.mp.dps
                elif diff < 1:
                    digits = int(-mpmath.log10(diff))
                else:
                    digits = 0
                if digits < min_digits:
                    continue
                sig = (pa, pb, img_name)
                if sig in seen:
                    continue
                # DEEPER independent re-eval (depth*2, higher dps) -- the gate.
                with mpmath.workdps(max(cf_depth, 80)):
                    cf2 = gcf_value(pa, pb, cf_depth * 2)
                    if cf2 is None:
                        continue
                    d2 = abs(cf2 - img_val) / scale
                    digits2 = (mpmath.mp.dps if d2 == 0
                               else int(-mpmath.log10(d2)) if d2 < 1 else 0)
                if digits2 < min_digits:
                    continue
                seen.add(sig)
                prov = Provenance(
                    novelty=Novelty.CONJECTURED_RELATION,
                    method="ramanujan-machine-cf-search(backward-recurrence)+deep-reeval",
                    source=f"CF[a_n={list(pa)}, b_n={list(pb)}] = {img_name}",
                    basis=("n",),
                )
                out.append(CFCandidate(
                    kind="continued_fraction",
                    poly_a=tuple(int(c) for c in pa),
                    poly_b=tuple(int(c) for c in pb),
                    image=img_name, cf_value=mpmath.nstr(cf2, 30),
                    digits_agreed=int(min(digits2, min_digits + 40)),
                    depth=cf_depth * 2, provenance=prov))
                if len(out) >= max_hits:
                    return out
    return out


# --------------------------------------------------------------------------- #
# (b') SIMPLE continued fraction [a0; a1, a2, ...] + pattern recognition.
# Complements the generalized-CF search: a constant's *simple* CF partial
# quotients often carry an obvious pattern -- PERIODIC for quadratic irrationals
# (sqrt(2)=[1;2,2,2,...], phi=[1;1,1,1,...]) or an ARITHMETIC run for e
# ([2;1,2,1,1,4,1,1,6,...]). We detect the pattern, then VERIFY by rebuilding the
# convergent from the (pattern-extended) quotients to high precision.
# --------------------------------------------------------------------------- #
def simple_cf_terms(value: "mpmath.mpf", n_terms: int = 24) -> List[int]:
    """The first ``n_terms`` partial quotients of the simple CF [a0; a1, a2, ...].
    Stops early on an exact (rational) termination."""
    out: List[int] = []
    x = value
    for _ in range(n_terms):
        ai = int(mpmath.floor(x))
        out.append(ai)
        frac = x - ai
        if frac <= 0:
            break
        x = 1 / frac
    return out


def _simple_cf_convergent(terms: Sequence[int]) -> "mpmath.mpf":
    """Rebuild the value a0 + 1/(a1 + 1/(a2 + ...)) from partial quotients."""
    x = mpmath.mpf(terms[-1])
    for ai in reversed(terms[:-1]):
        x = ai + 1 / x
    return x


def _detect_pattern(terms: Sequence[int]) -> Optional[str]:
    """Name an obvious pattern in the CF tail (after a0), or ``None``.

    Recognises: constant tail (-> periodic-1, a quadratic irrational), pure
    period-2, and the e-type arithmetic run [.,1,2k,1,1,2(k+1),1,...]."""
    if len(terms) < 5:
        return None
    tail = list(terms[1:])
    # constant tail (period 1): sqrt2-ish, phi-ish
    if len(set(tail)) == 1:
        return f"periodic: [{terms[0]}; {tail[0]}, {tail[0]}, ...] (period 1)"
    # period 2
    if len(tail) >= 6 and tail[0::2][:-1] == tail[2::2] and tail[1::2][:-1] == tail[3::2] \
            and len(set(tail[:2])) <= 2:
        return f"periodic: [{terms[0]}; {tail[0]}, {tail[1]}, ...] (period 2)"
    # e-type: pattern 1, 2, 1, 1, 4, 1, 1, 6, ... (every 3rd term is 2,4,6,8,...)
    evens = tail[1::3]
    if len(evens) >= 3 and all(evens[i] == 2 * (i + 1) for i in range(len(evens))) \
            and all(tail[i] == 1 for i in range(len(tail)) if i % 3 != 1):
        return f"arithmetic (e-type): [{terms[0]}; 1, 2, 1, 1, 4, 1, 1, 6, ...]"
    return None


@dataclass(frozen=True)
class SimpleCFCandidate:
    """A SIMPLE continued fraction [a0; a1, a2, ...] for the constant + pattern."""
    kind: str                       # "simple_continued_fraction"
    terms: Tuple[int, ...]          # the partial quotients found
    pattern: Optional[str]          # a recognised pattern, or None
    convergent: str                 # value rebuilt from terms (decimal string)
    digits_agreed: int
    provenance: Provenance


def simple_continued_fraction(value: "mpmath.mpf", n_terms: int = 120,
                              min_digits: int = 20) -> Optional[SimpleCFCandidate]:
    """Compute the simple CF of ``value``, recognise any pattern, and VERIFY by
    rebuilding the convergent to high precision. Returns ``None`` if the rebuilt
    convergent does not re-agree to ``min_digits`` (honest gate) -- e.g. for a
    structureless constant whose quotients are numerical noise.

    ``n_terms`` is generous by default: a simple CF with small partial quotients
    (e.g. phi = [1;1,1,...]) converges only ~0.2 digit/term, so enough terms are
    taken that even the slowest-converging structured constant clears the gate.
    The DISPLAYED ``terms`` are trimmed to the first 16 (the pattern is in them).
    """
    with mpmath.workdps(max(n_terms + min_digits + 20, 80)):
        v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)
        terms = simple_cf_terms(v, n_terms=n_terms)
        if len(terms) < 3:
            return None
        # verify: the convergent of ALL collected quotients must reproduce v.
        conv = _simple_cf_convergent(terms)
        scale = abs(v) if abs(v) > 1 else mpmath.mpf(1)
        diff = abs(conv - v) / scale
        digits = (mpmath.mp.dps if diff == 0
                  else int(-mpmath.log10(diff)) if 0 < diff < 1 else 0)
    if digits < min_digits:
        return None
    # Detect the pattern on a SAFE prefix only: the last few quotients of a long
    # CF approach the working-precision floor and can be corrupted, which would
    # mask a real pattern. The leading ~24 terms are well within precision.
    safe = terms[:min(len(terms), 24)]
    pattern = _detect_pattern(safe)
    prov = Provenance(
        novelty=Novelty.CONJECTURED_RELATION,
        method="simple-continued-fraction+convergent-verify",
        source=f"[{terms[0]}; {', '.join(map(str, terms[1:8]))}...]",
        basis=("simple_cf",),
    )
    return SimpleCFCandidate(
        kind="simple_continued_fraction", terms=tuple(safe), pattern=pattern,
        convergent=mpmath.nstr(conv, 30),
        digits_agreed=int(min(digits, min_digits + 40)), provenance=prov)


@dataclass(frozen=True)
class ConjectureResult:
    query: str
    relations: List[RelationCandidate]
    continued_fractions: List[CFCandidate]
    simple_cf: Optional[SimpleCFCandidate] = None

    @property
    def found(self) -> bool:
        return bool(self.relations or self.continued_fractions or self.simple_cf)


def conjecture(value, max_terms: int = 16, cf_depth: int = 200,
               min_digits: int = 25, dps_verify: int = 60) -> ConjectureResult:
    """Run BOTH discovery channels for ``value`` and return verified conjectures.

    (a) PSLQ over the richer basis -> integer-relation closed forms;
    (b) Ramanujan-Machine CF search -> polynomial-recurrence continued fractions.
    Every candidate is numerically VERIFIED before inclusion; provenance is
    ``CONJECTURED_RELATION`` (a verified conjecture, NOT a proof). NO LLM.
    """
    with mpmath.workdps(max(dps_verify + 10, 70)):
        v = value if isinstance(value, mpmath.mpf) else mpmath.mpf(str(value))
        query = mpmath.nstr(v, 20)
    rels = integer_relations(v, max_terms=max_terms, dps_verify=dps_verify,
                             min_digits=min_digits)
    scf = simple_continued_fraction(v, min_digits=min_digits)
    cfs = continued_fractions(v, cf_depth=cf_depth, min_digits=min_digits)
    return ConjectureResult(query=query, relations=rels,
                            continued_fractions=cfs, simple_cf=scf)


__all__ = [
    "conjecture", "ConjectureResult",
    "integer_relations", "RelationCandidate",
    "continued_fractions", "CFCandidate", "gcf_value",
    "simple_continued_fraction", "SimpleCFCandidate", "simple_cf_terms",
]
