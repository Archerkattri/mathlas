#!/usr/bin/env python3
"""Per-tier airtight benchmarks for mathlas: SEQUENCE / FORMAL / RAMANUJAN.

Companion to ``benchmarks/numeric_bench.py`` (the constant tier). Every tier
follows the same airtight-or-nothing discipline -- a result is a real,
independently-checkable fact, never a plausible guess:

  recovery@known        -- feed a KNOWN input, expect the correct, verified result.
  false-positive@random -- feed a STRUCTURELESS input, expect an honest "nothing"
                           (the honesty gate; a tool that fakes a hit here is worse
                           than useless -- the GPT-5 over-claim failure mode).

Run:   PYTHONPATH=. python3 benchmarks/tier_bench.py
Deps:  OEIS data in reference/downloads/oeis (stripped.gz, names.gz) for SEQUENCE;
       a Lean toolchain (reference/downloads/elan) for FORMAL. A missing dep makes
       that tier print SKIP (honest), never a fake pass. RAMANUJAN is pure mpmath.
"""
from __future__ import annotations

import time

import mpmath

from mathlas.sequence import identify_sequence
from mathlas.verify_apply import verify_formal, find_lean
from mathlas.ramanujan import conjecture


# ========================================================================== #
# SEQUENCE TIER -- OEIS exact contiguous term-match.
# ========================================================================== #
# (label, terms, expected A-number) -- canonical sequences with known A-numbers.
SEQ_KNOWN = [
    ("Fibonacci",   [1, 1, 2, 3, 5, 8, 13, 21, 34],   "A000045"),
    ("primes",      [2, 3, 5, 7, 11, 13, 17, 19],     "A000040"),
    ("Catalan",     [1, 1, 2, 5, 14, 42, 132, 429],   "A000108"),
    ("squares",     [0, 1, 4, 9, 16, 25, 36, 49],     "A000290"),
    ("factorials",  [1, 1, 2, 6, 24, 120, 720, 5040], "A000142"),
    ("triangular",  [0, 1, 3, 6, 10, 15, 21, 28],     "A000217"),
    ("powers of 2", [1, 2, 4, 8, 16, 32, 64, 128],    "A000079"),
    ("Bell",        [1, 1, 2, 5, 15, 52, 203, 877],   "A000110"),
]
# Structureless integer runs -- must NOT appear as a contiguous run in any OEIS seq.
SEQ_NEG = [
    ("random-1", [83, 1000003, 57, 90210, 1234, 6]),
    ("random-2", [104729, 3, 88, 700001, 42, 9]),
    ("random-3", [500009, 7, 313, 90001, 11, 65535]),
]


def bench_sequence():
    print("\n=== SEQUENCE TIER (OEIS exact term-match) ===")
    probe = identify_sequence([1, 1, 2, 3, 5, 8])
    if probe.data_dir is None and not probe.matches:
        print(f"  SKIP: OEIS data unavailable (honest skip). {probe.note}")
        return None
    rec = 0
    for label, terms, want in SEQ_KNOWN:
        r = identify_sequence(terms)
        anums = [m.a_number for m in r.matches]
        hit = want in anums
        rec += hit
        rank = f"top-{anums.index(want) + 1}" if hit else "NOT FOUND"
        print(f"  [{'OK  ' if hit else 'MISS'}] {label:12s} want {want} "
              f"got {anums[0] if anums else '-':8s} ({rank})")
    fp = 0
    for label, terms in SEQ_NEG:
        r = identify_sequence(terms)
        fp += r.identified
        got = r.matches[0].a_number if r.identified else "UNIDENTIFIED"
        print(f"  [{'FALSEPOS' if r.identified else 'clean   '}] {label:12s} -> {got}")
    nk, nn = len(SEQ_KNOWN), len(SEQ_NEG)
    print(f"  -> recovery {rec}/{nk} ({100 * rec / nk:.0f}%)  "
          f"false-pos {fp}/{nn} ({100 * fp / nn:.0f}%)   [DoD rec>=90%, fp<=5%]")
    return ("sequence", rec, nk, fp, nn)


# ========================================================================== #
# FORMAL TIER -- REAL Lean 4 kernel typecheck (no mathlib, core only).
# ========================================================================== #
LEAN_TRUE = [
    ("2+2=4",      "theorem t : 2 + 2 = 4 := by rfl"),
    ("n+0=n",      "theorem t (n : Nat) : n + 0 = n := by rfl"),
    ("not not b",  "theorem t (b : Bool) : (!!b) = b := by cases b <;> rfl"),
    ("True",       "theorem t : True := True.intro"),
]
LEAN_FALSE = [
    ("2+2=5",      "theorem t : 2 + 2 = 5 := by rfl"),
    ("1=0 : Nat",  "theorem t : (1 : Nat) = 0 := by rfl"),
    ("type error", 'theorem t : Nat := "hello"'),
]


def bench_formal():
    print("\n=== FORMAL TIER (real Lean kernel check) ===")
    exe = find_lean()
    if not exe:
        print("  SKIP: no Lean toolchain found (honest skip, not a fake pass)")
        return None
    print(f"  (lean: {exe})")
    correct = total = 0
    for label, snip in LEAN_TRUE:
        v = verify_formal(snip)
        ok = v.applies is True            # genuinely typechecks
        correct += ok; total += 1
        print(f"  [{'OK  ' if ok else 'MISS'}] TRUE  {label:12s} -> applies={v.applies}")
    for label, snip in LEAN_FALSE:
        v = verify_formal(snip)
        # correct iff REJECTED because Lean reported errors (not 'undetermined').
        rej = v.applies is False and v.failure is not None
        correct += rej; total += 1
        verdict = "rejected" if rej else ("UNDET" if v.failure is None else "accepted?!")
        print(f"  [{'OK  ' if rej else 'MISS'}] FALSE {label:12s} -> {verdict}")
    print(f"  -> {correct}/{total} correct verdicts   [DoD 100% -- the kernel is exact]")
    return ("formal", correct, total, 0, 0)


# ========================================================================== #
# RAMANUJAN TIER -- numerically-verified relation / CF conjecture.
# ========================================================================== #
RAMA_KNOWN = [
    ("golden ratio", lambda: (1 + mpmath.sqrt(5)) / 2),
    ("sqrt(2)",      lambda: mpmath.sqrt(2)),
    ("e",            lambda: mpmath.e),
    ("pi",           lambda: mpmath.pi),
    ("Catalan",      lambda: mpmath.catalan),
    ("zeta(3)",      lambda: mpmath.zeta(3)),
]
# Structureless constants -- expect NO verified relation and NO patterned CF.
RAMA_NEG = [
    ("sin(1)*log(7)", lambda: mpmath.sin(1) * mpmath.log(7)),
    ("exp(sin(2))",   lambda: mpmath.exp(mpmath.sin(2))),
]


def _kinds(r):
    out = []
    if r.relations:
        out.append(f"{len(r.relations)} PSLQ-rel")
    if r.simple_cf:
        out.append("simpleCF" + (f"<{r.simple_cf.pattern}>" if r.simple_cf.pattern else ""))
    if r.continued_fractions:
        out.append(f"{len(r.continued_fractions)} genCF")
    return ", ".join(out) or "nothing"


def bench_ramanujan():
    print("\n=== RAMANUJAN TIER (numerically-verified conjecture) ===")
    mpmath.mp.dps = 80
    rec = 0
    for label, gen in RAMA_KNOWN:
        t0 = time.time()
        r = conjecture(gen())
        rec += r.found
        print(f"  [{'OK  ' if r.found else 'MISS'}] {label:13s} -> {_kinds(r)}"
              f"  ({time.time() - t0:.1f}s)")
    fp = 0
    for label, gen in RAMA_NEG:
        r = conjecture(gen())
        # honesty gate: a structureless constant must yield no PSLQ relation, no
        # generalized-CF hit, and no *patterned* simple CF (a bare CF is fine --
        # it is the representation, not a claim).
        false_pos = bool(r.relations) or bool(r.continued_fractions) or \
            (r.simple_cf is not None and r.simple_cf.pattern is not None)
        fp += false_pos
        print(f"  [{'FALSEPOS' if false_pos else 'clean   '}] {label:13s} -> {_kinds(r)}")
    nk, nn = len(RAMA_KNOWN), len(RAMA_NEG)
    print(f"  -> recovery {rec}/{nk} ({100 * rec / nk:.0f}%)  "
          f"false-pos {fp}/{nn} ({100 * fp / nn:.0f}%)   [DoD rec>=80%, fp=0]")
    return ("ramanujan", rec, nk, fp, nn)


def main():
    print("=" * 60)
    print("mathlas per-tier airtight benchmark (sequence / formal / ramanujan)")
    print("=" * 60)
    results = [bench_sequence(), bench_formal(), bench_ramanujan()]
    print("\n" + "=" * 60)
    print("SUMMARY")
    for r in results:
        if r is None:
            continue
        if r[0] == "formal":
            _, correct, total, _, _ = r
            print(f"  {r[0]:10s}: {correct}/{total} correct verdicts")
        else:
            name, rec, nk, fp, nn = r
            print(f"  {name:10s}: recovery {rec}/{nk}  false-pos {fp}/{nn}")
    print("=" * 60)


if __name__ == "__main__":
    main()
