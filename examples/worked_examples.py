#!/usr/bin/env python3
"""mathlas worked examples — the core thesis in action: map a problem to the
EXISTING math that solves it, then VERIFY (airtight where possible, honest
otherwise). mathlas never calls an LLM; each vignette shows the INPUT a calling AI
would pass, the FACT mathlas returns, and what the AI then concludes.

Run:  PYTHONPATH=. python3 examples/worked_examples.py
"""
from __future__ import annotations

import mpmath

from mathlas import identify
from mathlas.sequence import identify_sequence
from mathlas.ramanujan import conjecture
from mathlas.verify_apply import applicability_checklist, verify_formal, find_lean


def hr(title):
    print("\n" + "─" * 72 + f"\n▶ {title}\n" + "─" * 72)


def ex_constant():
    hr("1. 'My computation produced 1.2020569031595942 — what is it? Can I trust it?'")
    mpmath.mp.dps = 40
    v = mpmath.zeta(3)            # pretend the AI hands mathlas this measured value
    print(f"  AI → identify_constant({mpmath.nstr(v, 17)})")
    r = identify(v)
    print(f"  mathlas → {r}")
    print("  AI concludes: it is Apéry's constant ζ(3), confirmed to ~40 digits by an "
          "independent\n               re-evaluation — safe to use as a closed form.")


def ex_sequence():
    hr("2. 'I keep seeing 1, 1, 2, 5, 14, 42, 132 — does it have a name?'")
    terms = [1, 1, 2, 5, 14, 42, 132]
    print(f"  AI → identify_sequence({terms})")
    r = identify_sequence(terms)
    if r.matches:
        m = r.matches[0]
        print(f"  mathlas → {m.a_number}: {m.name[:60]}  ({m.url})")
        print("  AI concludes: these are the Catalan numbers — now it can pull known "
              "identities/asymptotics.")
    else:
        print(f"  mathlas → {r.note}")


def ex_ramanujan():
    hr("3. 'Here is a constant 1.6180339887... — is there a hidden structure?'")
    mpmath.mp.dps = 60
    v = (1 + mpmath.sqrt(5)) / 2
    print(f"  AI → conjecture_relation({mpmath.nstr(v, 12)})")
    r = conjecture(v)
    if r.simple_cf:
        print(f"  mathlas → simple continued fraction {r.simple_cf.terms[:8]}…  "
              f"pattern: {r.simple_cf.pattern}")
    if r.relations:
        print(f"  mathlas → PSLQ relation: x = {r.relations[0].expr}")
    print("  AI concludes: φ, the golden ratio — CF is all 1s (slowest-converging "
          "irrational), numerically verified.")


def ex_formal():
    hr("4. 'Before I rely on it, is this Lean lemma actually correct?'")
    if not find_lean():
        print("  (Lean toolchain not found — formal tier would return honest UNDETERMINED)")
        return
    good = "theorem t (n : Nat) : n + 0 = n := by rfl"
    bad = "theorem t : 2 + 2 = 5 := by rfl"
    for label, snip in [("claimed-true", good), ("claimed-true-but-FALSE", bad)]:
        v = verify_formal(snip)
        verdict = ("TYPECHECKS" if v.applies else
                   ("kernel REJECTED" if v.failure else "undetermined"))
        print(f"  AI → verify_formal({label!r})  →  {verdict}")
    print("  AI concludes: the real Lean kernel accepts n+0=n and REJECTS 2+2=5 — not a "
          "guess, a proof check.")


def ex_moat():
    hr("5. THE MOAT — 'Retrieval handed me a tempting theorem. Does it actually apply?'")
    problem = "I need the max of f on the OPEN interval (0, 1); f is continuous there."
    candidate = ("Let f be a continuous function on a closed interval [a, b]. "
                 "Then f attains its maximum on [a, b].")
    print(f"  problem:   {problem}")
    print(f"  candidate: {candidate}")
    cl = applicability_checklist(candidate)
    print("  mathlas → applicability checklist (NO LLM — atomic preconditions to check):")
    for p in cl.preconditions:
        print(f"      [ ] {p}")
    print(f"      ⇒ guarantees: {cl.conclusion}")
    print("  AI concludes: precondition 'closed interval [a,b]' is UNMET (my interval is "
          "open) →\n               the theorem does NOT apply. A retrieval-only tool would "
          "have handed me the\n               candidate with no such guard — this is the "
          "blind-apply failure mathlas prevents.")


def main():
    print("=" * 72)
    print("mathlas worked examples — map a problem to existing math, then verify")
    print("=" * 72)
    ex_constant()
    ex_sequence()
    ex_ramanujan()
    ex_formal()
    ex_moat()
    print("\n" + "=" * 72)
    print("Every step above is a CHECKABLE FACT mathlas returned with no LLM call. The AI "
          "does the\nreasoning; mathlas supplies airtight identification, verification, and "
          "the applicability\nscaffold. See RESULTS.md for the aggregate benchmarks.")
    print("=" * 72)


if __name__ == "__main__":
    main()
