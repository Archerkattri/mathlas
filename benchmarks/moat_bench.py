#!/usr/bin/env python3
"""Applicability-MOAT benchmark -- the piece a retrieval-only tool (TheoremSearch)
structurally lacks.

mathlas does NOT judge applicability (no LLM); it DECOMPOSES a candidate result
into atomic, individually-checkable preconditions for the CALLING AI. This
benchmark measures the part mathlas owns deterministically, two ways:

  (A) DECOMPOSITION recall -- on labeled theorems with known hypotheses, does
      ``applicability_checklist`` surface each required precondition (by keyword
      coverage)? This is the scaffold the AI marks against its problem.
  (B) MISAPPLICATION-CATCH (the moat) -- for (problem, tempting-but-wrong
      candidate) pairs where the problem violates exactly ONE hypothesis, does the
      checklist surface a precondition naming the violated hypothesis? If yes, the
      scaffold ENABLES the AI to reject the misapplication -- the GPT-5 blind-apply
      failure mode. A retrieval-only tool returns the tempting candidate with NO
      such guard.

HONEST SCOPE: this validates that the scaffold provides the necessary atomic
conditions (the deterministic generator half of the generator/verifier split).
The final applies/doesn't-apply JUDGMENT is the calling AI's job, by design --
not measured here. NO LLM, no fabricated verdicts; matching is transparent keyword
grounding.
"""
from __future__ import annotations

from mathlas.verify_apply import applicability_checklist


# (label, statement, [expected-precondition keyword-sets]) -- each expected
# precondition is a set of keywords; it is "covered" iff SOME extracted
# precondition contains ALL of them (case-insensitive substring).
DECOMP = [
    ("Extreme Value Thm",
     "Let f be a continuous function on a closed interval [a, b]. "
     "Then f attains its maximum on [a, b].",
     [{"continuous"}, {"closed", "interval"}]),
    ("Cauchy's theorem",
     "If G is a finite group and p is a prime dividing the order of G, "
     "then G has an element of order p.",
     [{"finite", "group"}, {"prime"}, {"dividing", "order"}]),
    ("compact Hausdorff => normal",
     "Suppose X is a compact Hausdorff space. Then X is normal.",
     [{"compact"}, {"hausdorff"}]),
    ("Fermat's little theorem",
     "Let p be a prime and a an integer not divisible by p. "
     "Then a^(p-1) is congruent to 1 modulo p.",
     [{"prime"}, {"not divisible"}]),
    ("monotone convergence",
     "If a sequence is monotone and bounded, then it converges.",
     [{"monotone"}, {"bounded"}]),
    ("mean value theorem",
     "Let f be continuous on [a, b] and differentiable on (a, b). "
     "Then there exists c in (a, b) with f'(c) equal to the average rate of change.",
     [{"continuous"}, {"differentiable"}]),
    ("Banach fixed point",
     "Let T be a contraction on a complete metric space X. "
     "Then T has a unique fixed point.",
     [{"contraction"}, {"complete", "metric"}]),
]

# (problem, tempting-but-wrong candidate, violated-hypothesis keyword) -- the
# problem violates exactly that hypothesis; a useful checklist MUST surface a
# precondition containing the keyword so the AI can catch the misapplication.
TRAPS = [
    ("f is defined on the OPEN interval (a, b), not a closed one",
     "Let f be a continuous function on a closed interval [a, b]. "
     "Then f attains its maximum.",
     "closed"),
    ("G is an INFINITE group and p a prime",
     "If G is a finite group and p is a prime dividing the order of G, "
     "then G has an element of order p.",
     "finite"),
    ("X is compact but NOT Hausdorff",
     "Suppose X is a compact Hausdorff space. Then X is normal.",
     "hausdorff"),
    ("the sequence is bounded but NOT monotone",
     "If a sequence is monotone and bounded, then it converges.",
     "monotone"),
    ("f is continuous but NOT differentiable on (a, b)",
     "Let f be continuous on [a, b] and differentiable on (a, b). "
     "Then there exists c with f'(c) = 0.",
     "differentiable"),
    ("the metric space is NOT complete",
     "Let T be a contraction on a complete metric space X. "
     "Then T has a unique fixed point.",
     "complete"),
]


def _covered(keywords, preconds):
    low = [p.lower() for p in preconds]
    return any(all(k.lower() in p for k in keywords) for p in low)


def bench_decomposition():
    print("\n=== (A) DECOMPOSITION recall (does the scaffold surface each hypothesis?) ===")
    total_exp = total_cov = 0
    for label, stmt, expected in DECOMP:
        cl = applicability_checklist(stmt)
        covered = [e for e in expected if _covered(e, cl.preconditions)]
        total_exp += len(expected)
        total_cov += len(covered)
        print(f"  {label:26s}: {len(covered)}/{len(expected)} hyps covered "
              f"({len(cl.preconditions)} preconds extracted)")
        for e in expected:
            if e not in covered:
                print(f"       MISSED hyp keywords: {e}")
    print(f"  -> recall {total_cov}/{total_exp} = {100 * total_cov / total_exp:.0f}%   "
          "[DoD >= 90%]")
    return total_cov, total_exp


def bench_traps():
    print("\n=== (B) MISAPPLICATION-CATCH (the moat: surface the VIOLATED hypothesis) ===")
    caught = 0
    for problem, candidate, violated in TRAPS:
        cl = applicability_checklist(candidate)
        surfaced = any(violated.lower() in p.lower() for p in cl.preconditions)
        caught += surfaced
        print(f"  [{'CATCH' if surfaced else 'MISS '}] {problem[:46]:46s} "
              f"| '{violated}' {'in' if surfaced else 'NOT in'} checklist")
    print(f"  -> caught {caught}/{len(TRAPS)} = {100 * caught / len(TRAPS):.0f}%   "
          "[DoD = 100%: every violated hyp must be checkable]")
    return caught, len(TRAPS)


def main():
    print("=" * 64)
    print("mathlas applicability-MOAT benchmark (decomposition + misapplication)")
    print("=" * 64)
    cov, exp = bench_decomposition()
    caught, ntrap = bench_traps()
    print("\n" + "=" * 64)
    print(f"SUMMARY: decomposition recall {cov}/{exp} = {100 * cov / exp:.0f}%   "
          f"misapplication-catch {caught}/{ntrap} = {100 * caught / ntrap:.0f}%")
    print("NOTE: this validates the SCAFFOLD (the deterministic half). The final "
          "applies/not-applies\n      judgment is the calling AI's job, by design.")
    print("=" * 64)


if __name__ == "__main__":
    main()
