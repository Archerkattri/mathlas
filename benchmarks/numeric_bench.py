"""Numeric-beachhead benchmark: recovery@known + false-positive@random.

  recovery@known      -- feed the high-precision value of a KNOWN constant,
                         expect a verified closed form back.
  false-positive@random -- feed full-precision STRUCTURELESS irrationals
                         (not expressible in the basis), expect UNIDENTIFIED.
                         This is the honesty gate the GPT-5 episode lacked.

Definition of Done: recovery >= 90%, false-positive <= 5%.
"""
from __future__ import annotations

import mpmath

from mathlas import identify

DPS = 60

# (label, high-precision value) -- known closed forms the engine should recover.
KNOWN = [
    ("zeta(2) = pi^2/6", lambda: mpmath.zeta(2)),
    ("zeta(3) (Apery)", lambda: mpmath.zeta(3)),
    ("Catalan", lambda: mpmath.catalan),
    ("golden ratio", lambda: (1 + mpmath.sqrt(5)) / 2),
    ("log(2)", lambda: mpmath.log(2)),
    ("pi^2/6 (direct)", lambda: mpmath.pi ** 2 / 6),
    ("e", lambda: mpmath.e),
    ("sqrt(2) + sqrt(3)", lambda: mpmath.sqrt(2) + mpmath.sqrt(3)),
]

# Full-precision irrationals NOT in the basis -- expect no verified closed form.
# (Deliberately constructed, not typed decimals, to avoid trivial rational fits.)
NEGATIVE = [
    ("sin(1) * log(7)", lambda: mpmath.sin(1) * mpmath.log(7)),
    ("tan(2) + 1/3", lambda: mpmath.tan(2) + mpmath.mpf(1) / 3),
    ("exp(sin(2))", lambda: mpmath.exp(mpmath.sin(2))),
]


def run() -> None:
    mpmath.mp.dps = DPS

    print("== recovery@known ==")
    rec_hit = 0
    for label, gen in KNOWN:
        r = identify(gen())
        rec_hit += r.identified
        print(f"  [{'OK  ' if r.identified else 'MISS'}] {label}: {r}")

    print("== false-positive@random ==")
    fp = 0
    for label, gen in NEGATIVE:
        r = identify(gen())
        fp += r.identified
        print(f"  [{'FALSEPOS' if r.identified else 'clean   '}] {label}: {r}")

    nk, nn = len(KNOWN), len(NEGATIVE)
    print(f"\nrecovery {rec_hit}/{nk} ({100 * rec_hit / nk:.0f}%)  "
          f"false-positive {fp}/{nn} ({100 * fp / nn:.0f}%)")
    print("DoD: recovery >= 90%, false-positive <= 5%")


if __name__ == "__main__":
    run()
