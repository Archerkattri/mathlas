#!/usr/bin/env python3
"""Evaluate mathlas.formal_search.search_formal_math against a 25-query gold set.

Gold set: 25 natural-language / Loogle-pattern math queries whose canonical
mathlib4 declaration is unambiguous and well known. Every gold name (and every
accepted alternate) was verified to exist in mathlib4 on the build date by exact
lookup in the official doc-gen4 declaration index
(https://leanprover-community.github.io/mathlib4_docs/declarations/declaration-data.bmp,
410,507 declarations) — NOT by trusting the search services under test.

Scoring: Hit@1 / Hit@5 / Hit@10 per backend (loogle, leansearch) and for the
combined interleaved+deduped list that ``search_formal_math(backend="auto")``
returns.

MATCHING RULES (explicit):
  A returned hit ``h`` matches the gold entry iff for SOME accepted name ``g``
  (the canonical name or a listed alternate):
    1. h == g                       (exact), or
    2. h endswith "." + g           (g re-exported / specialized under an extra
                                     namespace, e.g. ``Nat.mul_comm`` for gold
                                     ``mul_comm``), or
    3. g endswith "." + h           (hit is the root-namespace variant of an
                                     accepted namespaced name).
  Nothing else matches. In particular primed variants (``foo'``), ``_iff``
  forms, and converse/sibling lemmas do NOT count unless explicitly listed as
  alternates. Matching is case-sensitive.

Backend honesty: Loogle is probed once up front (2 attempts). If it is down the
run proceeds LeanSearch-only, Loogle is reported ``available: False`` with the
real error, and no Loogle numbers are fabricated. Per-query calls are spaced
>= 1 s apart (polite rate limiting of the public services).

Usage (CPU + network only; no GPU):
  OMP_NUM_THREADS=2 python scripts/eval_formal_search.py [--k 10] [--json out.json]

Prints a markdown report (tables + per-query results) to stdout.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")  # run from repo root

from mathlas.formal_search import (  # noqa: E402
    search_formal_math,
    search_loogle,
)

RATE_LIMIT_S = 1.0  # minimum spacing between network calls
KS = (1, 5, 10)

# ---------------------------------------------------------------------------
# Gold set: 25 queries. "gold" lists ALL accepted declaration names; the first
# is the canonical one. Every name verified to exist in mathlib4 (see module
# docstring). kind: "nl" = natural language (20), "pattern" = Loogle-style
# type/pattern query (5).
# ---------------------------------------------------------------------------
GOLD: List[Dict[str, Any]] = [
    # -------------------------- analysis (7 NL) ---------------------------
    dict(id="A1", area="analysis", kind="nl",
         query="a bounded monotone sequence converges",
         gold=["tendsto_atTop_ciSup", "tendsto_atTop_iSup",
               "tendsto_of_monotone"]),
    dict(id="A2", area="analysis", kind="nl",
         query="intermediate value theorem for a continuous function on a closed interval",
         gold=["intermediate_value_Icc", "intermediate_value_Icc'",
               "intermediate_value_Ioo", "intermediate_value_uIcc"]),
    dict(id="A3", area="analysis", kind="nl",
         query="mean value theorem: there is a point where the derivative equals the slope of the secant line",
         gold=["exists_deriv_eq_slope", "exists_hasDerivAt_eq_slope"]),
    dict(id="A4", area="analysis", kind="nl",
         query="squeeze theorem: a sequence trapped between two sequences with the same limit converges",
         gold=["tendsto_of_tendsto_of_tendsto_of_le_of_le",
               "tendsto_of_tendsto_of_tendsto_of_le_of_le'"]),
    dict(id="A5", area="analysis", kind="nl",
         query="the derivative of the sine function is cosine",
         gold=["Real.deriv_sin", "Real.hasDerivAt_sin", "Complex.deriv_sin"]),
    dict(id="A6", area="analysis", kind="nl",
         query="Cauchy-Schwarz inequality: the norm of the inner product is at most the product of the norms",
         gold=["norm_inner_le_norm", "abs_real_inner_le_norm",
               "InnerProductSpace.Core.norm_inner_le_norm",
               "nnnorm_inner_le_nnnorm"]),
    dict(id="A7", area="analysis", kind="nl",
         query="every Cauchy sequence in a complete metric space converges",
         gold=["cauchySeq_tendsto_of_complete",
               "cauchySeq_tendsto_of_isComplete"]),
    # --------------------------- algebra (5 NL) ---------------------------
    dict(id="G1", area="algebra", kind="nl",
         query="Lagrange's theorem: the order of a subgroup divides the order of the group",
         gold=["Subgroup.card_subgroup_dvd_card"]),
    dict(id="G2", area="algebra", kind="nl",
         query="the order of an element of a finite group divides the order of the group",
         gold=["orderOf_dvd_card"]),
    dict(id="G3", area="algebra", kind="nl",
         query="first isomorphism theorem: a group modulo the kernel of a homomorphism is isomorphic to the range",
         gold=["QuotientGroup.quotientKerEquivRange"]),
    dict(id="G4", area="algebra", kind="nl",
         query="the inverse of a product is the product of the inverses in reverse order",
         gold=["mul_inv_rev"]),
    dict(id="G5", area="algebra", kind="nl",
         query="binomial theorem: the expansion of (x + y)^n as a sum of binomial coefficients",
         gold=["add_pow", "Commute.add_pow"]),
    # -------------------------- topology (4 NL) ---------------------------
    dict(id="T1", area="topology", kind="nl",
         query="the continuous image of a compact set is compact",
         gold=["IsCompact.image"]),
    dict(id="T2", area="topology", kind="nl",
         query="a closed subset of a compact space is compact",
         gold=["IsClosed.isCompact", "IsCompact.of_isClosed_subset"]),
    dict(id="T3", area="topology", kind="nl",
         query="a compact subset of a Hausdorff space is closed",
         gold=["IsCompact.isClosed"]),
    dict(id="T4", area="topology", kind="nl",
         query="Heine-Borel theorem: a set in a proper metric space is compact if and only if it is closed and bounded",
         gold=["Metric.isCompact_iff_isClosed_bounded"]),
    # ------------------------ number theory (4 NL) ------------------------
    dict(id="N1", area="number_theory", kind="nl",
         query="there are infinitely many prime numbers",
         gold=["Nat.exists_infinite_primes"]),
    dict(id="N2", area="number_theory", kind="nl",
         query="Fermat's little theorem: a to the power p minus one is congruent to one modulo a prime p",
         gold=["ZMod.pow_card_sub_one_eq_one", "ZMod.pow_card"]),
    dict(id="N3", area="number_theory", kind="nl",
         query="the square root of two is irrational",
         gold=["irrational_sqrt_two"]),
    dict(id="N4", area="number_theory", kind="nl",
         query="Bezout's identity: the gcd of two integers is a linear combination of them",
         gold=["Nat.gcd_eq_gcd_ab", "Int.gcd_eq_gcd_ab"]),
    # ------------------ Loogle-style pattern queries (5) ------------------
    dict(id="L1", area="algebra", kind="pattern",
         query="?a * ?b = ?b * ?a",
         gold=["mul_comm", "CommMonoid.mul_comm"]),
    dict(id="L2", area="analysis", kind="pattern",
         query="Real.sqrt ?a * Real.sqrt ?a",
         gold=["Real.mul_self_sqrt", "Real.sqrt_mul_self"]),
    dict(id="L3", area="data_structures", kind="pattern",
         query="List.length (?a ++ ?b)",
         gold=["List.length_append"]),
    dict(id="L4", area="analysis", kind="pattern",
         query="|- tsum _ = _ * tsum _",
         gold=["tsum_mul_left"]),
    dict(id="L5", area="number_theory", kind="pattern",
         query="Nat.gcd ?a ?b ∣ ?a",
         gold=["Nat.gcd_dvd_left"]),
]


def name_matches(hit_name: Optional[str], accepted: List[str]) -> bool:
    """Matching rules 1-3 from the module docstring."""
    if not hit_name:
        return False
    for g in accepted:
        if hit_name == g or hit_name.endswith("." + g) or g.endswith("." + hit_name):
            return True
    return False


def first_match_rank(hits: List[Dict[str, Any]], accepted: List[str]) -> Optional[int]:
    """1-based rank of the first matching hit, or None."""
    for i, h in enumerate(hits, start=1):
        if name_matches(h.get("name"), accepted):
            return i
    return None


def probe_loogle(timeout_s: float = 20.0, attempts: int = 2) -> Dict[str, Any]:
    """Probe Loogle availability with a trivial known-good query."""
    last: Dict[str, Any] = {}
    for i in range(attempts):
        last = search_loogle("Real.sqrt", k=1, timeout_s=timeout_s)
        if last.get("available"):
            return last
        time.sleep(RATE_LIMIT_S)
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=10, help="hits per backend per query")
    ap.add_argument("--json", type=str, default=None,
                    help="also dump raw per-query results to this JSON file")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="per-call network timeout (s)")
    args = ap.parse_args()

    today = datetime.date.today().isoformat()
    print(f"# formal_search eval — {today}\n")

    probe = probe_loogle(timeout_s=args.timeout)
    loogle_up = bool(probe.get("available"))
    print(f"Loogle availability probe: {'UP' if loogle_up else 'DOWN'}"
          + ("" if loogle_up else f" — {probe.get('error')}"))
    print("LeanSearch: probed implicitly by the first query.\n")
    backend = "auto" if loogle_up else "leansearch"
    if not loogle_up:
        print("Loogle is down -> running LeanSearch-only; Loogle reported "
              "unavailable, no numbers fabricated for it.\n")

    rows: List[Dict[str, Any]] = []
    leansearch_calls_ok = 0
    leansearch_calls_fail = 0
    for entry in GOLD:
        time.sleep(RATE_LIMIT_S)
        res = search_formal_math(entry["query"], k=args.k, backend=backend,
                                 timeout_s=args.timeout)
        blocks = res["backends"]
        ls = blocks.get("leansearch", {"available": False, "hits": [],
                                       "error": "not queried"})
        lg = blocks.get("loogle",
                        {"available": False, "hits": [],
                         "error": probe.get("error", "down at probe time")})
        if ls.get("available"):
            leansearch_calls_ok += 1
        else:
            leansearch_calls_fail += 1
        row = dict(entry)
        row["rank"] = {
            "loogle": first_match_rank(lg["hits"], entry["gold"]),
            "leansearch": first_match_rank(ls["hits"], entry["gold"]),
            "combined": first_match_rank(res["hits"], entry["gold"]),
        }
        row["top1"] = {
            "loogle": (lg["hits"][0].get("name") if lg["hits"] else None),
            "leansearch": (ls["hits"][0].get("name") if ls["hits"] else None),
        }
        row["loogle_available"] = bool(lg.get("available"))
        row["leansearch_available"] = bool(ls.get("available"))
        rows.append(row)
        r = row["rank"]["combined"]
        print(f"  [{entry['id']}] rank(combined)={r if r else 'miss'}  "
              f"ls_top1={row['top1']['leansearch']}  q={entry['query'][:60]!r}")

    # ----------------------------- scoring -------------------------------
    def hit_at(rows_subset: List[Dict[str, Any]], chan: str, k: int) -> float:
        n = len(rows_subset)
        if n == 0:
            return float("nan")
        good = sum(1 for r in rows_subset
                   if r["rank"][chan] is not None and r["rank"][chan] <= k)
        return good / n

    subsets = {
        "all (25)": rows,
        "natural-language (20)": [r for r in rows if r["kind"] == "nl"],
        "pattern (5)": [r for r in rows if r["kind"] == "pattern"],
    }
    print("\n## Hit@k\n")
    print("| subset | channel | Hit@1 | Hit@5 | Hit@10 |")
    print("|---|---|---|---|---|")
    for sname, srows in subsets.items():
        for chan in ("loogle", "leansearch", "combined"):
            if chan == "loogle" and not loogle_up:
                print(f"| {sname} | loogle | n/a (down) | n/a (down) | n/a (down) |")
                continue
            cells = " | ".join(f"{hit_at(srows, chan, k):.2f}" for k in KS)
            print(f"| {sname} | {chan} | {cells} |")

    print("\n## Availability\n")
    loogle_ok = sum(1 for r in rows if r["loogle_available"])
    if loogle_up:
        print(f"- loogle: UP at probe; {loogle_ok}/{len(GOLD)} per-query calls "
              "succeeded. Failed calls count as MISSES in the table above "
              "(conservative; the service can flap mid-run).")
    else:
        print(f"- loogle: DOWN at probe ({probe.get('error')}); not queried "
              "per-query, all loogle cells reported n/a.")
    print(f"- leansearch: {leansearch_calls_ok}/{len(GOLD)} calls succeeded"
          + (f", {leansearch_calls_fail} failed" if leansearch_calls_fail else ""))

    print("\n## Per-query results\n")
    print("| id | kind | query | gold (canonical) | loogle rank | leansearch rank | combined rank |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        rk = r["rank"]
        def fmt(x: Optional[int], down: bool = False) -> str:
            if down:
                return "down"
            return str(x) if x is not None else "miss"
        print(f"| {r['id']} | {r['kind']} | {r['query'][:70]} | `{r['gold'][0]}` | "
              f"{fmt(rk['loogle'], down=not r['loogle_available'])} | "
              f"{fmt(rk['leansearch'], down=not r['leansearch_available'])} | "
              f"{fmt(rk['combined'])} |")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"date": today, "loogle_up": loogle_up,
                       "loogle_probe_error": probe.get("error"),
                       "k": args.k, "rows": rows}, f, indent=2)
        print(f"\nraw results -> {args.json}")


if __name__ == "__main__":
    main()
