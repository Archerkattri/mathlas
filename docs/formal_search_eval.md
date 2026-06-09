# `search_formal_math` evaluation — finding canonical mathlib4 declarations

**Date of run: 2026-06-09** (live against the public Loogle and LeanSearch services).
Script: [`scripts/eval_formal_search.py`](../scripts/eval_formal_search.py). No LLM, no
API key, CPU + network only.

## Why this tool, and why this eval

`search_formal_math` closes the loop between mathlas's informal retrieval and its
formal verifier: instead of stopping at *"does this Lean snippet typecheck"*
(`verify_formal`), it answers *"here is the already-formalized mathlib theorem for the
statement you need"* — a declaration name + type with `external:loogle` /
`external:leansearch` provenance, ready to cite or `exact` in a proof. That composes
directly with `applicability_checklist`: the checklist tells the calling AI **which
hypotheses must hold** for a candidate result to apply, and `search_formal_math`
supplies the **kernel-checked formal statement** of that result, so the
needs-vs-guarantees match happens against real mathlib types rather than prose. The
question this eval answers is whether the thin, provenance-labeled proxy actually
surfaces the *canonical* declaration for well-known mathematics at useful rates — it
does: **Hit@5 = 0.96 combined over 25 queries** (24/25; Hit@1 = 0.64), with the two
backends visibly complementary (LeanSearch carries natural language, Loogle carries
type patterns).

## Headline results (Hit@k, 25 queries)

| subset | channel | Hit@1 | Hit@5 | Hit@10 |
|---|---|---|---|---|
| all (25) | loogle | 0.16 | 0.16 | 0.16 |
| all (25) | leansearch | 0.56 | 0.88 | 0.88 |
| all (25) | **combined (`backend="auto"`)** | **0.64** | **0.96** | **0.96** |
| natural-language (20) | loogle | 0.00 | 0.00 | 0.00 |
| natural-language (20) | leansearch | 0.60 | 0.95 | 0.95 |
| natural-language (20) | combined | 0.60 | 0.95 | 0.95 |
| pattern (5) | loogle | 0.80 | 0.80 | 0.80 |
| pattern (5) | leansearch | 0.40 | 0.60 | 0.60 |
| pattern (5) | combined | **0.80** | **1.00** | **1.00** |

Reading the table:

* **The backends are complementary, and the `auto` merge captures it.** LeanSearch
  alone gets Hit@1 = 0.56; adding Loogle lifts combined Hit@1 to 0.64 and pattern
  Hit@5 to 1.00. Loogle's 0.00 on natural-language queries is **by design**, not a
  failure — Loogle is a type/pattern engine and (correctly) returns parse
  errors/zero hits for English sentences. This is exactly why `search_formal_math`
  queries both.
* **Loogle's pattern score is availability-limited, not accuracy-limited.** Loogle
  flapped mid-run (below); on the 4 pattern queries where it answered it was
  **4/4 at rank 1**. The single pattern "miss" (L1) was a transient outage, counted
  as a miss anyway (conservative scoring).
* The only combined miss at k=10 is A7 ("every Cauchy sequence in a complete metric
  space converges"): LeanSearch returned near-neighbours
  (`cauchy_map_iff_exists_tendsto`, …) but not `cauchySeq_tendsto_of_complete` in
  the top 10.

## Backend availability on the run date

* **Loogle** (`loogle.lean-lang.org`): **unstable on 2026-06-09.** It returned
  502 Bad Gateway (nginx) earlier in the day, was UP at the eval's probe, then went
  down again for 5 consecutive per-query calls (N1–N4, L1) and recovered —
  **20/25 per-query calls succeeded**. Failed calls are scored as misses (no
  fabricated hits; the tool itself reports `available: False` + the real error).
  Loogle numbers above are therefore a *lower bound*.
* **LeanSearch** (`leansearch.net`): **25/25 calls succeeded.**

If Loogle is down at probe time, the script automatically runs LeanSearch-only and
says so (loogle cells reported `n/a (down)`).

## Gold set and matching rules

25 queries (20 natural-language, 5 Loogle-style type patterns) spanning analysis,
algebra, topology and number theory, each with an unambiguous, well-known canonical
mathlib4 declaration. **Every gold name and every accepted alternate was verified to
exist** by exact lookup in the official doc-gen4 declaration index
(`mathlib4_docs/declarations/declaration-data.bmp`, 410,507 declarations, snapshot of
2026-06-09) — *not* by trusting the search services under test.

A hit matches iff, for some accepted name `g`: `hit == g`, or `hit` ends with `"." + g`
(extra-namespace re-export/specialization, e.g. `Nat.mul_comm` for gold `mul_comm`),
or `g` ends with `"." + hit`. Nothing else — primed variants, `_iff` forms and sibling
lemmas do **not** count unless explicitly listed as alternates. Matching is
case-sensitive. Full accepted-alternate lists are in the `GOLD` table of the script.

**Gold-name corrections made during construction** (initial guesses that do *not*
exist in current mathlib, caught by the index check and replaced — no queries were
dropped): `inner_mul_le_norm_mul_norm` → `norm_inner_le_norm` (Cauchy–Schwarz was
renamed), `abs_add` → not used (now `abs_add_le`), `Gauss_sum`,
`real_inner_mul_le_norm_mul_norm`, `Monotone.tendsto_atTop_ciSup` — absent, never
used as gold.

## Per-query results (k = 10, ranks are 1-based; "down" = backend unavailable for that call)

| id | kind | query | gold (canonical) | loogle | leansearch | combined |
|---|---|---|---|---|---|---|
| A1 | nl | a bounded monotone sequence converges | `tendsto_atTop_ciSup` | miss | 1 | 1 |
| A2 | nl | intermediate value theorem for a continuous function on a closed interval | `intermediate_value_Icc` | miss | 1 | 1 |
| A3 | nl | mean value theorem: … derivative equals the slope of the secant line | `exists_deriv_eq_slope` | miss | 1 | 1 |
| A4 | nl | squeeze theorem: a sequence trapped between two sequences … converges | `tendsto_of_tendsto_of_tendsto_of_le_of_le` | miss | 3 | 3 |
| A5 | nl | the derivative of the sine function is cosine | `Real.deriv_sin` | miss | 1 | 1 |
| A6 | nl | Cauchy-Schwarz inequality: norm of the inner product ≤ product of norms | `norm_inner_le_norm` | miss | 1 | 1 |
| A7 | nl | every Cauchy sequence in a complete metric space converges | `cauchySeq_tendsto_of_complete` | miss | miss | **miss** |
| G1 | nl | Lagrange's theorem: order of a subgroup divides the order of the group | `Subgroup.card_subgroup_dvd_card` | miss | 1 | 1 |
| G2 | nl | the order of an element of a finite group divides the order of the group | `orderOf_dvd_card` | miss | 1 | 1 |
| G3 | nl | first isomorphism theorem: group mod kernel ≅ range | `QuotientGroup.quotientKerEquivRange` | miss | 3 | 3 |
| G4 | nl | the inverse of a product is the product of the inverses in reverse order | `mul_inv_rev` | miss | 1 | 1 |
| G5 | nl | binomial theorem: expansion of (x + y)^n | `add_pow` | miss | 1 | 1 |
| T1 | nl | the continuous image of a compact set is compact | `IsCompact.image` | miss | 2 | 2 |
| T2 | nl | a closed subset of a compact space is compact | `IsClosed.isCompact` | miss | 2 | 2 |
| T3 | nl | a compact subset of a Hausdorff space is closed | `IsCompact.isClosed` | miss | 2 | 2 |
| T4 | nl | Heine-Borel: compact ↔ closed and bounded (proper metric space) | `Metric.isCompact_iff_isClosed_bounded` | miss | 3 | 3 |
| N1 | nl | there are infinitely many prime numbers | `Nat.exists_infinite_primes` | down | 2 | 2 |
| N2 | nl | Fermat's little theorem: a^(p−1) ≡ 1 mod prime p | `ZMod.pow_card_sub_one_eq_one` | down | 1 | 1 |
| N3 | nl | the square root of two is irrational | `irrational_sqrt_two` | down | 1 | 1 |
| N4 | nl | Bezout's identity: gcd is a linear combination | `Nat.gcd_eq_gcd_ab` | down | 1 | 1 |
| L1 | pattern | `?a * ?b = ?b * ?a` | `mul_comm` | down | 2 | 2 |
| L2 | pattern | `Real.sqrt ?a * Real.sqrt ?a` | `Real.mul_self_sqrt` | 1 | 1 | 1 |
| L3 | pattern | `List.length (?a ++ ?b)` | `List.length_append` | 1 | miss | 1 |
| L4 | pattern | `\|- tsum _ = _ * tsum _` | `tsum_mul_left` | 1 | 1 | 1 |
| L5 | pattern | `Nat.gcd ?a ?b ∣ ?a` | `Nat.gcd_dvd_left` | 1 | miss | 1 |

NL-query "miss" in the loogle column means Loogle answered but (by design) found no
pattern match for an English sentence; "down" means that specific call failed during
the mid-run outage.

## Caveats

* Single run against **live public services on 2026-06-09**; LeanSearch's index and
  ranking, Loogle's index, and mathlib names all move — re-runs will differ
  (mathlib declarations get renamed; the gold set pins names verified on this date).
* Loogle availability was flapping on the run date; its scores are a lower bound
  (4/4 Hit@1 on the pattern calls it answered).
* k = 10 hits requested per backend per query; calls spaced ≥ 1 s apart (politeness).
* Gold "canonical name" judgments are ours; the accepted-alternate lists make the
  matching auditable, and the explicit rules forbid silent partial credit.

## Reproduce

```bash
cd third_party/math_engine
OMP_NUM_THREADS=2 python scripts/eval_formal_search.py --k 10 --json /tmp/formal_eval.json
```

Prints the markdown report (availability, Hit@k tables, per-query ranks) to stdout.
