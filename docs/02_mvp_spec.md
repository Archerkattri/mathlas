# math_engine — MVP spec: the numeric beachhead (v0)

> **Date:** 2026-06-05. **Beachhead chosen:** numeric / sequence / constant (airtight automatic verification). Locks the build for v0.

## Goal
Prove the engine's loop — **ROUTER → IDENTIFY → VERIFY → PROVENANCE** — end-to-end on the one domain where verification is airtight and automatic. Input a real number / constant / integer sequence; output the **existing** closed form / catalogued sequence that matches, **independently verified**, with an honest provenance label (`known` / `unidentified` — **never "novel"**).

## Why this domain first
- **Verification is airtight:** re-evaluate the candidate to *higher* precision than the search used, with a *different* library; require N-digit agreement. Rejects spurious low-precision integer-relation hits — the false-positive gate the Oct-2025 GPT-5 episode lacked.
- **Loved precedent + visceral demo:** OEIS/Superseeker, ISC/RIES — "paste a number → get the formula." We unify + route + verify them.
- **Full loop works today** with permissive pure-python tools (mpmath PSLQ, sympy, OEIS API).
- **It's the general architecture instantiated** — widening to formal/informal reuses the same `router → verify → provenance` spine.

## Architecture (v0 modules)
- `engine.py` — facade: `identify(value) → Result(candidates, best, provenance)`. (Routing is implicit in v0: real constant. Sequence/expr routing = next slice.)
- `identify.py` — constant → `mpmath.identify` / PSLQ against a **known-constant basis** → candidate closed forms (the "find existing math" step).
- `verify.py` — **independent** re-evaluation (sympy) at `dps_verify > dps_search`; `digits_agreed ≥ min_digits` gate.
- `provenance.py` — label `KNOWN_FORM` | `SEQUENCE_MATCH` | `UNIDENTIFIED`. **Never "novel."** Records source + method + basis.
- `benchmarks/numeric_bench.py` — recovery@known + false-positive@random.
- *(next slice)* `identify/sequence.py` — OEIS lookup; verify by term-match.

## The verify+provenance discipline (the differentiator vs TheoremSearch)
1. **Search** at `dps_search` (e.g. 30). 2. **Re-evaluate** each candidate *independently* (sympy) at `dps_verify` (e.g. 50). 3. **Keep** only candidates agreeing to ≥ `min_digits` (e.g. 20). 4. **Provenance** = `known/retrieved` with the exact source; `unidentified` if none verify. False positives become structurally hard, and every hit is auditable.

**Known subtlety (surface, don't hide):** a bare `identify` trivially fits any truncated decimal as a *rational*. "Meaningful" identification = a relation involving the **basis constants**, not a huge-denominator rational. v0 keeps both but the benchmark's negatives are full-precision *irrationals* (e.g. `sin(1)·log(7)`) so the gate is tested honestly.

## Dependencies (all permissive)
`mpmath` (BSD), `sympy` (BSD); `requests` (Apache, OEIS — optional extra). Pure-python, pip-installable. **Do not link RIES (GPL-3.0)** — call it as a service if added later.

## Definition of Done (beachhead proven)
- Benchmark: **recovery ≥ 90%** on curated known constants (ζ(2), ζ(3), Catalan, golden ratio, log 2, …); **false-positive ≤ 5%** on structureless irrationals.
- Every output carries an **independently-verified** provenance label.
- Offline for constants; online (OEIS) for sequences. `pip install` clean.
- Demo: `identify(zeta(2)) → "pi**2/6"  [known_form, verified 48 digits]`.

## Then widen
formal/Lean (free verify, ReProver/LeanSearch) → informal "describe → find + verify-applicability" (the moat + biggest market). Same `router → verify → provenance` spine.

## Open design decisions (steer welcome)
1. **Package/PyPI name** — `mathlas` is provisional/generic; availability-check before publishing (like splatreg → gsfit-was-taken).
2. **PSLQ basis breadth** — conservative known-constants (fewer false positives) vs aggressive (higher recall). The verify gate mitigates breadth risk.
3. **OEIS dependency in v0** — live network call vs bundling a local sequence subset for offline reproducibility.

## Constraint
**Execution DEFERRED** until DiT-FID (#562) frees the GPUs. Code is written now; the benchmark RUNS once the box is clear (light CPU, but honoring the post-crash no-CPU-during-GPU rule).

## Status update (2026-06-05) — built + validated
The full architecture is now implemented beyond this numeric v0 (retrieval, two-stage MAP, tiered VERIFY incl. the informal moat, CLI) — see **`docs/04_build.md`** for the latest-research scan, the kept/changed verdict on existing code, and citations. Light CPU-only validation passed: numeric **recovery 8/8, false-positive 0/3** (after fixing two real bugs: missing `apery` in the basis + the mpmath↔sympy constant-name mismatch that silently failed Catalan/ζ(3) verification); retrieval **paper-level Hit@20 = 15/15** on the test subset (hashing-embedder floor); the adversarial informal verifier correctly rejects a mis-retrieved candidate. The full 9.2M Qwen3 index remains a documented offline-GPU script (`scripts/build_index.py`), not run under the GPU-sharing constraint.
