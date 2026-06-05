# math_engine — build log: what was implemented, with the methods/papers used

> **Date:** 2026-06-05. Implements the architecture in `docs/01_landscape.md`
> (route → retrieve → MAP → VERIFY → PROVENANCE) on our own, with no dependency
> on TheoremSearch's running system/code (reference-only; their CC-BY/CC0
> *dataset* is used as raw data). All decisions below were taken after a fresh
> mid-2026 web scan of the latest methods; each is justified against alternatives.

## TL;DR of the latest-research scan (June 2026)

| Area | What the scan found is SOTA (2025-2026) | What we did |
|---|---|---|
| Math semantic retrieval | Open embedder **Qwen3-Embedding** tops MTEB (8B = 70.6); embed the NL *meaning*, not LaTeX; **hybrid dense+BM25 fused by RRF** is the robust pattern; graph/GNN premise selection beats text-only by ~25% but needs a trained GNN | Built **HybridRetriever** (dense + Okapi BM25 + RRF) over OUR OWN index; pluggable Qwen3 embedder (production) + zero-download fallback; graph-rerank documented as future work |
| Applicability verification | Generator/verifier **separation** + **decompose into atomic, individually-falsifiable conditions** (DeepSeekMath-V2); single LLM-as-judge & bare rubrics are **unreliable** (ProofGrader, Scaling Generative Verifiers) | Built tiered **VERIFY**: numeric (airtight, exists) + formal (Lean stub) + **informal = structured-adversarial precondition check** (the moat) |
| Mapping / premise→problem | LLM analogy is emergent pattern-matching, lacks a consistency mechanism; **two-stage abduction→deduction** beats one-shot | Rewrote **map.py** to two-stage: extract a requirement *signature* once (abduction), then match each candidate's guarantee to it (deduction) |
| Numeric identify (existing) | mpmath PSLQ + independent high-precision re-eval | **Fixed two real bugs** (see below); recovery 75%→**100%**, false-pos **0%** |

## How the existing code measured up (kept / changed)

- **KEPT (sound, matches current best practice):** `engine.py`, `identify.py`,
  `provenance.py`, `llm.py` (provider-agnostic ABC — correct), the `Retriever`
  interface and `ManualRetriever`. The numeric beachhead's *search-low /
  verify-high / independent-library* discipline is exactly right and survives the
  scan unchanged in spirit.
- **FIXED (real bugs in the numeric tier, found on first run):**
  1. `DEFAULT_BASIS` lacked `apery`, so **ζ(3) was unrecoverable**. Added it.
  2. `verify.py` re-evaluated candidates with `sympy.sympify`, but mpmath's
     constant names (`catalan`, `euler`, `apery`) sympify to **non-numeric
     Symbols** — so Catalan/ζ(3)/γ silently **failed verification**. Added a
     name-translation map (`catalan→Catalan`, `euler→EulerGamma`, `apery→zeta(3)`)
     and a numeric-result guard, preserving the independent-library airtightness.
     Result: numeric `recovery 6/8 → 8/8`, `false-positive 0/3` (unchanged).
  3. Added display-only `simplify` so `pi**2*(sqrt(24)/12)**2 → pi**2/6` (never
     touches the verified `expr`).
- **IMPROVED:** `map.py` → two-stage abduction→deduction (below); threads the
  problem text onto `Mapping` for the verifier.
- **EXTENDED:** `solve.py` → now runs the full route→retrieve→MAP→**VERIFY**→
  provenance loop and labels each result; `provenance.py` → added retrieval-domain
  labels.
- **BUILT (were missing):** the semantic **HybridRetriever** (+ `bm25.py`,
  `corpus.py`, `embed.py`), the **informal/formal VERIFY tiers**
  (`verify_apply.py`), and the **CLI** (`cli.py`).

## What was built — module by module

### Retrieval (our own index) — `embed.py`, `retrieve/{bm25,corpus,hybrid}.py`
- **`HybridRetriever`** = DENSE (an `Embedder` over each theorem's NL slogan) +
  SPARSE (Okapi BM25 over name+slogan+statement), fused by **Reciprocal Rank
  Fusion**. RRF is unsupervised, needs no score normalisation, and consistently
  beats either channel — the right call since the *default* embedder is weak and
  BM25 must carry exact symbol/term hits.
- **`Embedder`** is pluggable like the LLM. Production = **`Qwen3Embedder`**
  (open MTEB SOTA, instruction-prefixed queries, Matryoshka-truncatable dims),
  lazy-loaded. Default = **`HashingEmbedder`** (zero-download) so the whole
  pipeline + validation run CPU-only/offline.
- **`corpus.py`** reads the open dataset parquets as RAW DATA and builds our own
  `Document` records; the *embedded unit* is the NL **slogan**, not LaTeX (the
  load-bearing lesson). `bm25.py` is a pure-NumPy/SciPy CSR Okapi BM25.
- Full-corpus build is a **documented offline-GPU script** (`scripts/build_index.py`),
  deliberately NOT run here (resource limits); validation uses a small subset.

### Mapping — `map.py` (two-stage)
- **Abduction** (`extract_signature`, once/problem): objects + need + given +
  field-hints — the structure a solving result must match, candidate-independent.
- **Deduction** (`map_candidate`, per candidate): match the candidate's
  *guarantee* to that fixed signature (direct or via reduction), rather than
  re-reading the problem fresh each time (which lets the model drift to keyword
  similarity). This operationalises the "needs↔guarantees" insight as the
  *structural* mapping the analogy literature says beats one-shot.

### Verification — `verify_apply.py` (tiered, cheapest-first)
- **NUMERIC** (`verify_numeric_claim`): airtight; reuses `verify.py`'s
  independent high-precision re-eval. Use when a claim reduces to a numeric
  identity.
- **FORMAL** (`verify_formal`): Lean kernel-check **stub** with a fixed interface
  (slots a LeanDojo/Loogle checker in). Honors *typecheck ≠ correctness*.
- **INFORMAL** (`verify_informal`) — **the moat.** NOT "score this 0-7" (shown
  unreliable). Instead: (1) extract the candidate's hypotheses as an **atomic,
  problem-specific checklist** (a marking scheme from the result itself, not a
  generic rubric); (2) an **adversarial skeptic** must confirm every precondition
  against the problem or name the single one that fails; (3) `passes>1` runs the
  skeptic independently and takes the **worst** verdict (cheap
  consistency-for-rejection). Mirrors DeepSeekMath-V2's generator/verifier split.
- Honors both hard-won lessons: retrieval is **gated by verification**, never
  blindly prepended (it hurt strong models); and we check **need-vs-guarantee
  fit**, not mere coherence.

### CLI — `cli.py` (`mathlas "<problem>"` / `mathlas <number>`)
Auto-routes: a numeric arg → the airtight constant path (no LLM/network); a text
arg → retrieve→map→verify over `--corpus` (LLM optional; without one, prints
retrieval-only candidates). Wired as a `console_scripts` entry point.

## Validation run (light, CPU-only — honoring the no-GPU constraint)

- **Numeric benchmark** (`benchmarks/numeric_bench.py`): **recovery 8/8 (100%)**,
  **false-positive 0/3 (0%)** — both DoD targets met; the honesty gate holds
  against structureless irrationals (`sin(1)·log(7)` etc.).
- **Retrieval Hit@k** (`scripts/eval_retrieval.py`, 15 test queries whose target
  paper is in the dataset + 3000 distractors, **hashing-embedder floor**):
  **paper-level Hit@20 = 15/15 (100%)**, **theorem-level Hit@20 = 11/15 (73%)**.
  This is the BM25-led floor; the Qwen3 dense channel + full index is the
  production number.
- **Informal-verify pipeline** (deterministic mock LLM): the adversarial verifier
  **correctly rejects** a mis-retrieved candidate the mapping step let through
  (labels it `retrieved_rejected` with a concrete failure), keeping only the
  genuinely-applicable result (`retrieved_applies`) — demonstrating the wedge
  retrieval-only systems lack.
- **Structure checks:** all modules compile/import; abduction runs **once** per
  problem, deduction **per candidate** (confirmed); `EchoLLM` degrades gracefully.

## Citations (methods used)

- **Qwen3-Embedding** — Zhang et al., "Qwen3 Embedding: Advancing Text Embedding
  and Reranking Through Foundation Models," arXiv:2506.05176 (2026). *(open MTEB
  SOTA; production embedder + the "embed meaning not notation" lesson.)*
- **Reciprocal Rank Fusion** — Cormack, Clarke & Buettcher, "Reciprocal Rank
  Fusion outperforms Condorcet and individual rank learning methods," SIGIR 2009.
  *(hybrid fusion; reconfirmed best simple fusion across 2025-2026 RAG studies.)*
- **DeepSeekMath-V2** — "Towards Self-Verifiable Mathematical Reasoning,"
  arXiv:2511.22570 (2025). *(generator/verifier separation + iterate-until-no-issue;
  the informal-tier design.)*
- **ProofGrader** — Petrov et al., "Reliable Fine-Grained Evaluation of Natural
  Language Math Proofs," arXiv:2510.13888 (ICLR 2026). *(LLM-as-judge is
  unreliable; bare rubrics don't reliably help → decompose into atomic conditions.)*
- **Scaling Generative Verifiers** — arXiv:2511.13027 (2025). *(NL proof
  verification/selection at scale; marking-scheme generation.)*
- **Graph + text premise selection** — Petrovčič, Narvaez Denis & Todorovski,
  "Combining Textual and Structural Information for Premise Selection in Lean,"
  arXiv:2510.23637 (2025). *(GNN over the dependency graph beats text-only ReProver
  by >25% → documented structure-rerank as future work; our dataset has the edges.)*
- **LeanSearch v2** — "Global Premise Retrieval for Lean 4," arXiv:2605.13137
  (2026). *(staged sketch-retrieve-reflect; informs the formal-tier roadmap.)*
- **LLM analogical/structural reasoning** — Webb, Holyoak & Lu (LLM analogy) and
  structural-mapping work (e.g. arXiv:2603.29997): two-stage abduction→deduction
  beats one-shot. *(map.py design.)*
- **REAL-Prover** — arXiv:2505.20613 (2025); **LeanDojo** — arXiv:2306.15626
  (2023). *(retrieval-augmented Lean proving; formal-tier references.)*
- **Reference-only** — TheoremSearch, arXiv:2602.05216 (UW Math AI Lab, 2026):
  studied for lessons (analysis in `docs/03`); their *dataset* (CC-BY/CC0) used as
  raw data. Their API/MCP/index/code are **not** a runtime dependency.

## Deliberately deferred (documented, not run)

- **Full 9.2M Qwen3 index** — `scripts/build_index.py` (offline GPU; not run under
  the current GPU-sharing constraint).
- **Formal Lean tier** — interface fixed; wire a LeanDojo/Loogle checker.
- **Graph-rerank** — use the dataset's formal+informal dependency edges as a
  structure signal (the 2510.23637 result); higher value, needs a trained GNN.
- **OEIS sequence tier** — the next numeric slice (`identify/sequence.py`).
- **PyPI name** — `mathlas` is provisional; availability-check before publish.
