# math_engine — build log: what was implemented, with the methods/papers used

> **Date:** 2026-06-05. Implements the architecture in `docs/01_landscape.md`
> (route → retrieve → MAP → VERIFY → PROVENANCE) on our own, with no dependency
> on TheoremSearch's running system/code (reference-only; their CC-BY/CC0
> *dataset* is used as raw data). All decisions below were taken after a fresh
> mid-2026 web scan of the latest methods; each is justified against alternatives.

## THE DESIGN CORRECTION (2026-06-05 re-architecture) — a tool FOR an AI

> **mathlas is a tool that an AI *uses*, NOT a tool that uses an AI.** The first
> cut had the problem-domain (`map.py`/`verify_apply.py`/`solve.py`) calling an
> LLM *internally* (an `AnthropicLLM` brain, an `[llm]` dep, an API key) — which
> is backwards: it made the tool need a key and cost money. Corrected so that an
> AI (Claude Code / Cursor / any agent) plugs mathlas in and **calls it**, while
> **mathlas itself never calls an LLM and needs no API key — free, pluggable
> everywhere.** The AI is the brain; mathlas provides the capabilities it lacks.

| Was (backwards) | Now (correct) |
|---|---|
| `map.py` extracts a signature **via LLM** | `mapping_scaffold(problem, candidate)` returns the needs↔guarantees questions as **DATA** the AI answers — **no LLM** |
| `verify_apply` informal tier **judges via LLM** | `applicability_checklist(candidate)` returns the result's atomic preconditions as a **checklist** the AI marks — **no LLM** |
| `solve(problem, retr, llm)` is the interface | the **MCP server** (`server.py`) + plain library functions are the interface; `solve()` is a **secondary** bring-your-own-LLM convenience (default `EchoLLM`, no vendor) |
| `AnthropicLLM`, `pip install '.[llm]'`, API key | **deleted.** Core (numeric + retrieval + verify + MCP) runs with **ZERO LLM, ZERO API key** |
| — | **new MCP server** exposes 6 AI-callable tools (`identify_constant`, `search_existing_math`, `verify_numeric`, `verify_formal`, `applicability_checklist`, `mapping_scaffold`) |

Register in Claude Code: `claude mcp add mathlas -- python -m mathlas.server`.
The server prefers the official `mcp` SDK (FastMCP) and **falls back to a
dependency-free stdio JSON-RPC MCP server** if `mcp` is absent — so it always runs.

## TL;DR of the latest-research scan (June 2026)

| Area | What the scan found is SOTA (2025-2026) | What we did |
|---|---|---|
| Math semantic retrieval | Open embedder **Qwen3-Embedding** tops MTEB (8B = 70.6); embed the NL *meaning*, not LaTeX; **hybrid dense+BM25 fused by RRF** is the robust pattern; graph/GNN premise selection beats text-only by ~25% but needs a trained GNN | Built **HybridRetriever** (dense + Okapi BM25 + RRF) over OUR OWN index; pluggable Qwen3 embedder (production) + zero-download fallback; graph-rerank documented as future work |
| Applicability verification | Generator/verifier **separation** + **decompose into atomic, individually-falsifiable conditions** (DeepSeekMath-V2); single LLM-as-judge & bare rubrics are **unreliable** (ProofGrader, Scaling Generative Verifiers) | Built tiered **VERIFY**: numeric (airtight, exists) + formal (Lean stub) + **`applicability_checklist`** — mathlas does the *decomposition* (atomic preconditions, **no LLM**) and hands it to the calling AI to falsify (the AI is the verifier) |
| Mapping / premise→problem | LLM analogy is emergent pattern-matching, lacks a consistency mechanism; **two-stage abduction→deduction** beats one-shot | **`mapping_scaffold`** returns the needs↔guarantees structure + questions as **DATA** (no LLM) for the AI to reason over; the optional BYO-LLM `map_candidates` keeps the two-stage abduction→deduction scheme |
| Numeric identify (existing) | mpmath PSLQ + independent high-precision re-eval | **Fixed two real bugs** (see below); recovery 75%→**100%**, false-pos **0%** |

## How the existing code measured up (kept / changed)

- **KEPT (sound, matches current best practice):** `engine.py`, `identify.py`,
  `provenance.py`, the `Retriever` interface and `ManualRetriever`, the whole
  retrieval stack (`embed.py`, `retrieve/*`). The numeric beachhead's *search-low
  / verify-high / independent-library* discipline is exactly right and survives
  the scan unchanged in spirit. (`llm.py` kept only as the BYO-LLM ABC + `EchoLLM`
  stub for the secondary `solve()` path — see the design correction above.)
- **RE-ARCHITECTED (2026-06-05, the design correction):** removed the internal
  LLM dependency. Deleted `AnthropicLLM` + the `[llm]` optional dep; added the
  **MCP server** (`server.py`) and the **no-LLM scaffolds** `mapping_scaffold`
  (map.py) and `applicability_checklist` (verify_apply.py). Core now runs with
  **zero LLM / zero API key**; the AI is the brain.
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

### MCP server — `server.py` (the primary, AI-callable interface) **[NEW]**
- Exposes six tools, **all NO-LLM**, returning JSON data the calling AI reasons
  over: `identify_constant`, `search_existing_math`, `verify_numeric`,
  `verify_formal`, `applicability_checklist`, `mapping_scaffold`. The tool bodies
  are plain functions (single source of truth); both server backends call them.
- **Two backends, one wire protocol:** prefers the official **`mcp` SDK
  (FastMCP)**; if `mcp` is not installed, falls back to a **dependency-free stdio
  JSON-RPC** server (`serve_stdio`/`_dispatch`) implementing `initialize` /
  `tools/list` / `tools/call`. So registration (`claude mcp add mathlas --
  python -m mathlas.server`) works with or without the SDK.
- `search_existing_math` defaults to a **small built-in seed corpus** (a dozen
  well-known theorems) so it works with **zero downloads / GPU / corpus**; pass
  `corpus_dir` for the real index. Retrievers are cached per corpus (data-flow
  discipline — no re-index per call).

### Mapping — `map.py` (NO-LLM scaffold + optional BYO-LLM two-stage)
- **`mapping_scaffold(problem, candidate)` [PRIMARY, no LLM]:** returns the
  needs↔guarantees structure as DATA — a lightly-parsed problem signature
  (objects/need/given), the candidate's checklist, the explicit
  needs↔guarantees *questions*, and a JSON *answer template* — for the calling
  AI to fill in. The analogy reasoning is the AI's job; mathlas supplies the
  structure (the "needs↔guarantees" step the unit-distance disproof showed is
  the valuable one).
- **Optional BYO-LLM path** (`extract_signature`→`map_candidates`): the two-stage
  abduction→deduction scheme (match a candidate's guarantee to a once-extracted
  requirement signature), used only by the secondary `solve()`. mathlas never
  supplies the LLM.

### Verification — `verify_apply.py` (tiered, cheapest-first)
- **NUMERIC** (`verify_numeric_claim`): airtight; reuses `verify.py`'s
  independent high-precision re-eval. Use when a claim reduces to a numeric
  identity. **No LLM.**
- **FORMAL** (`verify_formal`): Lean kernel-check **stub** with a fixed interface
  (slots a LeanDojo/Loogle checker in). Honors *typecheck ≠ correctness*.
- **INFORMAL — `applicability_checklist(candidate)` [PRIMARY, no LLM].** mathlas
  does NOT judge; it does the *decomposition* half of the generator/verifier
  split: heuristically parses the result's prose into an **atomic, problem-
  specific checklist** of preconditions (cue-word + bracket-aware clause splitting
  — 'Let'/'Suppose'/'If'/'where'/…, conclusion via 'then …') plus the conclusion
  it guarantees, and hands it to the calling AI to falsify. The AI is the
  adversarial verifier (single LLM-as-judge inside a tool is unreliable —
  ProofGrader; decomposition into atomic conditions is what works — DeepSeekMath-V2).
- An **optional BYO-LLM** `verify_informal(mapping, llm)` remains for the
  standalone `solve()` (adversarial skeptic, `passes>1` worst-verdict), but it is
  secondary and mathlas never supplies the LLM.
- Honors both hard-won lessons: retrieval is **gated by verification**, never
  blindly prepended; and the checklist forces a **need-vs-guarantee fit** check,
  not mere coherence.

### CLI — `cli.py` (`mathlas <number>` / `mathlas "<problem>"` / `mathlas mcp`)
Auto-routes (**no LLM, no API key**): a numeric arg → the airtight constant path;
a text arg → `search_existing_math` then prints the `mapping_scaffold` questions +
`applicability_checklist` for the top candidate (the data an AI reasons over),
over the seed corpus or `--corpus DIR`; `mathlas mcp` runs the MCP server. Wired
as `console_scripts` entry points (`mathlas`, `mathlas-mcp`).

## Validation run (light, CPU-only, **NO API key** — honoring the no-GPU constraint)

- **Zero-LLM import & key:** `import mathlas` is clean with `ANTHROPIC_API_KEY`
  unset and the `anthropic` SDK absent; `AnthropicLLM` is gone. The core (numeric
  + retrieval + verify + MCP server) runs with **no LLM and no API key**.
- **Numeric benchmark** (`benchmarks/numeric_bench.py`): **recovery 8/8 (100%)**,
  **false-positive 0/3 (0%)** — both DoD targets met; the honesty gate holds
  against structureless irrationals (`sin(1)·log(7)` etc.). Airtight tier intact.
- **MCP server** (`mathlas/server.py`): builds and **lists all 6 tools** under
  *both* the official FastMCP SDK *and* the dependency-free stdio fallback; driven
  end-to-end as a subprocess (a) via a real `mcp` stdio client and (b) via raw
  JSON-RPC lines on the fallback — `initialize` / `tools/list` / `tools/call` all
  return correct responses, notifications correctly get no reply.
- **`search_existing_math` + `verify_numeric` round-trip** on the built-in seed
  corpus: a query for the contraction/fixed-point result surfaces **Banach** top;
  `verify_numeric("…","pi**2/6")` returns verified (37 digits) and rejects
  `pi**2/7` — airtight check works through the tool layer.
- **No-LLM scaffolds:** `applicability_checklist` parses real statements into
  atomic preconditions + conclusion (bracket-comma protected: `Let (X,d) …` stays
  one clause; `If A and B, then C` → A, B, C correctly); `mapping_scaffold`
  returns the needs↔guarantees questions + answer template — **no LLM called**.
- **Retrieval Hit@k** (`scripts/eval_retrieval.py`, 15 test queries whose target
  paper is in the dataset + 3000 distractors, **hashing-embedder floor**):
  **paper-level Hit@20 = 15/15 (100%)**, **theorem-level Hit@20 = 11/15 (73%)**.
  This is the BM25-led floor; the Qwen3 dense channel + full index is the
  production number.
- **Structure checks:** all modules compile/import; the optional `solve()` path
  defaults to `EchoLLM` and degrades gracefully with no key.

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
