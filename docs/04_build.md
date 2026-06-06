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
| — | **MCP server** exposes 13 AI-callable tools: the 7 core (`identify_constant`, **`identify_sequence`**, `search_existing_math`, `verify_numeric`, `verify_formal`, `applicability_checklist`, `mapping_scaffold`) + the 6-tool **discovery/web-augmentation layer** (`conjecture_relation`, `funsearch_evaluate`, `funsearch_register`, `funsearch_status`, `search_directive`, `add_finding`) |

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
- Exposes **thirteen** tools, **all NO-LLM**, returning JSON data the calling AI
  reasons over: the 7 core (`identify_constant`, `identify_sequence`,
  `search_existing_math`, `verify_numeric`, `verify_formal`,
  `applicability_checklist`, `mapping_scaffold`) + the 6 discovery/web-augmentation
  tools (`conjecture_relation`, `funsearch_evaluate`, `funsearch_register`,
  `funsearch_status`, `search_directive`, `add_finding`). The tool bodies are plain
  functions (single source of truth); both server backends call them.
- **Two backends, one wire protocol:** prefers the official **`mcp` SDK
  (FastMCP)**; if `mcp` is not installed, falls back to a **dependency-free stdio
  JSON-RPC** server (`serve_stdio`/`_dispatch`) implementing `initialize` /
  `tools/list` / `tools/call`. So registration (`claude mcp add mathlas --
  python -m mathlas.server`) works with or without the SDK.
- `search_existing_math` defaults to a **small built-in seed corpus** (a dozen
  well-known theorems) so it works with **zero downloads / GPU / corpus**; pass
  `corpus_dir` for the real index. Retrievers are cached per corpus (data-flow
  discipline — no re-index per call).

### Integer-sequence identification (OEIS) — `sequence.py` **[NEW, 2026-06-05]**
- **`identify_sequence(terms, max_results=5)` [airtight, NO LLM].** Given a list of
  integers, returns matching **OEIS** entries (A-number, name, `https://oeis.org/<A>`
  URL) by **EXACT term-match** against a *local* copy of the OEIS data — no fuzzy
  scoring, no embedding, no model. Either the terms occur (as a contiguous run) in
  a stored sequence or they do not; the verdict is mechanical (the numeric tier's
  *airtight-or-nothing* discipline, applied to sequences). Comparison is on Python
  `int` (arbitrary precision) so large OEIS terms never lose digits.
- **Data (downloaded once, gitignored, removable):** `stripped.gz` (terms) +
  `names.gz` (names) from `https://oeis.org/`, placed in
  `reference/downloads/oeis/` (~40 MB total). Override the location with
  `MATHLAS_OEIS_DIR` or the `data_dir` arg. If the data is absent,
  `identify_sequence` returns an **honest "data not available"** note — never a
  fake match.
- **Index (built once, cached — data-flow discipline):** on first call it parses
  ~396k sequences and builds an **n-gram index** (length-4 runs of consecutive
  terms → the A-numbers containing them), so lookups find candidates without
  scanning all of OEIS and the files are **never re-read** (the 50×-speedup
  lesson). Build ≈ 16 s once; subsequent lookups are sub-10 ms.
- **Matching (subsequence/offset-tolerant):** a query matches a stored sequence
  iff its terms appear as a **contiguous sub-run** anywhere in that sequence, so a
  leading-term/offset difference still matches — e.g. `[1,1,2,3,5,8,13,21]` hits
  Fibonacci A000045 (stored `0,1,1,2,3,5,8,…`, run found at offset 1). Each match
  reports the `offset` (0 ⇒ your terms are a leading prefix). Results rank by
  **A-number ascending** — OEIS gives low A-numbers to the canonical/foundational
  sequences, so the right entry surfaces rather than being buried under
  coincidental high-A-number prefix-sharers (this mirrors OEIS's own ordering).
- **CLI:** `mathlas 1,1,2,3,5,8,13,21` or `mathlas 2 3 5 7 11 13` auto-detects a
  sequence (≥ 2 integers, comma- or space-separated, `[]`/`()` ok) and prints the
  A-number/name/URL; `--oeis-dir DIR` overrides the data location; `--json` emits
  machine-readable output. A single number still routes to numeric mode.

### Formal (Lean) verify — `verify_apply.py::verify_formal` **[REAL CHECK, 2026-06-05]**
- The Lean tier is **no longer a stub.** When a Lean toolchain is installed and a
  `lean` snippet is supplied, `verify_formal` actually **runs the Lean
  type-checker** on the snippet (writes it to a temp file, invokes `lean`, reads
  the kernel's verdict) and reports whether it **TYPECHECKS** — a real
  kernel/proof-term check, not an opinion. `1 + 1 = 2 := rfl` ⇒ typechecks
  (`applies=True`, confidence 1.0); `1 + 1 = 3 := rfl` ⇒ the kernel's type error is
  captured (`applies=False`). **No LLM, no network at call time.**
- **The honest caveat is threaded through every verdict:** a typecheck proves the
  snippet is well-typed and its proof term passes the kernel — it does **NOT**
  prove the stated theorem is the right *applicability claim* for the problem
  (`typecheck ≠ proves-it-applies`; that mapping is the calling AI's job). Honors
  the long-standing "typecheck ≠ correctness" lesson.
- **Lean discovery (cached per process):** `find_lean()` resolves a runnable Lean
  via `LEAN` env → a real toolchain `lean` under the mathlas elan install
  (`reference/downloads/elan/toolchains/<tc>/bin/lean`, invoked directly to avoid
  the elan proxy's cwd warnings) → `elan which lean` → `lean` on `PATH`. A 120 s
  timeout guards against a hang.
- **Honest UNDETERMINED when Lean is absent (never a fake pass):** if no toolchain
  is found, or no snippet is given, the verdict is `applies=False` with the
  typecheck condition left **undetermined** (`satisfied=None`) and a clear note —
  the prior stub behaviour, but only when Lean genuinely cannot run.
- **Toolchain install (gitignored, removable):** elan + a recent Lean (the build
  used **Lean 4.30.0**) into `reference/downloads/elan` — see the one-liner in the
  README; deliberately **NOT** mathlib (too heavy, not needed for a bare-snippet
  kernel check). Installed in ≈ 1 min here (~2.7 GB on disk).
- **Server (`tool_verify_formal`):** now reports `lean_available`, `typechecks`,
  `applies`, `checked` (True iff Lean actually ran and gave a definite verdict),
  `stub` (the inverse), the kernel `detail`, and the caveat in `note`.

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

### DISCOVERY + WEB-AUGMENTATION layer — `ramanujan.py`, `funsearch.py`, `webaug.py` **[NEW, 2026-06-06]**

The "discovery + web-augmentation" layer — three more capabilities, all **NO LLM,
no network inside mathlas, no API key**, each returning DATA for the calling AI.
Built CPU-only while the 8B index build ran (did **not** touch the GPU,
`corpus.py`/`embed.py`, or `index_build/`). The INVARIANT holds: mathlas never
calls an LLM; numeric stays 8/8; the original 7 tools are unchanged.

**1) Ramanujan Machine — `ramanujan.py` / `conjecture_relation(value, max_terms?, cf_depth?)`.**
Goes *beyond* `identify_constant`'s flat basis with two channels, each gated by an
independent high-precision re-verification (reusing `verify.py`'s discipline):
- **(a) PSLQ over a RICHER basis** — we build the basis *vector* ourselves (the
  value, `x²`, the named atoms `pi,e,γ,Catalan,log2,log3,√2,√3,√5,ζ(3),ζ(5)`,
  their squares, pairwise products, and `1`) and run `mpmath.pslq` directly (vs
  `identify`'s flat name list), then reconstruct the integer relation as `x`'s
  closed form and re-verify it in **sympy** at higher precision. Recovers e.g.
  `ζ(2) → π²/6` (64 digits).
- **(b) Ramanujan-Machine continued fraction** — a bounded brute-force over TINY
  integer polynomials `aₙ=a₀+a₁n` (deg ≤ 1, positive leading), `bₙ=b₀+b₁n+b₂n²`
  (deg ≤ 2) whose **generalized CF** `a₀ + b₁/(a₁ + b₂/(a₂ + …))` (backward
  recurrence) matches a simple rational *image* `(px+q)/(rx+s)` of the constant.
  Each hit is locked by a **deeper** re-evaluation (depth×2, higher dps) before
  it is reported. Recovers the RM seed `π`: `aₙ=2n+1, bₙ=n² ⇒ 4/π` (60+ digits)
  and Euler's `aₙ=n+1, bₙ=n+1 ⇒ e−1`.
- **(b′) Simple CF + pattern** — the simple CF `[a₀; a₁, a₂, …]`, with pattern
  recognition (period-1/2 for quadratic irrationals; the e-type arithmetic run),
  verified by rebuilding the convergent. `e → [2;1,2,1,1,4,1,1,6,…]` (pattern
  named), `φ → [1;1,1,…]`, `√2 → [1;2,2,…]`, `√3 → [1;1,2,…]`.

Provenance is a **new** `Novelty.CONJECTURED_RELATION` (`"conjectured_relation"`)
— honestly a *numerically-verified conjecture, NOT a proof* (the AI can take it to
`verify_formal` / a human / the literature). Cites **Raayoni et al., *Nature* 590
(2021)** (the Ramanujan Machine) + **Ferguson-Bailey-Arno, *Math. Comp.* 68
(1999)** (PSLQ).

**2) FunSearch harness — `funsearch.py` / `funsearch_evaluate` + `funsearch_register` + `funsearch_status`.**
The **harness half only** of FunSearch — the CALLING AI is the program generator;
mathlas scores and stores, **no LLM**:
- **`evaluate`** runs a candidate program in a **sandboxed subprocess** — fresh
  `python -I -S` (isolated, no site), hard wall-clock `timeout`, the `socket`
  module stubbed out (no network), POSIX `RLIMIT_CPU`/`RLIMIT_AS` (CPU + memory),
  cwd in a throwaway temp dir. The candidate's own stdout is ignored; a single
  JSON result line is parsed. A signal-kill (e.g. `SIGXCPU` from the rlimit) is
  reported as an effective timeout. The program is AI-written and **not trusted**;
  the timeout is the primary guard, the rest defence-in-depth.
- **`register`** stores a scored program in an on-disk **MAP-Elites** DB
  (Mouret & Clune 2015) under `reference/downloads/funsearch/<problem>.json`
  (gitignored): one elite per behaviour cell + the global best, atomic writes.
- **`status`** returns the best program(s) + score + `few_shot_context` — the
  FunSearch best-shot prompt assembled as **DATA** the AI writes the next, better
  variant from (mathlas never sends it to a model).
- Ships two runnable scorers so the loop closes end-to-end: **`cap_set`** (size of
  a cap set in ℤ₃ⁿ — no three distinct points sum to 0 mod 3; FunSearch's headline
  result) and **`online_bin_packing`** (an online heuristic; score = −avg bins).
  Tested: cap_set starter 5 → a greedy 16 (known max 20, headroom for the AI);
  buggy / wrong-entry / infinite-loop / network programs all fail safely.

Cites **Romera-Paredes et al., *Nature* 625 (2024)** (FunSearch) + **OpenEvolve**
(open prior art) + **Mouret & Clune (2015)** (MAP-Elites).

**3) Web-augmented retrieval — `webaug.py` / `search_directive(problem)` + `add_finding(...)`.**
The local corpus is finite; the AI has the web. mathlas makes **no web call** — it
tells the AI *what* to search and ingests what comes back:
- **`search_directive`** reuses the NO-LLM needs↔guarantees signature
  (`map._heuristic_signature`) + transparent domain cues to return arXiv query
  strings, candidate sub-fields + their **arXiv categories** (`math.NT`/`math.CO`/…),
  named methods/inequalities/theorems to look for, and which OTHER mathlas tools to
  run (`identify_constant`/`conjecture_relation` when it spots a high-precision
  decimal, `identify_sequence` for an integer list, etc.).
- **`add_finding(statement, slogan, source, name?)`** appends a web-found result to
  the **live corpus** — a gitignored JSONL sidecar
  (`reference/downloads/findings.jsonl`) — through the **BM25/sparse channel +
  meta, with NO embedding-model load** (THE KEY CONSTRAINT: growing the index must
  not load the 8B per finding, so it works on any machine). It is retrievable
  **immediately**: `tool_search_existing_math` now RRF-fuses live findings (pure
  BM25 over the sidecar, cached by mtime) into its results, labelled provenance
  `web_added` (a new `Novelty.WEB_ADDED`). A dense vector is attached **only if** a
  Qwen3 index is *already* loaded in-process (we sniff the server's retriever cache
  and reuse its encoder — never construct a model); otherwise dense is skipped.
- **Batch dense reindex (documented, deferred):** `scripts/reindex_findings.py`
  loads the embedder **once** on a GPU box and fills in `dense_vec` for the backlog
  of findings that lack one (idempotent/resumable; `--dry-run` loads no model).
  This is the only place a model is loaded for findings.

### CLI — `cli.py` (`mathlas <number>` / `mathlas <sequence>` / `mathlas "<problem>"` / `mathlas mcp`)
Auto-routes (**no LLM, no API key**): a single number → the airtight constant path;
**≥ 2 integers** (`1,1,2,3,5,8` or `1 1 2 3 5 8`, `[]`/`()` ok) → the airtight OEIS
`identify_sequence` path (`--oeis-dir DIR` to relocate the data); a text arg →
`search_existing_math` then prints the `mapping_scaffold` questions +
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
- **MCP server** (`mathlas/server.py`): builds and **lists all 13 tools** under
  *both* the official FastMCP SDK *and* the dependency-free stdio fallback; driven
  end-to-end as a subprocess via raw JSON-RPC lines on the fallback —
  `initialize` / `tools/list` / `tools/call` all return correct responses,
  notifications correctly get no reply.
- **`search_existing_math` + `verify_numeric` round-trip** on the built-in seed
  corpus: a query for the contraction/fixed-point result surfaces **Banach** top;
  `verify_numeric("…","pi**2/6")` returns verified (37 digits) and rejects
  `pi**2/7` — airtight check works through the tool layer.
- **Discovery + web-augmentation layer (CPU-only, no key, no GPU):**
  - `conjecture_relation` over the MCP wire on `e` returns the **verified** e-type
    simple CF `[2;1,2,1,1,4,1,1,6,…]` (pattern recognised) + the generalized CFs
    `e−1` and (on `π`) `4/π` to 60+ digits, and PSLQs `ζ(2) → π²/6` (64 digits) over
    the richer basis; provenance `conjectured_relation` (honest: verified, not proved).
  - `funsearch_evaluate` sandbox-scores the cap_set starter (5) and a greedy
    variant (16; max 20), with **timeout / network-block / buggy / wrong-entry**
    all failing safely; `funsearch_register` → MAP-Elites DB + global-best report;
    `funsearch_status` → best program + few-shot context. Full loop runs end-to-end
    over the wire on both shipped problems.
  - `search_directive("evaluate sum 1/n^4")` returns sub-fields (analysis / number
    theory / special functions), arXiv categories, named results, and suggests
    `conjecture_relation`; `add_finding(...)` appends a result **with no model load**
    (`dense_added=False`), and it is then retrieved via `search_existing_math`
    (RRF-fused, provenance `web_added`). A dense vector is added only when a
    retriever is already resident (verified by reusing a loaded hashing embedder).
- **`identify_sequence` (OEIS, airtight):** with the local OEIS data present,
  `[1,1,2,3,5,8,13,21] → A000045` (Fibonacci, found at offset 1) and
  `[2,3,5,7,11,13] → A000040` (primes, top match) — over **396 329** local
  sequences; index built once (≈ 16 s) then cached (sub-10 ms lookups). Catalan/
  factorials/powers-of-2/squares also resolve to their canonical A-numbers; a
  structureless run → honest UNIDENTIFIED. Verified through the MCP `tools/call`
  layer and the CLI (comma- and space-separated).
- **`verify_formal` (Lean, REAL kernel check):** with Lean 4.30.0 installed,
  `theorem t : 1 + 1 = 2 := rfl` **typechecks** (`checked=True, applies=True`) and
  `1 + 1 = 3 := rfl` is reported as a kernel type error (`applies=False`); a
  hypothesis-bearing theorem (`Nat.add_le_add_right`) also typechecks. With Lean
  forced unavailable → honest UNDETERMINED (`applies=False`, typecheck condition
  `None`), **never a fake pass**. Verified through the MCP `tools/call` layer.
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

- **OEIS** — The On-Line Encyclopedia of Integer Sequences, OEIS Foundation Inc.,
  `https://oeis.org`. *(data source for `identify_sequence`: the published
  `stripped.gz`/`names.gz` bulk files, used under the OEIS end-user license;
  matched exactly, locally — not their API.)*
- **Lean 4 / elan** — de Moura & Ullrich, "The Lean 4 Theorem Prover and
  Programming Language," CADE 2021; toolchain installed via `elan`
  (`github.com/leanprover/elan`). *(the formal tier's real kernel typecheck.)*
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
- **Ramanujan Machine** — Raayoni, Gottlieb, Manor, Pisha, Harris, Mendlovic,
  Haviv, Hadad & Kaminer, "Generating conjectures on fundamental constants with the
  Ramanujan Machine," *Nature* 590, 67-73 (2021). *(the continued-fraction /
  polynomial-recurrence conjecture form in `ramanujan.py`.)*
- **PSLQ** — Ferguson, Bailey & Arno, "Analysis of PSLQ, an integer relation
  finding algorithm," *Math. Comp.* 68, 351-369 (1999). *(the integer-relation
  engine over the richer basis; via `mpmath.pslq`.)*
- **FunSearch** — Romera-Paredes, Barekatain, Novikov, Balog, Kumar, Dupont, Ruiz,
  Ellenberg, Wang, Fawzi, Kohli & Fawzi, "Mathematical discoveries from program
  search with large language models," *Nature* 625, 468-475 (2024). *(the
  generator/harness split — mathlas is the harness half, no LLM — in `funsearch.py`.)*
- **OpenEvolve** — open-source FunSearch/AlphaEvolve re-implementation. *(open prior
  art for the program-search loop.)*
- **MAP-Elites** — Mouret & Clune, "Illuminating search spaces by mapping elites,"
  arXiv:1504.04909 (2015). *(the islands-of-elites program DB in `funsearch.py`.)*
- **Reference-only** — TheoremSearch, arXiv:2602.05216 (UW Math AI Lab, 2026):
  studied for lessons (analysis in `docs/03`); their *dataset* (CC-BY/CC0) used as
  raw data. Their API/MCP/index/code are **not** a runtime dependency.

## Delivered after the first cut (2026-06-05)

- **OEIS sequence tier — DONE** (`mathlas/sequence.py`, MCP `identify_sequence`,
  CLI sequence mode). Airtight exact term-match against a local OEIS copy; see the
  module section above. Tested: `[1,1,2,3,5,8,13,21] → A000045`,
  `[2,3,5,7,11,13] → A000040`.
- **Formal Lean tier — DONE (real check)** (`verify_formal` runs the Lean kernel;
  Lean 4.30.0 installed under `reference/downloads/elan`). Stub replaced; honest
  UNDETERMINED only when Lean is genuinely unavailable. See the module section.

## Deliberately deferred (documented, not run)

- **Full 9.2M Qwen3 index** — `scripts/build_index.py` (offline GPU; not run under
  the current GPU-sharing constraint).
- **Lean + mathlib** — only a *bare-snippet* kernel check is wired (no mathlib);
  checking snippets that `import Mathlib` would need the (heavy) mathlib build.
  A LeanDojo/Loogle premise-retrieval layer on top remains future work.
- **Graph-rerank** — use the dataset's formal+informal dependency edges as a
  structure signal (the 2510.23637 result); higher value, needs a trained GNN.
- **PyPI name** — `mathlas` is provisional; availability-check before publish.
