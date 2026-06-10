# mathlas — design & methods

## Design principle: a tool FOR an AI (no LLM, no API key)

mathlas is a tool that an AI *uses*, **not** a tool that uses an AI. It **never calls
an LLM and needs no API key** — so it is free and pluggable into Claude Code, Cursor,
or any MCP client. The AI is the brain; mathlas provides the capabilities an AI lacks
and returns **data** (candidates, verdicts, checklists, scaffolds) for the AI to
reason over. Concretely, the judgment steps are returned as structure, not opinion:

| Capability | How mathlas provides it (no LLM) |
|---|---|
| "what existing result fits?" | `search_existing_math` over its **own** dense+BM25+RRF index |
| "does it apply to my problem?" | `applicability_checklist` / `mapping_scaffold` return the result's atomic preconditions + needs↔guarantees questions for the AI to mark |
| "is this numeric claim true?" | `verify_numeric` / `identify_constant` — independent high-precision re-evaluation |
| "does this Lean statement check?" / "is my Lean proof correct?" | `verify_formal` — the real Lean kernel, incl. full proof checking (or an honest UNDETERMINED) |
| "what is this sequence/constant?" | `identify_sequence` (exact OEIS), `conjecture_relation` (PSLQ + Ramanujan-Machine CF) |

## Architecture

```
search_existing_math ─▶ mapping_scaffold + applicability_checklist ─▶ (AI judges) ─▶ verify_numeric / verify_formal
   (own index)            (needs↔guarantees, no LLM)                                  (airtight)
```

The MCP server exposes **twelve** AI-callable tools, all NO-LLM, returning JSON:
the **8 core** (`identify_constant`, `identify_sequence`, `search_existing_math`,
`search_formal_math`, `verify_numeric`, `verify_formal`, `applicability_checklist`,
`mapping_scaffold`) + the **4-tool discovery / web-augmentation layer**
(`conjecture_relation`, `funsearch` — one tool, `action=evaluate/register/status` —
`search_directive`, `add_finding`). Tool bodies are plain functions (single source
of truth); both server backends call them.

## Modules

- **Retrieval — `embed.py`, `retrieve/{bm25,corpus,hybrid}.py`.** `HybridRetriever`
  = a dense `Embedder` over each theorem's natural-language **slogan** (embed the
  *meaning*, not the LaTeX) + Okapi **BM25** over name+slogan+statement+label, fused
  by **Reciprocal Rank Fusion** (unsupervised, no score normalisation, robust when
  one channel is weak). Production embedder = **Qwen3-Embedding** (open MTEB SOTA),
  lazy-loaded; a zero-download `HashingEmbedder` lets the pipeline run CPU-only.
  `corpus.py` reads the open dataset parquets as raw data into our own `Document`
  records.

- **MCP server — `server.py`.** Two backends, one wire protocol: prefers the official
  `mcp` SDK (FastMCP); if absent, falls back to a **dependency-free stdio JSON-RPC**
  server implementing `initialize` / `tools/list` / `tools/call`. So
  `claude mcp add mathlas -- python -m mathlas.server` works with or without the SDK.
  `search_existing_math` defaults to a small built-in seed corpus (works with zero
  downloads); pass `corpus_dir` for the real index. Retrievers are cached per corpus.

- **Numeric — `identify.py`, `verify.py`.** `mpmath.identify` (PSLQ) against a basis
  of well-known constants proposes a closed form; an **independent** sympy re-eval at
  higher precision verifies it (search-low / verify-high / different-library). The
  proposing step is permissive; the verify pass is what filters spurious
  low-precision relations — airtight or nothing.

- **Sequence (OEIS) — `sequence.py`.** `identify_sequence(terms)` returns matching
  OEIS entries by **exact contiguous term-match** against a local copy of the OEIS
  bulk data (no fuzzy scoring, no model). An n-gram index (length-4 runs → A-numbers)
  is built once and cached; matches are subsequence/offset-tolerant and ranked by
  A-number (OEIS's canonical ordering). Comparison is on arbitrary-precision `int`.

- **Formal (Lean) — `verify_apply.py::verify_formal`.** When a Lean toolchain is
  installed and a snippet is given, it **runs the real Lean kernel** on the snippet
  and reports whether it typechecks; otherwise an honest UNDETERMINED, never a fake
  pass. Caveat threaded through every verdict: a typecheck proves the snippet is
  well-typed, **not** that the stated theorem is the right applicability claim
  (`typecheck ≠ proves-it-applies`; that mapping is the AI's job).
  **Proof checking (`check_lean_proof`)**: given a Lean 4 proposition + an
  AI-written proof, mathlas elaborates `theorem _mathlas_check : <statement> :=
  <proof>` (leading `import` lines hoisted to the file top, the proof indented
  uniformly so tactic blocks keep their layout) and kernel-checks the full
  declaration → `VERIFIED_PROOF` / `REFUTED` (kernel error verbatim — the agent's
  repair-loop payload) / `UNDETERMINED` (no toolchain, 60 s timeout, or an import
  the bare toolchain can't resolve). `sorry`/`admit` are **rejected**, both by a
  comment-stripped source scan and by the kernel's `sorryAx` diagnostics — Lean
  exits 0 on a sorried proof, so the exit code alone would fake-pass. mathlas
  never *generates* proofs; the generator/verifier split is absolute.

- **Mapping + verification scaffolds — `map.py`, `verify_apply.py`.**
  `mapping_scaffold` returns the needs↔guarantees structure (problem signature, the
  candidate's checklist, the explicit questions, a JSON answer template) as data.
  `applicability_checklist` does the *decomposition* half of the generator/verifier
  split — heuristically parsing a result's prose into atomic, problem-specific
  preconditions (cue-word + bracket-aware clause splitting) — and hands it to the AI
  to falsify. A single LLM-as-judge inside a tool is unreliable (ProofGrader);
  decomposition into atomic conditions is what works (DeepSeekMath-V2). An optional
  bring-your-own-LLM path (`solve()`) exists but is secondary; mathlas never supplies
  the LLM.

- **Discovery — `ramanujan.py`.** `conjecture_relation` goes beyond `identify`'s flat
  basis: (a) PSLQ over a **richer** basis (the value, its square, named atoms, their
  squares + pairwise products) reconstructed as a closed form and re-verified in
  sympy; (b) a **Ramanujan-Machine** generalized-continued-fraction search over tiny
  integer polynomials, each hit locked by a deeper re-evaluation; (c) the simple CF +
  pattern recognition. Provenance is `conjectured_relation` — a numerically-verified
  conjecture, **not** a proof.

- **FunSearch harness — `funsearch.py`.** The harness half only — the AI is the
  program generator. `funsearch_evaluate` runs a candidate in a **sandboxed
  subprocess** (`python -I -S`, hard wall-clock timeout, `socket` stubbed out, POSIX
  `RLIMIT_CPU`/`RLIMIT_AS`, throwaway cwd; the program is untrusted, the timeout is
  the primary guard). `funsearch_register` stores it in an on-disk **MAP-Elites** DB
  (one elite per behaviour cell + global best). `funsearch_status` returns the best
  programs + the few-shot context the AI writes the next variant from. Ships two
  runnable scorers: `cap_set` and `online_bin_packing`.

- **Web-augmentation — `webaug.py`.** The corpus is finite; the AI has the web.
  `search_directive(problem)` returns a structured search plan (arXiv queries,
  sub-fields + arXiv categories, named results, which mathlas tools to also run) —
  no web call. `add_finding(...)` appends a web-found result to a live JSONL corpus
  through the BM25/sparse channel **with no embedding-model load** (the key
  constraint), retrievable immediately via `search_existing_math` (RRF-fused,
  provenance `web_added`). A dense vector is attached only if a Qwen3 index is already
  resident; otherwise the batch `scripts/reindex_findings.py` fills the backlog later.

- **CLI — `cli.py`.** Auto-routes a single number → numeric; ≥2 integers → OEIS
  sequence; a text arg → search + scaffold/checklist; `mathlas mcp` runs the server.

## Data & toolchains (optional, gitignored, removable)

- **OEIS** (~40 MB, for `identify_sequence`): download `stripped.gz` + `names.gz`
  from `https://oeis.org/` into `reference/downloads/oeis/` (override with
  `MATHLAS_OEIS_DIR`). Absent → honest "data not available", never a fake match.
- **Lean toolchain** (for a real `verify_formal` kernel check): install elan + a
  recent Lean (e.g. 4.30) under `reference/downloads/elan` (see the README one-liner;
  a bare-snippet check, no mathlib). Absent → honest UNDETERMINED.
- **Full index** (`scripts/build_index_mp.py`, offline multi-GPU): builds the
  Qwen3-Embedding index over the permissive theorem corpus; the served `index.npz` +
  sidecar meta. `scripts/reindex_findings.py` embeds the web-finding backlog.

## Citations (methods used)

- **OEIS** — The On-Line Encyclopedia of Integer Sequences, OEIS Foundation Inc.,
  `https://oeis.org`. *(data for `identify_sequence`: the bulk `stripped.gz`/`names.gz`
  files under the OEIS end-user license; matched exactly, locally — not their API.)*
- **Lean 4 / elan** — de Moura & Ullrich, "The Lean 4 Theorem Prover and Programming
  Language," CADE 2021. *(the formal tier's real kernel typecheck.)*
- **Qwen3-Embedding** — Zhang et al., "Qwen3 Embedding: Advancing Text Embedding and
  Reranking Through Foundation Models," arXiv:2506.05176 (2026). *(production embedder;
  the "embed meaning not notation" lesson.)*
- **Reciprocal Rank Fusion** — Cormack, Clarke & Buettcher, SIGIR 2009. *(the hybrid
  dense+BM25 fusion.)*
- **DeepSeekMath-V2** — "Towards Self-Verifiable Mathematical Reasoning,"
  arXiv:2511.22570 (2025). *(generator/verifier separation; the informal-tier design.)*
- **ProofGrader** — Petrov et al., "Reliable Fine-Grained Evaluation of Natural
  Language Math Proofs," arXiv:2510.13888 (ICLR 2026). *(LLM-as-judge is unreliable →
  decompose into atomic conditions.)*
- **Ramanujan Machine** — Raayoni et al., "Generating conjectures on fundamental
  constants with the Ramanujan Machine," *Nature* 590, 67–73 (2021). *(the CF /
  polynomial-recurrence conjecturer in `ramanujan.py`.)*
- **PSLQ** — Ferguson, Bailey & Arno, "Analysis of PSLQ, an integer relation finding
  algorithm," *Math. Comp.* 68, 351–369 (1999). *(the integer-relation engine.)*
- **FunSearch** — Romera-Paredes et al., "Mathematical discoveries from program search
  with large language models," *Nature* 625, 468–475 (2024). *(the generator/harness
  split — mathlas is the harness half.)* **OpenEvolve** — open prior art.
- **MAP-Elites** — Mouret & Clune, "Illuminating search spaces by mapping elites,"
  arXiv:1504.04909 (2015). *(the program DB in `funsearch.py`.)*
- **Reference-only** — **TheoremSearch**, arXiv:2602.05216 (UW Math AI Lab, 2026):
  studied for lessons; their openly-licensed (CC-BY/CC0) **dataset** is used as raw
  data to build our **own** index. Their API/MCP/index/code are **not** a runtime
  dependency.
