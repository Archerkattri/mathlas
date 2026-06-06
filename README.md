# mathlas

> **A tool FOR an AI — no API key, free.** Plug it into Claude Code, Cursor, or
> any MCP client. The **AI is the brain**; mathlas gives it the capabilities it
> lacks: **search over existing math**, **integer-sequence (OEIS) identification**,
> **airtight numeric/formal verification** (incl. a real Lean kernel check),
> structured **needs↔guarantees scaffolds**, honest **provenance** (never
> "novel"), and a **discovery + web-augmentation layer** — a Ramanujan-Machine
> relation/continued-fraction conjecturer, a sandboxed **FunSearch** program-search
> harness, and a web-search **directive + live-corpus** ingestion channel.
> Apache-2.0, mostly-pure-Python.

mathlas is a tool that an AI *uses*, **not** a tool that uses an AI. It **never
calls an LLM and needs no API key** — so it is free and pluggable everywhere.
Most solvable problems stay unsolved not because the formula is missing, but
because nobody connected the *right existing result* to the problem. An AI can do
that connecting — *if it has the right tool*. mathlas is that tool: it returns
**data** (candidates, verdicts, checklists, scaffolds) for the AI to reason over,
and does the parts an AI can't do reliably on its own — airtight verification and
search over its **own** index.

## What mathlas gives the AI (all NO-LLM, returning data)

```
search_existing_math ─▶ mapping_scaffold + applicability_checklist ─▶ (AI judges) ─▶ verify_numeric / verify_formal
   (own index)            (needs↔guarantees, no LLM)                                  (airtight)
```

| Tool | What it does | Airtight? |
|---|---|---|
| `identify_constant(value, basis?)` | a real value → a known closed form + provenance | ✅ independent high-precision re-eval |
| `identify_sequence(terms, max_results?)` | an integer sequence → matching **OEIS** entries (A-number, name, URL) by exact term-match | ✅ exact match vs a local OEIS copy (no fuzzy/LLM) |
| `search_existing_math(query, k, corpus_dir?)` | query → ranked candidate **existing** results (our own dense+BM25+RRF index) | retrieval |
| `verify_numeric(value, closed_form)` | digit-agreement verdict | ✅ different engine, higher precision |
| `verify_formal(statement, lean?)` | runs the **real Lean kernel** on a snippet → typechecks? (else honest UNDETERMINED) | ✅ real kernel check when a snippet+Lean are present (`typecheck ≠ proves-it-applies`) |
| `applicability_checklist(candidate_statement)` | the result's hypotheses as an atomic **checklist** for the AI to check | heuristic parse, no LLM |
| `mapping_scaffold(problem, candidate_statement)` | the **needs↔guarantees** questions + fill-in template for the AI | structured, no LLM |
| `conjecture_relation(value, max_terms?, cf_depth?)` | **Ramanujan Machine**: PSLQ over a *richer* basis (powers/products/zeta) + continued-fraction / polynomial-recurrence **conjectures** | ✅ every candidate numerically **verified** (a *conjecture*, not a proof) |
| `funsearch_evaluate / _register / _status(...)` | **FunSearch harness**: sandbox-score an AI-written program, store it in a MAP-Elites DB, return the few-shot to write a better one | deterministic sandbox, no LLM |
| `search_directive(problem)` | **web-search plan**: arXiv queries + sub-fields + named results + which mathlas tools to also run (mathlas makes **no** web call) | structured, no LLM |
| `add_finding(statement, slogan, source, name?)` | ingest a web-found result into the **live corpus** (BM25, **no model load**) → retrievable via `search_existing_math` | provenance `web_added` |

## Install

```bash
pip install -e .                 # core (numeric + retrieval + verify + scaffolds): NO LLM, NO API key
pip install -e '.[mcp]'          # + the official MCP SDK for the server (a dep-free stdio fallback ships built in)
pip install -e '.[retrieve]'     # + pyarrow, to read the open theorem dataset (real index)
pip install -e '.[embed]'        # + sentence-transformers/torch, for the Qwen3 embedder (offline GPU)
```

## Register with Claude Code (no API key)

```bash
claude mcp add mathlas -- python -m mathlas.server
```

That's it — mathlas now appears as **thirteen** tools the agent can call (the
seven core tools above + the six-tool discovery/web-augmentation layer). (Cursor /
any MCP client: point it at the same `python -m mathlas.server` stdio command.)
The server prefers the official `mcp` SDK and **falls back to a dependency-free
stdio JSON-RPC server** if `mcp` isn't installed, so it always runs.

> **Data tools (optional, gitignored, removable):** `identify_sequence` needs a
> local copy of OEIS, and `verify_formal` needs a Lean toolchain. Both degrade
> honestly (a clear "data/toolchain not available" note, never a fake answer) if
> absent. To enable them:
> ```bash
> # OEIS (~40 MB total) — for identify_sequence
> mkdir -p reference/downloads/oeis
> curl -sSL -o reference/downloads/oeis/stripped.gz https://oeis.org/stripped.gz
> curl -sSL -o reference/downloads/oeis/names.gz    https://oeis.org/names.gz
> # Lean toolchain (~hundreds of MB) — for a real verify_formal kernel check
> export ELAN_HOME="$PWD/reference/downloads/elan"
> curl -sSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
>   | sh -s -- -y --default-toolchain leanprover/lean4:stable --no-modify-path
> ```

### A worked example — an AI using the tools

```
User:  "Does x = cos(x) have a unique solution I can reach by iterating?"

AI →   search_existing_math("contraction mapping unique fixed point complete metric space")
       ← [{name:"Banach Fixed-Point Theorem", statement:"Let (X,d) be a complete metric
            space and T a contraction. Then T has a unique fixed point ...", ...}, ...]

AI →   mapping_scaffold(problem, banach.statement)        # needs↔guarantees questions (no LLM)
AI →   applicability_checklist(banach.statement)
       ← preconditions: ["(X,d) is a complete metric space", "T: X→X is a contraction"]
          conclusion:    "T has a unique fixed point"

AI  (reasons): [0,1] is complete; cos is a contraction there (|cos'|=|sin|≤sin 1<1).
       Every precondition holds ⇒ Banach applies ⇒ unique fixed point, reachable by iteration.

AI →   verify_numeric("0.7390851332151607", "<the Dottie-number closed form, if claimed>")  # airtight check of any numeric claim
```

mathlas supplied the search, the scaffold, the checklist, and the airtight
numeric check; **the AI did the judging**. No LLM was called *inside* mathlas.

## The discovery + web-augmentation layer (NO LLM)

Three capabilities that go past a finite, offline corpus — the AI drives, mathlas
is the deterministic harness.

**1) Ramanujan Machine — `conjecture_relation`.** Beyond `identify_constant`'s
flat basis: PSLQ over a *richer* basis (powers, pairwise products of known
constants, `log`/`exp`/`zeta` values) **and** a Ramanujan-Machine continued-
fraction / polynomial-recurrence search (small integer polynomials `p(n),q(n)`
whose generalized CF `a₀ + b₁/(a₁ + b₂/(a₂ + …))` matches the constant), plus the
simple CF + pattern. **Every candidate is numerically verified** before it is
returned; provenance is `conjectured_relation` — a *verified conjecture, not a
proof*. E.g. `e` → the simple CF `[2; 1,2,1,1,4,1,1,6,…]` (pattern recognised) and
the generalized CF `aₙ=n+1, bₙ=n+1 ⇒ e−1`; `π` → `aₙ=2n+1, bₙ=n² ⇒ 4/π` (verified
to 60+ digits). Cites Raayoni et al., *Nature* 2021, + PSLQ.

**2) FunSearch harness — `funsearch_evaluate / _register / _status`.** *You* (the
AI) are the program generator; mathlas is the deterministic harness — **no LLM**.
`funsearch_evaluate` runs a candidate Python program in a **sandboxed subprocess**
(hard timeout, network stubbed, POSIX CPU/memory rlimits, throwaway cwd) against a
registered scorer and returns its score; `funsearch_register` stores it in an
on-disk **MAP-Elites** program DB (gitignored); `funsearch_status` returns the
best program(s) + the few-shot context to write the next, better one. Ships two
runnable problems — `cap_set` (size of a cap set in ℤ₃ⁿ, FunSearch's headline
result) and `online_bin_packing`. Cites Romera-Paredes et al., *Nature* 2024;
OpenEvolve as the open prior art.

**3) Web-augmented retrieval — `search_directive` + `add_finding`.** The corpus is
finite; the AI has the web. `search_directive(problem)` returns a **structured
search plan** (arXiv query strings, candidate sub-fields + arXiv categories, named
methods/inequalities to look for, which mathlas tools to also run) — mathlas makes
**no** web call. The AI searches, then `add_finding(statement, slogan, source)`
appends the result to the **live corpus** via the BM25/sparse channel **with no
embedding-model load** (the key constraint: growing the index never loads the 8B),
so it is immediately retrievable through `search_existing_math` (RRF-fused),
provenance `web_added`. A dense vector is added only if a Qwen3 index is *already*
loaded in-process; otherwise the batch `scripts/reindex_findings.py` embeds the
backlog later on a GPU box.

```python
import mpmath
from mathlas import conjecture, search_directive, add_finding
print(conjecture(mpmath.e).simple_cf.pattern)     # 'arithmetic (e-type): [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]'
print(search_directive("evaluate sum 1/n^4").named_results[:2])  # ['Euler-Maclaurin', 'Abel summation']

import mathlas.funsearch as fs
prog = fs.get_problem("cap_set").starter_src
r = fs.evaluate(prog, "cap_set")                  # sandboxed score
fs.register(prog, r.score, "cap_set", behavior=r.behavior)
ctx = fs.status("cap_set").few_shot_context        # the prompt YOU write the next variant from
```

## Use without an MCP client — CLI / Python (still no LLM)

```bash
# Numeric: paste a constant, get the verified closed form (airtight, no network)
mathlas 1.6449340668482264364724151666460251892
#   1.64493406684823 -> pi**2/6  [known_form, verified 51 digits]

# Sequence: paste an integer sequence, get the matching OEIS entries (airtight)
mathlas 1,1,2,3,5,8,13,21       #   A000045  Fibonacci numbers ...  https://oeis.org/A000045
mathlas 2 3 5 7 11 13           #   A000040  The prime numbers.     https://oeis.org/A000040

# Problem: search existing math + print the scaffold/checklist an AI reasons over
mathlas "a bounded sequence has a convergent subsequence" --k 5
mathlas "<problem>" --corpus reference/theorem-search-dataset --limit 5000   # real index
mathlas mcp                                                                   # run the MCP server
```

```python
import mpmath
from mathlas import (identify, identify_sequence, mapping_scaffold,
                     applicability_checklist, verify_closed_form)
print(identify(mpmath.zeta(2)))            # 1.64493406684823 -> pi**2/6 [known_form, verified 51 digits]
print(identify_sequence([1,1,2,3,5,8,13,21]).matches[1].a_number)  # 'A000045' (Fibonacci; needs local OEIS data)

from mathlas.server import tool_search_existing_math
hits = tool_search_existing_math("contraction unique fixed point", k=3)["candidates"]
scaf = mapping_scaffold("show x=cos x has a unique fixed point", hits[0]["statement"])
chk  = applicability_checklist(hits[0]["statement"])   # preconditions for the AI to check
ok   = verify_closed_form(mpmath.mpf("1.6449340668482264"), "pi**2/6").ok   # airtight: True
```

## What's verified (light, CPU-only, no API key)

- **Numeric: recovery 8/8, false-positive 0/3** (`benchmarks/numeric_bench.py`).
- **MCP server**: starts under both the official SDK *and* the dependency-free
  fallback; lists all **thirteen** tools; a `search_existing_math` + `verify_numeric`
  round-trip succeeds on the built-in seed corpus.
- **Discovery layer**: `conjecture_relation` recovers + **verifies** the e-type
  simple CF `[2;1,2,1,1,4,1,1,6,…]` and the generalized CFs for `e−1` / `4/π`
  (60+ digits), and PSLQs `ζ(2) → π²/6` over the richer basis; `funsearch_evaluate`
  sandbox-scores a program (cap_set starter 5 → a greedy 16, max 20), with timeout /
  network-block / error capture exercised; `funsearch_register`/`_status` persist a
  MAP-Elites DB and emit the few-shot.
- **Web-augmentation**: `search_directive` returns structured arXiv queries +
  sub-fields + named results; `add_finding` appends a result with **no model load**
  and it is then retrieved through `search_existing_math` (RRF-fused, provenance
  `web_added`).
- **Sequences (OEIS)**: with the local OEIS data present, `identify_sequence`
  exact-matches `[1,1,2,3,5,8,13,21] → A000045` (Fibonacci) and
  `[2,3,5,7,11,13] → A000040` (primes), over ~396k local sequences; the index is
  built once and cached. With no data → an honest "data not available" note.
- **Formal (Lean)**: `verify_formal` runs the **real Lean kernel** — it confirms
  `theorem t : 1 + 1 = 2 := rfl` **typechecks** and reports `1 + 1 = 3 := rfl` as a
  type error. With no Lean toolchain → honest UNDETERMINED (never a fake pass).
- **Retrieval**: paper-level Hit@20 = 15/15 on the test subset, hashing-embedder
  floor (`scripts/eval_retrieval.py`); production uses Qwen3 + the full index
  (`scripts/build_index.py`, offline GPU).

## Bring-your-own-LLM (footnote, optional)

A secondary standalone helper, `solve(problem, retriever, llm=…)`, will run the
needs↔guarantees reasoning loop *for* you **if you supply your own LLM** (any
provider — subclass `LLM`). mathlas ships **no vendor SDK and no default model**;
the default is a no-op `EchoLLM` stub, so the package still imports and runs with
zero API key. This path is convenience only — the primary interface is the MCP
tools above, where the AI is the brain.

## Docs

`docs/00_vision.md` · `docs/01_landscape.md` (research sweep) ·
`docs/02_mvp_spec.md` · `docs/03_theoremsearch_analysis.md` (reference-only study) ·
**`docs/04_build.md`** (what was built + methods + citations).

## Positioning

The closest system, **TheoremSearch** (UW Math AI Lab), is recall-optimized
retrieval only — it finds the *statement you already formulated* and never checks
applicability, routes across tools, accepts numeric inputs, or labels provenance.
mathlas adds exactly those **and** the design that makes it composable: it is a
*tool an AI plugs in*, not a closed lab agent or an LLM wrapper. TheoremSearch is
**reference-only**; we reuse just their openly-licensed (CC-BY/CC0) **dataset** as
raw data to build our **own** index — not their API, MCP, index, or code.
