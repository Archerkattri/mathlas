# mathlas

> **A tool FOR an AI — no API key, free.** Plug it into Claude Code, Cursor, or
> any MCP client. The **AI is the brain**; mathlas gives it the capabilities it
> lacks: **search over existing math**, **airtight numeric/formal verification**,
> structured **needs↔guarantees scaffolds**, and honest **provenance** (never
> "novel"). Apache-2.0, mostly-pure-Python.

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
| `search_existing_math(query, k, corpus_dir?)` | query → ranked candidate **existing** results (our own dense+BM25+RRF index) | retrieval |
| `verify_numeric(value, closed_form)` | digit-agreement verdict | ✅ different engine, higher precision |
| `verify_formal(statement, lean?)` | Lean verdict | stub (clearly marked; interface ready) |
| `applicability_checklist(candidate_statement)` | the result's hypotheses as an atomic **checklist** for the AI to check | heuristic parse, no LLM |
| `mapping_scaffold(problem, candidate_statement)` | the **needs↔guarantees** questions + fill-in template for the AI | structured, no LLM |

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

That's it — mathlas now appears as six tools the agent can call. (Cursor / any
MCP client: point it at the same `python -m mathlas.server` stdio command.) The
server prefers the official `mcp` SDK and **falls back to a dependency-free stdio
JSON-RPC server** if `mcp` isn't installed, so it always runs.

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

## Use without an MCP client — CLI / Python (still no LLM)

```bash
# Numeric: paste a constant, get the verified closed form (airtight, no network)
mathlas 1.6449340668482264364724151666460251892
#   1.64493406684823 -> pi**2/6  [known_form, verified 51 digits]

# Problem: search existing math + print the scaffold/checklist an AI reasons over
mathlas "a bounded sequence has a convergent subsequence" --k 5
mathlas "<problem>" --corpus reference/theorem-search-dataset --limit 5000   # real index
mathlas mcp                                                                   # run the MCP server
```

```python
import mpmath
from mathlas import identify, mapping_scaffold, applicability_checklist, verify_closed_form
print(identify(mpmath.zeta(2)))            # 1.64493406684823 -> pi**2/6 [known_form, verified 51 digits]

from mathlas.server import tool_search_existing_math
hits = tool_search_existing_math("contraction unique fixed point", k=3)["candidates"]
scaf = mapping_scaffold("show x=cos x has a unique fixed point", hits[0]["statement"])
chk  = applicability_checklist(hits[0]["statement"])   # preconditions for the AI to check
ok   = verify_closed_form(mpmath.mpf("1.6449340668482264"), "pi**2/6").ok   # airtight: True
```

## What's verified (light, CPU-only, no API key)

- **Numeric: recovery 8/8, false-positive 0/3** (`benchmarks/numeric_bench.py`).
- **MCP server**: starts under both the official SDK *and* the dependency-free
  fallback; lists all six tools; a `search_existing_math` + `verify_numeric`
  round-trip succeeds on the built-in seed corpus.
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
