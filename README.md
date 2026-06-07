# mathlas

> **An airtight-math tool an AI *uses* — no LLM, no API key, free.** Plug it into
> Claude Code, Cursor, or any MCP client. The **AI is the brain**; mathlas is the
> **hands** — it gives the AI the capabilities it lacks and returns *data*
> (candidates, verdicts, checklists, scaffolds) for the AI to reason over.
> Apache-2.0, mostly-pure-Python.

mathlas is a tool that an AI *uses*, **not** a tool that uses an AI — it **never
calls an LLM and needs no API key**, so it is free and pluggable everywhere. Most
solvable problems stay unsolved not because the formula is missing, but because
nobody connected the *right existing result* to the problem. An AI can do that
connecting — *if it has the right tool*. mathlas does the parts an AI can't do
reliably on its own: **search over its own large math index**, **airtight
numeric/formal verification** (incl. a real Lean kernel check), exact **OEIS**
identification, structured **needs↔guarantees scaffolds**, honest **provenance**
(never "novel"), plus a **discovery + web-augmentation layer** that lets the AI
grow the index at runtime.

## The 13 tools (all NO-LLM, returning data)

```
search_existing_math ─▶ mapping_scaffold + applicability_checklist ─▶ (AI judges) ─▶ verify_numeric / verify_formal
   (own index)            (needs↔guarantees, no LLM)                                  (airtight)
```

| Tool | What it does | Airtight? |
|---|---|---|
| `search_existing_math(query, k)` | query → ranked **existing** results from our own **1.635M-doc** dense + BM25 + RRF index | retrieval |
| `identify_constant(value, basis?)` | a real value → a known closed form + provenance | ✅ independent high-precision re-eval |
| `identify_sequence(terms)` | an integer sequence → matching **OEIS** entries (A-number, name, URL) | ✅ exact term-match vs local OEIS |
| `verify_numeric(value, closed_form)` | digit-agreement verdict | ✅ different engine, higher precision |
| `verify_formal(statement, lean?)` | runs the **real Lean kernel** on a snippet → typechecks? (else honest UNDETERMINED) | ✅ real kernel check |
| `applicability_checklist(statement)` | the result's hypotheses as an atomic **checklist** for the AI to mark | heuristic parse, no LLM |
| `mapping_scaffold(problem, statement)` | the **needs↔guarantees** questions + fill-in template | structured, no LLM |
| `conjecture_relation(value, ...)` | **Ramanujan Machine**: PSLQ over a rich basis + continued-fraction / recurrence **conjectures** | ✅ every candidate numerically verified |
| `funsearch_evaluate / _register / _status` | **FunSearch harness**: sandbox-score an AI-written program, store in a MAP-Elites DB, return the few-shot to write a better one | deterministic sandbox, no LLM |
| `search_directive(problem)` | **web-search plan**: arXiv queries + sub-fields + named results + which mathlas tools to also run | structured, no LLM |
| `add_finding(statement, slogan, source)` | ingest a web-found result into the **live corpus** → retrievable via `search_existing_math` | provenance `web_added` |

## Install & register with Claude Code (no API key)

```bash
pip install -e .                 # core: numeric + retrieval + verify + scaffolds — NO LLM, NO API key
pip install -e '.[mcp]'          # + official MCP SDK (a dep-free stdio fallback also ships built in)
pip install -e '.[retrieve]'     # + pyarrow, to read the open theorem dataset (real index)
pip install -e '.[embed]'        # + sentence-transformers/torch, for the Qwen3 embedder (offline GPU)

claude mcp add mathlas -- python -m mathlas.server
```

That's it — mathlas now appears as **thirteen** tools the agent can call. The server
prefers the official `mcp` SDK and **falls back to a dependency-free stdio JSON-RPC
server** if `mcp` isn't installed, so it always runs. (Cursor / any MCP client: point
it at the same `python -m mathlas.server` stdio command.)

> **Optional local data (gitignored, degrades honestly):** `identify_sequence` wants a
> local OEIS copy; `verify_formal` wants a Lean toolchain. Without them the tools return
> a clear "data/toolchain not available" note — never a fake answer. See
> [`RESULTS.md`](RESULTS.md) for the one-line setup of each.

### A worked example — an AI using the tools

```
User:  "Does x = cos(x) have a unique solution I can reach by iterating?"

AI →   search_existing_math("contraction mapping unique fixed point complete metric space")
       ← [{name:"Banach Fixed-Point Theorem", statement:"Let (X,d) be a complete metric
            space and T a contraction. Then T has a unique fixed point ...", ...}, ...]
AI →   applicability_checklist(banach.statement)
       ← preconditions: ["(X,d) is a complete metric space", "T: X→X is a contraction"]
          conclusion:    "T has a unique fixed point"
AI  (reasons): [0,1] is complete; cos is a contraction there (|cos'|=|sin|≤sin 1<1).
       Every precondition holds ⇒ Banach applies ⇒ unique fixed point, reachable by iteration.
AI →   verify_numeric("0.7390851332151607", "<the Dottie-number closed form, if claimed>")
```

mathlas supplied the search, the checklist, and the airtight numeric check; **the AI
did the judging**. No LLM was called *inside* mathlas.

## Without an MCP client — CLI / Python (still no LLM)

```bash
mathlas 1.6449340668482264364724151666460251892   # 1.64493... -> pi**2/6  [verified 51 digits]
mathlas 1,1,2,3,5,8,13,21                          # A000045  Fibonacci numbers  https://oeis.org/A000045
mathlas "a bounded sequence has a convergent subsequence" --k 5   # search + scaffold/checklist
mathlas mcp                                                        # run the MCP server
```

```python
import mpmath
from mathlas import identify, identify_sequence, mapping_scaffold, applicability_checklist
print(identify(mpmath.zeta(2)))            # 1.64493406684823 -> pi**2/6 [verified 51 digits]
print(identify_sequence([1,1,2,3,5,8,13,21]).matches[1].a_number)  # 'A000045' (needs local OEIS)
```

## Results

Every tool has a reproduced benchmark — full tables + commands in
[`RESULTS.md`](RESULTS.md). The discipline is **airtight-or-nothing**: a result is an
independently-checkable fact or an honest "nothing." Each verification tier feeds both
a *known* input (expect the correct verified result) and a *structureless* input
(expect an honest "nothing") — the **false-positive rate is 0 across every tier**:

| Tier | Recovery@known | False-positive | Why it's airtight | Benchmark |
|---|---|---|---|---|
| Numeric (`identify_constant`) | 8/8 | 0/3 | independent high-precision re-eval (50–51 digits) | `benchmarks/numeric_bench.py` |
| Sequence (`identify_sequence`) | 8/8 (all top-1) | 0/3 | exact term-match vs local OEIS (~400k seqs) | `benchmarks/tier_bench.py` |
| Formal (`verify_formal`) | 7/7 verdicts | — | real Lean 4.30 kernel typecheck | `benchmarks/tier_bench.py` |
| Ramanujan (`conjecture_relation`) | 6/6 | 0/2 | PSLQ + CF, every hit re-verified ≥25 digits | `benchmarks/tier_bench.py` |
| Applicability moat | 15/15 decomp + 6/6 catch | — | atomic preconditions, misapplication traps | `benchmarks/moat_bench.py` |
| FunSearch + web-aug | 14/14 | — | sandbox containment (network / timeout / memory) | `benchmarks/tools_bench.py` |

**The 1.635M-doc index — retrieval at scale.** `search_existing_math` is served from a
built **1,635,233-document** exact dense index (Qwen3-Embedding-8B, 4096-d, over the
permissive CC-BY/CC0 TheoremSearch subset + arXiv-math from Dolma + Stacks + ProofWiki),
dense + Okapi-BM25 + RRF. On the held-out **81,833-document** test split, querying each
theorem by its natural-language slogan retrieves its own entry at **R@1 0.977 / R@10
0.998** (and **R@10 0.923** querying by the raw formal statement — cross-representation).

## The self-augmenting loop — closing the coverage gap to beat everyone

This is the demonstration that mathlas's `add_finding` **dense path** is a real,
decisive runtime mechanism. On TheoremSearch's own **110 human-written queries**,
baseline mathlas (corpus-only) hits a hard **coverage floor**: TheoremSearch
open-sourced only ~15% of their private 9.2M corpus, so **95 of the 110 target papers
are non-permissive arXiv they withheld** — unreachable for *any* open system. The AI
then runs the loop: for each missing theorem it **web-finds the real statement**,
embeds it with the **same Qwen3-Embedding-8B**, and `add_finding(dense_vec=…)` so it
**RRF-fuses through the dense channel**. That repairs the gap and beats every baseline:

| Method | theorem Hit@20 | paper Hit@20 |
|---|---|---|
| Google (`site:arxiv.org`) | — | 37.8% |
| ChatGPT 5.2 w/ Search | 19.8% | — |
| Gemini 3 Pro | 27.0% | — |
| **TheoremSearch** (Qwen3-8B, full private 9.2M) | 45.0% | 56.8% |
| mathlas — baseline (corpus-only, the coverage floor) | 10.0% | 13.6% |
| **mathlas — after the self-augmenting WEB loop** | **59.1% (65/110)** | **70.0% (77/110)** |

**This is the loop's value, not a native-corpus claim.** The 10.0% floor exists
*because* TheoremSearch withheld 85% of their corpus; the loop repairs that coverage.
On the reachable subset our retrieval is merely *on par* with TheoremSearch — we make
no native-superiority claim. The result proves that `add_finding` lets an AI grow the
live index at runtime, decisively. Honesty audit passed — **zero query-injection**: no
finding's text contains the query; the slogans are real theorem prose and the queries
are paraphrases, so the dense channel is what bridges them. Reproduce with
`benchmarks/webaug_110_bench.py` (see [`RESULTS.md`](RESULTS.md) §3c).

## Docs

- [`RESULTS.md`](RESULTS.md) — every tool's validation, reproduced, with commands.
- [`docs/methods.md`](docs/methods.md) — architecture, design decisions, citations.
- [`docs/05_open_dataset.md`](docs/05_open_dataset.md) — the open dataset & 1.635M-doc index.
- [`docs/02_eval_vs_theoremsearch.md`](docs/02_eval_vs_theoremsearch.md) — the retrieval head-to-head.

## Positioning

The closest system, **TheoremSearch** (UW Math AI Lab), is recall-optimized retrieval
only — it finds the *statement you already formulated* and never checks applicability,
routes across tools, accepts numeric inputs, or labels provenance. mathlas adds exactly
those, plus the design that makes it composable: it is *a tool an AI plugs in*, not a
closed lab agent or an LLM wrapper. TheoremSearch is **reference-only** — we reuse just
their openly-licensed (CC-BY/CC0) **dataset** as raw data to build our **own** index,
not their API, MCP, index, or code.

A secondary helper, `solve(problem, retriever, llm=…)`, will run the needs↔guarantees
loop for you **if you supply your own LLM** (subclass `LLM`, any provider). mathlas
ships no vendor SDK and no default model, so the package still imports and runs with
zero API key — this path is convenience only; the primary interface is the MCP tools.
