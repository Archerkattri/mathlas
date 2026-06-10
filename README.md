<p align="center">
  <img src="https://raw.githubusercontent.com/Archerkattri/mathlas/main/assets/banner.png" alt="mathlas" width="680">
</p>

# mathlas

[![PyPI](https://img.shields.io/pypi/v/mathlas-mcp)](https://pypi.org/project/mathlas-mcp/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20618603-1682D4.svg)](https://doi.org/10.5281/zenodo.20618603)
[![Glama score](https://glama.ai/mcp/servers/Archerkattri/mathlas/badges/score.svg)](https://glama.ai/mcp/servers/Archerkattri/mathlas)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

> **An airtight-math tool an AI *uses* — no LLM, no API key, free.** Plug it into
> Claude Code, Cursor, or any MCP client. The **AI is the brain**; mathlas is the
> **hands** — it gives the AI the capabilities it lacks and returns *data*
> (candidates, verdicts, checklists, scaffolds) for the AI to reason over.
> Apache-2.0. The code is free for any use; published corpus/index artifacts carry their own per-source terms (CC-BY/CC0).

---

## Is this for you?

- **You use Claude Code / Cursor and want your AI to stop hallucinating math** — `search_existing_math` finds the real theorem from a 3.68M-doc index; `verify_numeric` and `verify_formal` check claims with zero hallucination risk.
- **You have a numeric constant or integer sequence you can't identify** — `identify_constant` runs PSLQ + closed-form matching (50-digit precision); `identify_sequence` does an exact OEIS term-match.
- **You need the formal (Lean/mathlib) name of a result** — `search_formal_math` proxies the public Loogle + LeanSearch services and returns declaration names + types, provenance-labeled.
- **You're building an agent pipeline that needs airtight math in the loop** — all 12 tools are pure data-returning MCP tools, no LLM inside, composable with any framework.

---

## Install & register with Claude Code (no API key)

One line, nothing to install first (needs [uv](https://docs.astral.sh/uv/)):

```bash
claude mcp add mathlas -- uvx mathlas-mcp
```

`uvx mathlas-mcp` fetches + runs the server in an isolated env on first use. Prefer pip?

```bash
pip install mathlas-mcp              # core: numeric + retrieval + verify + scaffolds
pip install 'mathlas-mcp[mcp]'       # + official MCP SDK
pip install 'mathlas-mcp[retrieve]'  # + pyarrow, to read the real index
pip install 'mathlas-mcp[embed]'     # + sentence-transformers/torch, for the Qwen3 embedder

claude mcp add mathlas -- python -m mathlas.server
```

mathlas now appears as **twelve** tools the agent can call. The server prefers the official `mcp` SDK and **falls back to a dependency-free stdio JSON-RPC server** if `mcp` isn't installed — it always runs. (Cursor / any MCP client: point it at the same `uvx mathlas-mcp` or `python -m mathlas.server` stdio command.)

> **Optional local data (degrades honestly):** `identify_sequence` wants a local OEIS copy; `verify_formal` wants a Lean toolchain. Without them the tools return a clear "data/toolchain not available" — never a fake answer. See [`RESULTS.md`](RESULTS.md) for the one-line setup of each.

---

## A worked example — an AI using the tools

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

mathlas supplied the search, the checklist, and the airtight numeric check. The AI did the judging. **No LLM was called inside mathlas.**

---

## Results

The discipline is **airtight-or-nothing**: a result is an independently-checkable fact or an honest "nothing." The **false-positive rate is 0 across every tier** (full tables + commands in [`RESULTS.md`](RESULTS.md)):

| Tier | Recovery@known | False-positive | Why it's airtight | Benchmark |
|---|---|---|---|---|
| Numeric (`identify_constant`) | 8/8 | 0/3 | independent high-precision re-eval (50–51 digits) | `benchmarks/numeric_bench.py` |
| Sequence (`identify_sequence`) | 8/8 (all top-1) | 0/3 | exact term-match vs local OEIS (~400k seqs) | `benchmarks/tier_bench.py` |
| Formal (`verify_formal`) | 7/7 verdicts | — | real Lean 4.30 kernel typecheck | `benchmarks/tier_bench.py` |
| Ramanujan (`conjecture_relation`) | 6/6 | 0/2 | PSLQ + CF, every hit re-verified ≥25 digits | `benchmarks/tier_bench.py` |
| Applicability moat | 15/15 decomp + 6/6 catch | — | atomic preconditions, misapplication traps | `benchmarks/moat_bench.py` |
| FunSearch + web-aug | 14/14 | — | sandbox containment (network / timeout / memory) | `benchmarks/tools_bench.py` |

**The 3.68M-doc index.** `search_existing_math` is served from a **3,683,428-document** dense index (Qwen3-Embedding-8B, 4096-d): the **1.34M** permissive CC-BY/CC0 TheoremSearch subset + **2.34M** slogan-embedded arXiv-math documents from Dolma, dense + Okapi-BM25 + RRF. Honest headline recall at full 3.68M scale: **R@1 0.614 / R@10 0.832** querying by a document's raw *body* against its slogan-embedded entry — the hard **cross-representation** self-recall regime. (At the earlier 1.635M build, the easier same-representation slogan→slogan self-recall was R@1 0.977 / R@10 0.998 on its 81,833-doc held-out split.)

## The self-augmenting loop — beating TheoremSearch

On TheoremSearch's own **110 human-written queries**, baseline mathlas hits a coverage floor — TheoremSearch withheld 85% of their private 9.2M corpus, so 95 target papers are unreachable for any open system. The AI then runs the loop: for each missing theorem it web-finds the real statement, embeds it with the same Qwen3-Embedding-8B, and `add_finding(dense_vec=…)` fuses it through the dense channel at runtime:

| Method | theorem Hit@20 | paper Hit@20 |
|---|---|---|
| Google (`site:arxiv.org`) | — | 37.8% |
| ChatGPT 5.2 w/ Search | 19.8% | — |
| Gemini 3 Pro | 27.0% | — |
| **TheoremSearch** (Qwen3-8B, full private 9.2M) | 45.0% | 56.8% |
| mathlas — baseline (corpus-only) | 10.0% | 13.6% |
| **mathlas — after self-augmenting web loop** | **59.1% (65/110)** | **70.0% (77/110)** |

The 10.0% floor exists *because* TheoremSearch withheld 85% of their corpus — the loop repairs that coverage gap. Reproduce with `benchmarks/webaug_110_bench.py`.

---

## The 12 tools

```
search_existing_math ─▶ mapping_scaffold + applicability_checklist ─▶ (AI judges) ─▶ verify_numeric / verify_formal
   (own index)            (needs↔guarantees, no LLM)                                  (airtight)
```

**Core four** — what most agents use:

| Tool | What it does |
|---|---|
| `search_existing_math(query, k)` | query → ranked results from the **3.68M-doc** dense + BM25 + RRF index |
| `identify_constant(value)` | a real value → known closed form + provenance (50-digit re-eval) |
| `verify_numeric(value, closed_form)` | digit-agreement verdict — different engine, higher precision |
| `verify_formal(statement, lean?, proof?)` | runs the **real Lean kernel** — typecheck a snippet, or pass `proof` to **kernel-check a full Lean 4 proof**: `VERIFIED_PROOF` / `REFUTED` (the kernel's exact error, for the repair loop) / honest UNDETERMINED |

**Full toolkit:**

| Tool | What it does |
|---|---|
| `search_formal_math(query, backend)` | mathlib declaration names + types via the public **Loogle** (pattern/type) + **LeanSearch** (natural language) services, provenance-labeled; honest "service unavailable" — with a 7-day on-disk cache that serves the last good response when a service is down, clearly labeled `(cached, <age> old)` |
| `identify_sequence(terms)` | integer sequence → matching OEIS entries (exact term-match) |
| `applicability_checklist(statement)` | result's hypotheses as an atomic checklist for the AI to mark |
| `mapping_scaffold(problem, statement)` | needs↔guarantees questions + fill-in template |
| `conjecture_relation(value)` | Ramanujan Machine: PSLQ over rich basis + CF/recurrence conjectures |
| `funsearch(action, problem_id, …)` | FunSearch harness in one tool — `action=evaluate` (sandbox-score an AI-written program), `register` (MAP-Elites DB), `status` (best + few-shot) |
| `search_directive(problem)` | web-search plan: arXiv queries + sub-fields + which tools to run |
| `add_finding(statement, slogan, source)` | ingest a web-found result into the live corpus |

All tools return data. No tool calls an LLM. `search_formal_math` is the one tool that itself makes a web call (to the public Loogle/LeanSearch services); everything else is fully local.

### Proof checking — the repair loop

`verify_formal` doesn't just typecheck statements: give it a proposition and *your* Lean 4 proof, and the **real kernel** checks the full declaration. mathlas never writes a proof (the generator/verifier split is absolute) — but when your proof is wrong, the kernel tells you *exactly why*, verbatim, in `kernel_error`. That turns proof writing into a tight loop: **the agent writes a proof → mathlas's kernel says exactly what's wrong → the agent repairs and re-calls.**

```jsonc
verify_formal(statement="∀ n : Nat, n + 0 = n", proof="by\n  intro n\n  rfl")
// → {"proof_status": "VERIFIED_PROOF", "checked": true, ...}
verify_formal(statement="2 + 2 = 5", proof="rfl")
// → {"proof_status": "REFUTED", "kernel_error": "error: Not a definitional equality:
//     the left-hand side 2 + 2 is not definitionally equal to the right-hand side 5 ...", ...}
```

No fake passes, by construction: `sorry`/`admit` holes are **REJECTED** (Lean itself exits 0 on a sorried proof — mathlas scans the source *and* the kernel's `sorryAx` diagnostics); a missing toolchain, a timeout (60 s cap), or an import this bare toolchain can't resolve all return an honest `UNDETERMINED`, never a verdict. The whole contract is pinned by `tests/test_proof_check.py` (20 tests against the real Lean 4.30 kernel: correct term *and* tactic-block proofs verified, wrong proofs refuted with the kernel's message, sorried proofs rejected, toolchain-absent honest).

---

## CLI / Python

```bash
mathlas 1.6449340668482264364724151666460251892   # -> pi**2/6  [verified 51 digits]
mathlas 1,1,2,3,5,8,13,21                          # -> A000045 Fibonacci  https://oeis.org/A000045
mathlas "a bounded sequence has a convergent subsequence" --k 5   # search + scaffold
mathlas mcp                                                        # run the MCP server
```

```python
import mpmath
from mathlas import identify, identify_sequence, mapping_scaffold, applicability_checklist
print(identify(mpmath.zeta(2)))            # -> pi**2/6 [verified 51 digits]
print(identify_sequence([1,1,2,3,5,8,13,21]).matches[1].a_number)  # -> 'A000045'
```

---

## Docs

- [`RESULTS.md`](RESULTS.md) — every tool's validation, reproduced, with commands.
- [`docs/methods.md`](docs/methods.md) — architecture, design decisions, citations.
- [`docs/05_open_dataset.md`](docs/05_open_dataset.md) — the open dataset & the index.
- [`docs/02_eval_vs_theoremsearch.md`](docs/02_eval_vs_theoremsearch.md) — the retrieval head-to-head.
- [`docs/REGISTRY_PUBLISH.md`](docs/REGISTRY_PUBLISH.md) — publishing to the official MCP registry.

## Positioning — retrieval is table stakes; verification is the moat

**Credit where due:** the closest system, **TheoremSearch** (UW Math AI Lab), now ships a production REST API **and its own MCP endpoint** (`api.theoremsearch.com/mcp`) over a 9.2M-document corpus — on raw recall over math literature it is the system to beat, and "we're MCP-native, they're a lab tool" is no longer a differentiator. We reuse only their openly-licensed (CC-BY/CC0) dataset subset as raw data for our own index — not their API, MCP, index, or code.

What no competitor has is everything that happens **after** retrieval:

- **Verification tiers** — `verify_numeric` (independent 50-digit re-evaluation) and `verify_formal` (a **real Lean kernel** typecheck — including **full proof checking** with the kernel's error returned verbatim for agent repair loops — or an honest UNDETERMINED). Retrieval hands you a candidate; mathlas can also *check the claim* and *check your proof of it*.
- **`applicability_checklist`** — decomposes a candidate theorem into atomic preconditions the AI verifies one by one, catching misapplications (open vs closed interval, infinite vs finite group). **No competitor has one.**
- **The self-augmenting `add_finding` loop** — the AI web-finds a missing statement, embeds it, and fuses it into the live index at runtime: **59.1% vs TheoremSearch's 45.0% theorem Hit@20** on their own 110-query benchmark (see above).
- **Zero-false-positive discipline** — every tier returns an independently-checkable fact or an honest "nothing"; measured false-positive rate is **0 across all tiers** ([`RESULTS.md`](RESULTS.md)).
- **Free, no API key, provenance-labeled** — every result carries where it came from (`known_constant`, `conjectured_relation`, `web_added`, `external:loogle`, …), and the index is built 100% from openly-licensed data.

| | **mathlas** | TheoremSearch | LeanSearch / Loogle | Wolfram MCP | sympy-mcp |
|---|---|---|---|---|---|
| Informal math retrieval | ✅ 3.68M docs, open | ✅ 9.2M docs (~85% private) | ❌ (mathlib decls only) | ❌ | ❌ |
| Formal (mathlib) search | ✅ proxies both → one MCP tool | ❌ | ✅ (is exactly this) | ❌ | ❌ |
| Numeric verification | ✅ airtight 50-digit re-eval | ❌ | ❌ | ⚠️ CAS eval | ⚠️ CAS (no claim-check framing) |
| Formal verification | ✅ real Lean kernel (statements **and full proofs**, repair-loop errors) | ❌ | ❌ (search, not check) | ❌ | ❌ |
| Applicability checklist | ✅ **unique** | ❌ | ❌ | ❌ | ❌ |
| Self-augmenting corpus | ✅ `add_finding` (59.1 vs 45.0 Hit@20) | ❌ | ❌ | ❌ | ❌ |
| Constant/sequence ID | ✅ PSLQ + OEIS + Ramanujan-Machine | ❌ | ❌ | ⚠️ some | ❌ |
| Provenance labels | ✅ every result | ❌ | n/a | ❌ | n/a |
| Cost / key | **free, no key** | free endpoint | free | **paid Wolfram API key** | free |
| MCP | ✅ stdio, `uvx` one-liner | ✅ remote endpoint | ❌ (mathlas proxies them) | ✅ | ✅ |

(sympy-mcp is a fine CAS-manipulation server — its scope barely overlaps: it rewrites expressions you give it; mathlas finds, scopes, and verifies *existing* math.)

---

## Official MCP registry

mathlas is published as `io.github.Archerkattri/mathlas` (see [`docs/REGISTRY_PUBLISH.md`](docs/REGISTRY_PUBLISH.md) and [`server.json`](server.json)).

mcp-name: io.github.Archerkattri/mathlas
