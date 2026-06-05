# math_engine — landscape synthesis (5-stream sweep, 2026-06-05)

> Sources: 5 web-research agents (mapping methods · labs' papers · GitHub prior-art · community/demand · math-representation). Primary-sourced; unverified items flagged inline.

## TL;DR
The **retrieval** half ("describe → find the existing theorem") is increasingly solved and **already partly shipped** (TheoremSearch, Feb 2026). The unbuilt, defensible white space is the **orchestration + verification + provenance** layer: a **router** that maps a problem to the right existing-math tool, an **automatic check that the retrieved math actually applies to *your* instance**, and a **retrieved-vs-novel label**. Every credible lab system is the same loop — *propose → map-to-existing → verify → keep* — and the hardest, least-solved component is **automatic verification-of-applicability for INFORMAL problems** + the **retrieved-vs-novel gate**. Build narrow on a domain with airtight automatic verification first (numeric/sequence/constant), prove the loop, widen to formal (Lean, free verify), then attack the informal dream where the real moat lives.

## 1. The competitor that already shipped — TheoremSearch
theoremsearch.com (UW Math AI Lab, arXiv 2602.05216, Feb 2026). NL "describe a result" → 9.2M theorems (arXiv + Stacks + ProofWiki + more). **45% Hit@20**, beats GPT-5.2-search (19.8%) & Gemini-3-Pro (27%). Public **REST + MCP** server (`api.theoremsearch.com/mcp`). **Gaps = our wedge:** retrieval-ONLY (no applicability check), research-math-only corpus, model+index **closed** (only ingest code + HF dataset open), targets mathematicians (not cross-field engineers).

## 2. The universal pattern (all labs)
`propose (LLM) → map-to-existing → verify → keep`, looped. "Map-to-existing" takes 3 forms: **emergent** web+memory (Aletheia, GPT-5, unit-distance), **engineered premise-retrieval** (ReProver, LeanSearch), **evolutionary reuse** (FunSearch, AlphaEvolve). The **verifier** is the discriminator: formal/auto (Lean, `evaluate()`) = trustworthy+scalable; NL critic (Aletheia) = fallible; human (unit-distance) = gold but doesn't scale. Two tricks to copy: **thinking/output decoupling + self-critique** (Aletheia, 65.7→95.1% on IMO-ProofBench-Adv); **cheap→expensive verify cascade** (AlphaEvolve).

## 3. The deep insight — "needs ↔ guarantees" matching
The unit-distance disproof (arXiv 2605.20695) shows the valuable step = *"my problem NEEDS an object with property X; theorem T GUARANTEES X."* Not keyword/formula similarity — matching a problem's **requirement** to a theorem's **guarantee**, across sub-fields, while retaining "unfashionable" tools humans pruned (the model used Golod-Shafarevich class-field towers everyone had deprioritized). This is the core IP and has **no open implementation for informal math**.

## 4. The white space (converged across all 5 streams)
1. **Unified ROUTER** across retrieve / PSLQ / OEIS / Lean / RIES / FunSearch — *"conspicuously absent"* (only a rule-based, compute-only "Math Intelligence Router" Claude skill exists).
2. **Automatic verification-of-applicability for INFORMAL problems** — everyone does formal-only (free verify) or delegates informal verification to humans. **THE load-bearing missing piece** (named independently by 3 of 5 agents).
3. **Retrieved-vs-novel provenance gate** — unsolved everywhere; the GPT-5 Oct-2025 embarrassment (claimed 10 "open" Erdős solved → Bloom corrected: they were *existing references*). We *want* retrieval, so we must **label** it correctly (Aletheia's "subconscious plagiarism" risk).
4. **Cross-field reach** — serve the engineer/physicist who doesn't know the math's *name* or *home field* (the trapezoidal-rule rediscovery: a 1994 medical paper re-derived 2000-year-old math, peer-reviewed, cited dozens of times). Biggest latent market.

## 5. Reusable crumbs (permissive, buildable-on)
- **Formal retrieve:** ReProver (MIT, premise retriever lifts 47.5→51.4%), LeanSearch v2 (open + live API, CC-BY), Loogle (Apache, exact structural), LeanDojo-v2 (MIT, premise data).
- **Numeric/symbolic identify:** `mpmath.identify` / `sympy.nsimplify` (BSD, PSLQ), OEIS wrappers (`python-oeis`/`joeis`), RIES (⚠️ GPL-3.0 — call as service, don't link), Ramanujan Machine (~351★).
- **Discovery:** OpenEvolve (MIT, ~6.5k★ — the open AlphaEvolve/FunSearch engine), CodeEvolve.
- **Verify:** Lean + LeanDojo gym (MIT) — correct-by-construction, highest-leverage free component.
- **Formula embeddings:** TangentCFT (SLT+OPT), SSEmb (structure+semantic, ARQMath-3 SOTA), MathBERT.
- **Provenance schema:** Aletheia's Autonomous-Math-Levels + Human-AI Interaction Cards (Apache/CC-BY).
- **TheoremSearch — reference only:** their openly-licensed (CC-BY/CC0) *dataset* is usable as data; their **API / MCP / running system are NOT a runtime dependency** — we build our own retrieval.

## 6. The "math-for-math model" (the thesis) — directionally right, hybrid in practice
Do NOT build a from-scratch math-only model first. Proven path = **structure-aware retrieval** (GNN/RGCN premise selection +25–34% over text; graph-invariant HOL embeddings 83→90.3%) + **symbolic apply/verify** (Lean kernel) + **LLM glue**, with Lean formal terms as substrate. The dedicated math→math model (**Mathesis**, arXiv 2601.00125: hypergraph-in / transform-out + differentiable symbolic kernel, Jan 2026) is *being attempted but UNPROVEN* (preliminary miniF2F only). Novel representation contribution = **meaning-not-appearance embeddings via equivalence-supervised (e-graph / equality-saturation) training**. Bottleneck: autoformalization is leaky (honest SOTA ≤22.5% Pass@128 on grad theorems; agentic ~52%).

## 7. Demand (real, articulated by mathematicians)
Billey & Tenner **"Fingerprint databases for theorems"** (arXiv 1304.3866, 2013) = the manifesto: index theorems by canonical, language-independent fingerprints; per-domain DBs (OEIS, FindStat, House of Graphs, ISC) + a general finder. Loved precedents: **OEIS+Superseeker, ISC/RIES**. Segments by intensity: AI-for-math tooling builders (fastest) > Lean/Isabelle users (captive, 5+ tools spawned) > **cross-field engineers/physicists (biggest latent, hardest to reach)** > grad students > competition mathematicians. (Reddit blocked in scan env — community-volume signal is indirect.)

## 8. Recommended build path
- **Architecture:** ROUTER over {retrieve, identify, discover} → **VERIFY** (automatic where possible) → **PROVENANCE** label. The router + applicability-verify + retrieved-vs-novel gate are the novel, unbuilt contributions.
- **MVP beachhead (recommended): numeric / sequence / constant → known formula.** Airtight automatic verification (high-precision numeric match), loved demo (OEIS/ISC, unified+routed), full loop works *today* with permissive tools (mpmath-PSLQ + OEIS), weeks not months. Proves the engine end-to-end + dodges the unsolved informal-verify gap.
- **Then widen:** formal/Lean (free verify, build on ReProver/LeanSearch) → informal "describe → find + verify-applicability" (the dream + the real moat + biggest market).
- **Benchmark FIRST:** (problem → known result) pairs with automatic verification. For numeric this is easy + airtight — and it's how we avoid the GPT-5 over-claim.
- **Open from day 1** (Apache-2.0); ship the working MVP before publicizing (TheoremSearch is a real incumbent).
- **Name:** `math_engine` is provisional/generic — pick + availability-check a real name (like splatreg).
