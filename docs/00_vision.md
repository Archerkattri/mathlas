# math_engine — an open math-application engine (vision)

> **Provisional name** (check availability before committing, like splatreg). **Date:** 2026-06-05. **Status:** research sweep launched.

## The idea
Given a problem, find the **existing** math — a formula/equation/theorem humans *already* derived — that **applies** to it. **Not** creating new math: **mapping** a problem to the perfect existing tool. The math already exists, but it's hard for humans to locate and connect the right piece, so countless solvable things stay unsolved. AI can do that mapping **if it has the right tool**. This is that tool — **open, for everyone**, not a closed lab agent.

## The core gap (and the proof it's real)
The bottleneck isn't the formulas — it's the **trigger/map**: knowing *which* existing result applies to a *new* problem. The May-2026 **unit-distance disproof** is the proof: an OpenAI model didn't invent new math — it **mapped century-old class-field-tower theory onto a problem nobody had pointed it at** (because everyone *assumed* the grid was optimal). The value was the **connection**. (Bubeck: *"executed like an amazing mathematician… didn't invent fundamentally new tools."*) The Aletheia/Gemini study (arXiv 2601.22401) operationalized exactly this — finding most "open" Erdős problems were open from **obscurity, not difficulty**.

## Not an LLM — a "math-for-math" mapping model
LLMs map **words→words**. We need a model that maps **math→math**: given a problem's mathematical *structure*, recognize which existing result/transform applies. That's **premise selection + transform-recognition over mathematical structure** (theorem graphs, formula embeddings, neuro-symbolic) — not token-prediction over prose. The hard, unsolved core, and the white space.

## Architecture: retrieve → map → route → verify
- **CORPUS** — the open math islands: Lean **mathlib** (~210k theorems, a real dependency graph), **OEIS** (~400k, CC BY-SA), **arXiv-math**, **Wikidata** formulas (CC0), DLMF references. No full formalization needed — the mapper reads prose + structured where available.
- **MAPPER** (the math-for-math model) — problem → candidate existing results + the transform/analogy that connects them. The brain.
- **TOOLBOX** the engine *routes to* per domain (these are components, not rivals): **Lean** (formal proof), **PSLQ/LLL + Inverse Symbolic Calculator** (number → known formula), **Ramanujan Machine** (constant relations), **FunSearch/AlphaEvolve**-style search (combinatorial), **SymPy** (symbolic).
- **VERIFICATION** — non-negotiable. Every retrieved application is checked (Lean / numeric / symbolic). This is what separates a real hit from a confident hallucination — every credible 2026 result paired the model with a checker.

## Open positioning
OpenAI (Aletheia) and DeepMind build this **inside closed agents**. No **open, standalone, composable** "give me the existing math that solves this + verify it" tool exists. `math_engine` fills that — Apache-2.0, the tool *anyone* (human or AI) plugs in.

## Next
Research sweep launched (5 streams: mapping methods · the labs' approaches/papers · GitHub prior art · community ideas/demand · math-representation models). Then: synthesize the crumbs → the mapper architecture + a **narrow-domain MVP** + a **benchmark** (problems whose solution *is* an existing result → measure retrieval+verification accuracy). Build narrow, prove it, widen.
