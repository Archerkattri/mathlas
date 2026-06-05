# TheoremSearch — deep analysis (paper + repo), and what it means for math_engine

> Sources: the arXiv paper (2602.05216, read in full) + a read-only crawl of the cloned repo
> (`reference/TheoremSearch/`). Date 2026-06-05. Purpose: learn from the closest existing
> system so we fast-forward past what they solved and focus on what they left open. **Learn,
> don't vendor — the repo has no LICENSE (all-rights-reserved).**

## TL;DR
TheoremSearch is a **recall-optimized semantic *lookup*** over 9.2M theorem statements: parse LaTeX →
have an LLM write an English "slogan" → embed the slogan → vector search. It is excellent at "find the
lemma hiding in an obscure paper" (Hit@20 ~45%), and it ships a clean **public REST + MCP backend** and an
**open dataset**. It does **not** map a problem to the right statement (the human does that), **verify** that a
retrieved theorem applies, **route** across solver tools, accept **numeric/sequence** inputs, or label
**provenance**. Those five are our wedge — confirmed absent in the serving code. Bonus: they **scraped OEIS/
DLMF/FindStat and then dropped them**, and their advertised graph/NL↔FL angle is **unshipped** — so both our
numeric beachhead and a graph-driven verified-retrieval frontier are open territory.

## 1. What they built (the production pipeline)
`parse → sloganize → embed → serve`. Two generations live side by side:
- **Legacy flat path** (one denormalized table) — **this is what the public `api.theoremsearch.com/search` + `/mcp` actually serve.**
- **Normalized v2** (`statement`/`paper`/`slogan`/`embedding`/`*_dependency`) — served at `/graph/*`, richer, newer.

1. **Parse** — the real LaTeX extraction lives in an **external lib `arXiTeX`** (pip-from-git, *not in the repo*). plasTeX (6.9M) → TeX-logging (1.8M) → regex (0.5M) fallbacks. 20 statement kinds.
2. **Sloganize** — an LLM writes a 1–4 sentence ASCII-English "denotation" of each theorem, *stripping symbols/proofs*. Production model is now **Qwen3-235B** (the paper/README saying "DeepSeek V3" is **stale**). An **escalation chain** (minimal→standard→comprehensive→final) re-runs only on statements self-flagged `INSUFFICIENT CONTEXT`, so big-model tokens are spent only on hard cases. Their internal ranking: **Claude Opus 4.5 > Gemini 3 > DeepSeek V3.1** for slogan quality (Opus 4.5 Hit@10 0.776).
3. **Embed** — Qwen3-Embedding-8B (4096-dim, normalized). They tried BERT-768 → Qwen-1024 → Qwen3-8B → Gemma-768 and standardized on Qwen3-8B.
4. **Serve** — `binary_quantize → bit(4096) Hamming HNSW ANN → full-precision cosine rerank → optional citation boost (cosine + λ·ln(citations))`. **No cross-encoder in production** (the paper's Qwen3-Reranker was an experiment, not deployed). ~3s/query. Key scale trick: all filters live *inside* the ANN CTE so pgvector keeps scanning until k filtered rows are found.

## 2. The core lesson: embed the *meaning*, not the *notation*
The whole system exists because **embedders choke on symbol-heavy LaTeX** and mathematicians query in prose. So they translate every theorem into an NL slogan and embed that. Two corollaries from their notes:
- **Don't embed global context** (definitions/notation) — it drowns out the theorem's own content.
- **Context helps slogan *generation*** (Body+Introduction > Body alone) but the *embedded unit* stays the lean slogan.
This is the single most reusable idea, and **we (Claude) are their best-measured slogan generator** — so any representation we build on this can be better than theirs.

## 3. The graph — advertised, barely used (the "math as structure" reality)
They build two dependency graphs, and this is the most important nuance for our structure angle:
- **Informal graph** — edges between arXiv statements from 4 signals (deterministic `\ref`/`\label` = trust 1.0, heuristic 0.7, llm 0.5, **judge 0.0/dropped**). Used as (a) context in slogan prompts, (b) an OOM-prone, **disabled-by-default** `/graph/pagerank` route.
- **Formal graph** — a precision **Lean 4 / Mathlib dependency graph**. v1 = homemade InfoTree probe (from LeanDojo); v2 = external `lean-graph` extractor with **typed edges** (extends/field/sig/proof/def/docref, rich metadata), 1.6M→16M edges, 30 projects (Mathlib + FLT, PFR, carleson, sphere-packing…). Used **only as training supervision** (a linear query head: Recall@100 0.16→**0.54**) — **never traversed at inference**.

**Punchline:** the headline product (`/search`) **ignores the graph entirely** — it's pure slogan kNN + citation boost. Their *stated* killer app, **organic NL↔FL matching** (align arXiv prose ↔ Lean declarations), is **unshipped** (milestone M1 not started; they call it a literature gap). So **genuinely graph-driven, structure-aware, verified retrieval is unclaimed** — they advertise the graph but don't serve it.

## 4. Results reality
- Hit@10 0.432 theorem / 0.505 paper, Hit@20 0.450/0.568 — **2× the best LLM-with-search**, but a **recall** metric: ~55% of expert queries miss in the top 20.
- Evaluated on **111 blind expert queries** (mathematicians wrote NL descriptions of theorems they knew, *without* seeing the corpus). Grading is **fuzzy title-match + ≥2 shared 4-grams** (`matlas-comparison/06_grade.py`) — robust to LaTeX variation, **no correctness/applicability check ever**. It is a *find-the-statement* benchmark, never a *does-it-solve-the-problem* benchmark.

## 5. The two appendix findings that ARE our thesis
- **Appendix B (real users):** every win came *after the user did the hard mapping themselves* — reduced "smooth variety over a separably closed field has a k-point" → "k-points of 𝔸ⁿ are dense" (via étale local structure), or knew their result was "analogous to finite abelian groups." **The tool looks up the statement you already formulated; the human does the needs↔guarantees reduction/analogy.** That mapping is our engine's job.
- **Appendix C (RAG demo):** Claude alone answered a research question *confidently wrong*; Claude + their DB-as-RAG got it right. But "right" was judged by **Claude's reasoning + a human expert** — **no automatic check that the retrieved theorems apply**. Their flagship demo *is* the gap.

## 6. What's reusable — the open DATASET and the METHOD, **not their running system**
> **Hard line (2026-06-05, user):** we use ONLY their openly-licensed **dataset** (as data) and the documented **method/lessons** (as knowledge). Their **live API / MCP / index / code are reference-only — NOT a runtime dependency.** `math_engine` builds its own retrieval. (We do NOT call their endpoints; an API client written by mistake was deleted.)

- **Open dataset** `uw-math-ai/theorem-search-dataset` (CC-BY/CC0) — papers, statements, **slogans, AND both dependency graphs (informal + formal edges)**. Usable as **raw input data** to build *our own* index.
- **Open-weight models** (Qwen3-8B embed, Qwen3-235B slogan) — the *method* is reproducible with our own implementation/models (and Claude is their best-measured slogan generator).
- **Scale recipe** (knowledge, not their code): binary-quant→Hamming→cosine; filters-inside-ANN-CTE; tar+byte-offsets for 9M sources in S3; SLURM hash-sharding; slogan escalation chain.
- *(Their public REST `/search`, MCP `/mcp`, and `/graph/*` endpoints exist — noted for reference only. We do NOT depend on them.)*

## 7. Critical design lessons (their honest failure-mode docs)
- **"RAG hurts strong models on familiar libraries"** — feeding retrieved premises to Sonnet on Mathlib dropped it **83%→75%**: retrieval can *displace* correct parametric recall. ⇒ our engine must not naively dump retrieved math into an LLM; retrieval should be *gated by verification*, not prepended blindly.
- **"typecheck ≠ correctness"** — 22/24 typecheck but only **5/24 correct**. ⇒ a formal check that something *compiles* is not a check that it *solves the problem*; applicability verification must check the right thing.
- **Graph-supervised retrieval saturates ~0.78 recall** (embedding-bound) — pure embeddings have a ceiling; structure/verification is how you go past it.

## 8. The wedge — confirmed absent in the serving code (grepped)
| Capability | In TheoremSearch? | Evidence |
|---|---|---|
| Applicability **verification** | **No** | "judge/verify" only verify graph *edges*, never query↔theorem fit. Retrieval ends at cosine + citation boost. |
| **Routing** across tools (PSLQ/OEIS/Lean/SymPy) | **No** | query always embedded → one pgvector index; no solver dispatch. |
| **Numeric / constant / sequence** inputs | **No** | `query: str` only. **They scraped OEIS/DLMF/FindStat then dropped them — explicitly walked away.** |
| **Provenance** / retrieved-vs-novel | **No** | results are ranked hits; no "exists in literature vs novel," no confidence-of-existence. |
| **Cross-field** reach | **No** | pure math corpus; no "same result under another name in field Z." |
| Graph-**driven** retrieval | **No** (advertised, unshipped) | demo ignores the graph; formal graph is training-only; NL↔FL matching unstarted. |

## 9. Strategic implication for math_engine
1. **Numeric beachhead is doubly-validated** — TheoremSearch *proved the numeric corpora exist* (they scraped OEIS/DLMF/FindStat) and then *abandoned* the structured-numeric-lookup. Airtight auto-verification + open territory = still the right first proof of the `router → verify → provenance` spine.
2. **Retrieval is OURS** — we build our own index over their openly-licensed (CC-BY/CC0) *dataset* + open corpora (arXiv/mathlib/OEIS), applying the slogan lesson with our own implementation. We do **NOT** depend on their API/running system (reference-only). Effort concentrates on mapping + verification.
3. **A third frontier opened:** genuinely **graph-driven, verified** retrieval (the thing they advertise but didn't ship) — using their *open* formal+informal graph edges as the structure substrate for "needs↔guarantees" matching. Higher research value, harder.
4. **Verification design constraints (from their failures):** don't prepend retrieved math blindly (it hurts strong models); don't equate typecheck with correctness; embeddings saturate (~0.78) so verification/structure is the way past the ceiling.
