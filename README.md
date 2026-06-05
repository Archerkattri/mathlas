# mathlas

> An **open** math-application engine: map a problem to the **existing** math that
> solves it, then **verify** that it applies — with an honest provenance label
> (never "novel"). Apache-2.0, mostly-pure-Python, provider-agnostic.

Most solvable problems stay unsolved not because the formula is missing, but
because nobody connected the *right existing result* to the problem. mathlas
is that connector — and, unlike a retrieval-only lookup, it **checks that the
retrieved math actually applies**. It builds its **own** index (it depends on no
third-party running system) and the LLM brain + embedder are pluggable.

## Architecture

```
route ─▶ retrieve ─▶ MAP ─▶ VERIFY ─▶ PROVENANCE
        (own index) (needs↔   (tiered)  (retrieved/
                     guarantees)          unidentified — never "novel")
```

- **Numeric domain** (airtight, no LLM/network): a real value/constant → a known
  closed form, verified by independent high-precision re-evaluation.
- **Problem domain**: a described problem → existing theorems that apply, via our
  own hybrid (dense + BM25 + RRF) retrieval, two-stage needs↔guarantees mapping,
  and **structured-adversarial applicability verification** (the differentiator).

## Install

```bash
pip install -e .                 # core (numeric domain): mpmath, sympy, numpy, scipy
pip install -e '.[retrieve]'     # + pyarrow, to read the open theorem dataset
pip install -e '.[embed]'        # + sentence-transformers/torch, for the Qwen3 embedder (GPU)
pip install -e '.[llm]'          # + an example LLM backend (any provider works)
```

## Use — CLI

```bash
# Numeric: paste a constant, get the verified closed form (no LLM, no network)
mathlas 1.6449340668482264364724151666460251892
#   1.64493406684823 -> pi**2/6  [known_form, verified 48 digits]

mathlas 1.2020569031595942853997381615114499908              # -> zeta(3)
mathlas --basis pi,e,catalan 0.9159655941772190150546035149  # -> Catalan
# (pass enough digits: PSLQ needs more than a 16-digit float to lock on safely —
#  too few digits returns UNIDENTIFIED rather than guessing.)

# Problem: describe a result, retrieve + map + verify over a corpus
mathlas "filtered colimit commutes with an adequate functor" \
    --corpus reference/theorem-search-dataset --limit 5000 --k 5      # retrieval-only
mathlas "<problem>" --corpus DIR --llm anthropic                   # + map & verify
```

## Use — Python

```python
import mpmath
from mathlas import identify
print(identify(mpmath.zeta(2)))        # 1.64493406684823 -> pi**2/6 [known_form, verified 48 digits]

from mathlas import solve, EchoLLM
from mathlas.retrieve.corpus import load_documents
from mathlas.retrieve.hybrid import HybridRetriever
docs = load_documents("reference/theorem-search-dataset", limit=5000)
retr = HybridRetriever(docs)                       # our own dense+BM25+RRF index
sol  = solve("show f is continuous given open preimages", retr, my_llm)
print(sol.best.provenance.novelty.value)           # retrieved_applies | retrieved_rejected | ...
```

## What's verified (light, CPU-only)

- Numeric: **recovery 8/8, false-positive 0/3** (`benchmarks/numeric_bench.py`).
- Retrieval: **paper-level Hit@20 = 15/15** on the test subset, hashing-embedder
  floor (`scripts/eval_retrieval.py`); production uses Qwen3 + the full index
  (`scripts/build_index.py`, offline GPU).
- The adversarial informal verifier rejects a mis-retrieved candidate the mapping
  step let through — the gap retrieval-only tools leave open.

## Docs

`docs/00_vision.md` · `docs/01_landscape.md` (research sweep) ·
`docs/02_mvp_spec.md` · `docs/03_theoremsearch_analysis.md` (reference-only study) ·
**`docs/04_build.md`** (what was built + methods + citations).

## Positioning

The closest system, **TheoremSearch** (UW Math AI Lab), is recall-optimized
retrieval only — it finds the *statement you already formulated* and never checks
applicability, routes across tools, accepts numeric inputs, or labels provenance.
mathlas adds exactly those, built independently. TheoremSearch is
**reference-only**; we reuse just their openly-licensed (CC-BY/CC0) **dataset** as
raw data to build our **own** index — not their API, MCP, index, or code.
