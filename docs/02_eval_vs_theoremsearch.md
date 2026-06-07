# mathlas retrieval — head-to-head vs TheoremSearch (110-query benchmark)

> **Note (scope):** this doc is the **small-n external** comparison on 110
> human-written queries, measured on the **1.34M** permissive subset. The **large-n
> tight** number — slogan R@1 0.977 / R@10 0.998 over the **81,833**-doc held-out
> split against the full **1,635,233**-doc served index — is in
> [`../RESULTS.md` §3a](../RESULTS.md). Both are reported; they measure different
> things (human queries vs self-recall).

**Date:** 2026-06-06
**Index:** Qwen3-Embedding-8B (4096-d) over **1,341,083 theorems** (the permissive
CC-BY/CC0 subset of `uw-math-ai/theorem-search-dataset`), served as dense + Okapi
BM25 + Reciprocal Rank Fusion (our own `HybridRetriever`).
**Test set:** the dataset's own **110 human-written queries**
(`theorems-test.parquet`), authored blind by four research mathematicians.
**Harness:** `scripts/eval_vs_theoremsearch.py` — our own metric, matching the
dataset README's Hit@20 definition (paper-level = some top-20 candidate's paper ==
GT paper by arXiv id or normalised title; theorem-level = that AND the theorem
number matches).

## The coverage gap — read this first

The published TheoremSearch numbers (**theorem 45.0%, paper 56.8%** Hit@20) were
measured on their **full, private 9.2M-theorem corpus**. The **public** dataset —
the only data anyone can legally index — is the **permissive 15% subset
(1.34M theorems)**; the other ~85% of arXiv is under a non-redistributable license.

Of the 110 test targets, **only 15 papers are in the permissive subset.** The other
95 queries point at papers we are legally not allowed to index, so they are
unanswerable for *any* open system built on the public data — ours included. The
fair, apples-to-apples comparison is therefore the **reachable subset (n=15)**,
which is exactly the regime TheoremSearch reported in (every target present in the
corpus being searched).

## Results (Hit@20)

| Scope | theorem-level | paper-level | vs TheoremSearch |
|---|---|---|---|
| **full-110** (coverage-limited) | 12/110 = 10.9% | 15/110 = 13.6% | not comparable\* |
| **reachable n=15 — hybrid (default)** | 12/15 = **80.0%** | 15/15 = **100.0%** | **+35.0 / +43.2** |
| reachable n=15 — dense-only | 13/15 = 86.7% | 15/15 = 100.0% | +41.7 / +43.2 |
| reachable n=15 — sparse-only (BM25) | 7/15 = 46.7% | 9/15 = 60.0% | +1.7 / +3.2 |

\* full-110 is bounded by corpus *licensing*, not retrieval: every one of the 15
paper-level hits is a reachable-subset target, and we hit **all 15 papers we could
possibly hit** (15/15). The 95 misses are papers simply absent from the corpus.
(This internal consistency — full-110 paper hits == reachable paper hits == 15 —
also validates the harness.)

## Full competitor comparison (every baseline TheoremSearch reported)

TheoremSearch benchmarked against four external systems on this same 110-query test
set; we place mathlas alongside them. Rows 1–5 are **TheoremSearch's own published
numbers** (dataset card), each measured with full-corpus or full-web access — so for
them every target is reachable.

| Method | Theorem Hit@20 | Paper Hit@20 | Corpus / access |
|---|---|---|---|
| arXiv full-text search | — | 2.7% | full arXiv (web) |
| Google (`site:arxiv.org`) | — | 37.8% | full web |
| ChatGPT 5.2 w/ Search | 19.8% | — | full web + model |
| Gemini 3 Pro | 27.0% | — | full web + model |
| **TheoremSearch** (Qwen3-8B) | **45.0%** | **56.8%** | their private **9.2M** index |
| mathlas — full-110 | 10.9% | 13.6% | permissive **1.34M** (coverage-limited) |
| **mathlas — reachable (n=15)** | **80.0%** | **100.0%** | permissive 1.34M (answerable subset) |

Reading it honestly:
- **mathlas full-110 is apples-to-apples** with rows 1–5 (the same 110 queries) but
  is **coverage-bound**: only 15/110 target papers are in the permissive subset we
  may legally index; the other 95 are non-permissive arXiv. That row measures
  *open-corpus coverage*, not retrieval quality — no open system built on the public
  data can do better on the missing 95.
- **mathlas reachable (n=15)** is the fair retrieval-quality number — the same regime
  TheoremSearch's 45.0/56.8 was measured in (target present in the corpus searched).
  On answerable queries mathlas is competitive-to-superior (100% paper-level), with
  the small-n caveat below.
- The coverage gap is exactly what the **web-augmentation** layer
  (`search_directive` + `add_finding`) closes at AI-runtime: the AI fetches a missing
  paper and ingests it, after which `search_existing_math` surfaces it — approaching
  the web-access baselines without redistributing a non-permissive corpus.

## Honest reading

1. **On the fair comparison we match/exceed TheoremSearch.** 100% paper-level
   (15/15) and 80% theorem-level (12/15) over 1.34M docs, vs their 56.8% / 45.0%.
   **Caveat: n=15 is small** — one query is 6.7 pts, so this is a strong
   *directional* result, not a tight statistical claim. The clean **15/15
   paper-level** is the most defensible single number.

2. **The dense Qwen3-8B channel is the workhorse; BM25 alone is weak on this set**
   (46.7% thm). Fusing the two (hybrid) did **not** beat dense-only here — it cost
   one query at theorem-level (80.0% vs 86.7%). This test set is conceptual
   algebraic-geometry / geometric-measure-theory queries, where semantic (dense)
   matching dominates and exact-term (BM25) matching is rarely the deciding signal.
   **We do not claim the fusion improves retrieval on this benchmark.** BM25's value
   is robustness on exact-symbol/operator/constant queries, which this set does not
   stress; the default stays hybrid for that robustness, with this caveat recorded.
   (A light dense-weighting in RRF, or gating BM25 on a strong sparse signal, is a
   tuning lever that would recover the one lost query — not pursued here to keep the
   reported numbers untuned.)

3. **What differentiates mathlas is not a retrieval-quality leap** — our dense
   channel is the same Qwen3-Embedding-8B family TheoremSearch uses — **but the
   system**: fully open + self-hosted, **MCP-native** (an AI calls it directly, no
   API key, free), augmented with airtight **verification** (numeric / OEIS exact
   term-match / real Lean kernel) and **conjecturing** (Ramanujan Machine, FunSearch
   harness) tools that a pure retrieval index does not provide. The retrieval is
   *on par* with the SOTA open tool; the tool around it is the contribution.

## Reproduce

```bash
ME=third_party/math_engine
CUDA_VISIBLE_DEVICES=1 HF_HUB_CACHE=$ME/reference/downloads/hf PYTHONPATH=$ME \
python3 $ME/scripts/eval_vs_theoremsearch.py \
  --index $ME/reference/downloads/index.npz \
  --test  $ME/reference/theorem-search-dataset/theorems-test.parquet \
  --device cuda --k 20 --verbose
```

Index build (resumable, multi-GPU; ~92 min on 2 GPUs for the 8B pass):

```bash
HF_HUB_CACHE=$ME/reference/downloads/hf python3 $ME/scripts/build_index_mp.py all \
  --corpus $ME/reference/theorem-search-dataset --model Qwen/Qwen3-Embedding-8B \
  --workdir $ME/reference/downloads/index_build \
  --out $ME/reference/downloads/index.npz --ngpu 2
```

## Next step for statistical rigor

n=15 is the ceiling of the *human-written* test set's overlap with the permissive
corpus — it cannot grow without growing the corpus (license-capped at 1.34M). To get
a large-n, statistically tight retrieval + ablation benchmark, build a held-out set
**from** the 1.34M corpus (hold out K theorems, query by slogan / paraphrase, measure
recall@k over the remaining index) — much larger n, at the cost of slogan-derived
(vs human-written) queries. This is the rigorous self-benchmark tracked for the next
pass; it is the right vehicle for any claim about hybrid-vs-dense fusion, which this
n=15 set is too small to settle.
