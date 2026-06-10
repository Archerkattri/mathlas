# Retrieval head-to-head vs TheoremSearch (110-query benchmark)

This is the small-n **external** comparison: mathlas retrieval measured on the
dataset's own 110 human-written queries, against every baseline TheoremSearch
reported. The large-n tight numbers — body→slogan R@1 0.614 / R@10 0.832 at the
current 3.68M index, and slogan→slogan R@1 0.977 / R@10 0.998 over the 81,833-doc
held-out split of the earlier 1.635M build — are in
[`../RESULTS.md` §3a0/§3a](../RESULTS.md). The two regimes measure different
things (human queries vs self-recall); both are reported.

**Index:** Qwen3-Embedding-8B (4096-d) over the served **3,683,428-document** index
(the permissive CC-BY/CC0 subset of `uw-math-ai/theorem-search-dataset`, 1,341,083
docs, + 2,342,345 slogan-embedded Dolma arXiv-math docs) — dense + Okapi BM25 +
Reciprocal Rank Fusion. **Numbers below re-measured 2026-06-10 on this 3.68M index**
(the earlier 1.34M-index run is kept in the history table for honesty — the larger
index *cost* 2 reachable papers at top-20).
**Test set:** the dataset's own **110 human-written queries**
(`theorems-test.parquet`), authored blind by four research mathematicians.
**Metric:** Hit@20, matching the dataset README's definition (paper-level = some
top-20 candidate's paper equals the GT paper by arXiv id or normalised title;
theorem-level = that and the theorem number matches). Harness:
`scripts/eval_vs_theoremsearch.py`.

## The coverage gap

The published TheoremSearch numbers (**theorem 45.0%, paper 56.8%** Hit@20) were
measured on their **full, private 9.2M-theorem corpus**. The **public** dataset — the
only data anyone can legally index — is the **permissive 15% subset (1.34M
theorems)**; the other ~85% of arXiv is under a non-redistributable license.

Of the 110 test targets, **only 15 papers are in the permissive subset.** The other
95 queries point at papers no open system may legally index, so they are unanswerable
for *any* open system built on the public data — mathlas included. The fair,
apples-to-apples retrieval comparison is therefore the **reachable subset (n=15)**,
which is exactly the regime TheoremSearch reported in (every target present in the
corpus being searched). The full-110 number is bounded by corpus *licensing*, not
retrieval quality.

## Results (Hit@20) — 3.68M index, re-measured 2026-06-10

| Scope | theorem-level | paper-level | vs TheoremSearch |
|---|---|---|---|
| **full-110** (coverage-limited) | 11/110 = 10.0% | 13/110 = 11.8% | not comparable\* |
| **reachable n=15 — hybrid (default)** | 11/15 = **73.3%** | 13/15 = **86.7%** | **+28.3 / +29.9** |
| reachable n=15 — dense-only | 13/15 = 86.7% | 13/15 = 86.7% | +41.7 / +29.9 |
| reachable n=15 — sparse-only (BM25) | 7/15 = 46.7% | 9/15 = 60.0% | +1.7 / +3.2 |

\* full-110 is bounded by corpus *licensing*, not retrieval: every paper-level hit is
a reachable-subset target (full-110 paper hits == reachable paper hits == 13, which
also validates the harness). The 95 misses are papers simply absent from the corpus.

**Index-growth effect (honesty):** on the earlier **1.34M** index (2026-06-06 run)
the reachable row was 12/15 = 80.0% theorem / **15/15 = 100.0%** paper (dense-only
86.7 / 100.0; sparse unchanged). Growing the index to 3.68M added 2.34M Dolma
distractors that **crowd 2 of the 15 reachable papers out of the top-20**
(paper-level 100.0% → 86.7%; full-110 paper 13.6% → 11.8%). Theorem-level
dense-only is unchanged at 86.7%. The trade is more coverage for slightly more
crowding on this small reachable subset; both runs are reported.

## Full comparison (every baseline TheoremSearch reported)

TheoremSearch benchmarked against four external systems on this same 110-query test
set; mathlas is placed alongside them. Rows 1–5 are **TheoremSearch's own published
numbers** (dataset card), each measured with full-corpus or full-web access — so for
them every target is reachable.

| Method | Theorem Hit@20 | Paper Hit@20 | Corpus / access |
|---|---|---|---|
| arXiv full-text search | — | 2.7% | full arXiv (web) |
| Google (`site:arxiv.org`) | — | 37.8% | full web |
| ChatGPT 5.2 w/ Search | 19.8% | — | full web + model |
| Gemini 3 Pro | 27.0% | — | full web + model |
| **TheoremSearch** (Qwen3-8B) | **45.0%** | **56.8%** | private **9.2M** index |
| mathlas — full-110 | 10.0% | 11.8% | open **3.68M** (coverage-limited) |
| **mathlas — reachable (n=15)** | **73.3%** | **86.7%** | open 3.68M (answerable subset) |

## How to read it

- **full-110 is apples-to-apples** with rows 1–5 (the same 110 queries) but is
  **coverage-bound**: only 15/110 target papers are in the permissive subset that may
  legally be indexed; the other 95 are non-permissive arXiv. That row measures
  *open-corpus coverage*, not retrieval quality — no open system built on the public
  data can do better on the missing 95. The self-augmenting web loop closes exactly
  this gap at AI-runtime (to 59.1% / 70.0%, re-confirmed 2026-06-10 on this same
  3.68M index; see [`../RESULTS.md` §3c](../RESULTS.md)).
- **reachable (n=15)** is the fair retrieval-quality number — the same regime
  TheoremSearch's 45.0 / 56.8 was measured in. On answerable queries mathlas is
  competitive-to-superior (86.7% paper-level), with the small-n caveat below.

## Honest reading

1. **On the fair comparison, mathlas exceeds TheoremSearch** — 86.7% paper-level
   (13/15) and 73.3% theorem-level (11/15) over the open 3.68M index, vs 56.8% /
   45.0%. **n=15 is small** — one query is 6.7 pts — so this is a strong *directional*
   result, not a tight statistical claim. (On the earlier 1.34M index this row was
   15/15 = 100% paper-level; see the index-growth note above.)

2. **The dense Qwen3-8B channel is the workhorse; BM25 alone is weak on this set**
   (46.7% thm). Hybrid fusion did *not* beat dense-only here — it cost two queries at
   theorem-level (73.3% vs 86.7%). This test set is conceptual algebraic-geometry /
   geometric-measure-theory queries, where semantic (dense) matching dominates and
   exact-term (BM25) matching is rarely decisive. mathlas makes **no claim** that the
   fusion improves retrieval on this benchmark; BM25's value is robustness on
   exact-symbol / operator / constant queries, which this set does not stress, and the
   default stays hybrid for that robustness.

3. **What differentiates mathlas is not a retrieval-quality leap** — the dense channel
   is the same Qwen3-Embedding-8B family TheoremSearch uses — **but the system around
   it**: fully open and self-hosted, MCP-native (an AI calls it directly, no API key,
   free), and augmented with airtight **verification** (numeric / OEIS exact term-match
   / real Lean kernel) and **conjecturing** (Ramanujan Machine, FunSearch harness)
   tools that a pure retrieval index does not provide. The retrieval is *on par* with
   the SOTA open tool; the tool around it is the contribution.

## Reproduce

```bash
ME=third_party/math_engine
CUDA_VISIBLE_DEVICES=1 HF_HUB_CACHE=$ME/reference/downloads/hf PYTHONPATH=$ME \
python3 $ME/scripts/eval_vs_theoremsearch.py \
  --index $ME/reference/downloads/index_full_dense.npz \
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

For a large-n, statistically tight retrieval benchmark, the self-recall numbers are
the complementary measurement: body→slogan R@1 0.614 / R@10 0.832 at the current
3.68M index, and (at the earlier 1.635M build, n=81,833 held-out) slogan→slogan
R@1 0.977 / R@10 0.998; see [`../RESULTS.md` §3a0/§3a](../RESULTS.md).
