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

## Source-aware retrieval (opt-in) — the full matrix

Measured **2026-06-10 on the served 3,683,428-doc index, CPU-only** (the index is
served with the binary sidecar so the 30 GB matrix stays on disk; the 110 dense
ranks come from one exact streamed fp32 pass over the matrix — the same
renormalised-fp32 math `from_index` serves). Harness:
`scripts/eval_source_weights.py` (stages `codes` → `selfrecall` → `dense110` →
`eval110`; query vectors cached once by `scripts/embed_webaug110_queries.py`).
Logs: `logs/eval_sw_codes.log`, `logs/eval_sw_dense110.log`. Raw numbers:
`reference/downloads/retrieval_upgrades/source_weights_results.json`.

`search_existing_math` takes optional `source_filter` / `source_weights` over the
canonical source keys (arxiv / dolma / stacks / proofwiki / other — the served
index is 1.30M arxiv + 2.34M dolma + 12.7k stacks + 23.9k proofwiki + 2.4k other).
**Both default off; the default-off row below reproduced the measured 10.0 / 11.8%
baseline exactly** (the eval asserts it, and `tests/test_source_aware.py` pins the
default ranking byte-identical — order *and* scores).

**What down-weighting/excluding dolma buys on the 110 human queries (Hit@20):**

| dolma knob | full-110 theorem | full-110 paper | reachable-15 theorem | reachable-15 paper |
|---|---|---|---|---|
| off (default) | 11/110 = 10.0% | 13/110 = 11.8% | 11/15 = 73.3% | 13/15 = 86.7% |
| weight 0.5 | 13/110 = 11.8% | 14/110 = 12.7% | 13/15 = 86.7% | 14/15 = 93.3% |
| weight 0.25 | 13/110 = 11.8% | 14/110 = 12.7% | 13/15 = 86.7% | 14/15 = 93.3% |
| weight 0 | 13/110 = 11.8% | 14/110 = 12.7% | 13/15 = 86.7% | 14/15 = 93.3% |
| **exclude** | **13/110 = 11.8%** | **15/110 = 13.6%** | **13/15 = 86.7%** | **15/15 = 100.0%** |

`exclude` **fully recovers the pre-growth (1.34M-index) paper-level 13.6% and the
15/15 = 100% reachable row**, with theorem-level *above* the old index (11.8% vs
10.9%): any soft down-weight already restores 2 of the crowded-out targets; the
hard in-channel exclude (dolma never takes a channel slot) restores the last
paper-level miss. Same small-sample caveat as the rest of this page — on the
reachable subset 1 query = 6.7 pts.

**What it costs — why it ships opt-in, default off.** On the n=3000 body→slogan
self-recall, ~65% of targets (1942/3000) ARE Dolma docs; down-weighting dolma is
catastrophic exactly for those queries:

| dolma knob | all R@10 | dolma-target R@10 (n=1942) | non-dolma-target R@10 (n=1058) |
|---|---|---|---|
| off (default) | 0.999 | 0.999 | 1.000 |
| weight 0.5 | 0.925 | 0.884 | 1.000 |
| weight 0.25 | 0.709 | 0.551 | 1.000 |
| weight 0 / exclude | 0.353 | 0.000 | 1.000 |

So this is a **per-query-intent knob, not a global default**: an AI excludes/demotes
dolma when it wants canonical theorem statements (the TheoremSearch-110 regime) and
leaves the knobs off when the web-mined corpus is part of the answer space.
Semantics + score math (`score(d) = w_src(d) · Σ_c 1/(rrf_k + rank_c(d))`; filters
applied in-channel so depth is preserved): `mathlas/retrieve/hybrid.py`.

## Dual-channel dense (the v1.2 statement channel) on the same 110 queries

Measured 2026-06-10, same harness, with the dense channel replaced by the
shipped dual-channel max-sim ranking (slogan matrix + the new statement matrix;
`scripts/eval_source_weights.py dense110 --dual` then `eval110 --dual`; log
`logs/eval110_dual.log`). The open question was whether the statement channel
is "the structural fix" that recovers the index-growth regression without the
source knob. Answer, honestly: **partial**.

| config (Hit@20) | full-110 thm | full-110 paper | reachable-15 thm | reachable-15 paper |
|---|---|---|---|---|
| single channel, default | 10.0% | 11.8% | 73.3% | 86.7% |
| **dual channel, default** | **10.9%** | **12.7%** | **80.0%** | **93.3%** |
| single channel + exclude dolma | 11.8% | **13.6%** | 86.7% | **100.0%** |
| dual channel + exclude dolma (or any weight) | 11.8% | 12.7% | 86.7% | 93.3% |

The dual channel structurally recovers one of the two crowded-out papers and
one theorem at default settings (no knob), but it does **not** recover the full
pre-growth paper-level 13.6% (15/110) on its own, and combining it with the
dolma knobs is not additive here (the dual ranking tops out at 14/15 reachable
paper-level where single + exclude reaches 15/15). On this benchmark the
source-aware exclude on the single channel remains the full mitigation; the
dual channel's decisive win is the full-corpus statement-shaped self-recall
(R@1 0.614 -> 0.965; `docs/RETRIEVAL_UPGRADE_NOTES.md`). Same n=15 caveat:
every delta in this table is 1-2 queries.

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
