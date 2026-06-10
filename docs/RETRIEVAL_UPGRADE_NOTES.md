# Retrieval upgrades — 2026-06-09 (roadmap items 1 + 2 + RRF-k tune)

Targets the body→slogan cross-representation gap on the served 3.68M index
(`reference/downloads/index_full_dense.npz`). Three upgrades, each measured on
the SAME fixed query sample; everything below is reproducible with
`scripts/eval_retrieval_upgrades.py` (stages cache their artifacts under
`reference/downloads/retrieval_upgrades/`; numbers accumulate in
`results.json` there).

## Baseline (reconfirmed before any change)

`scripts/eval_dense_recall.py --index reference/downloads/index_full_dense.npz
--n 3000 --seed 0` on GPU1, 2026-06-09:

```
recall@1 = 61.4%   recall@5 = 80.0%   recall@10 = 83.2%   recall@20 = 86.6%
```

Exactly the published 0.614 / 0.832. The new harness's `dense only` row on the
same sample reproduces it digit-for-digit (cross-validation of the harness).
The 3000 query embeddings are cached
(`retrieval_upgrades/queries_n3000_seed0.npz`, via the new
`--save-queries` flag) so every later eval uses the identical query set.

**Eval caveat (applies to everything below):** this is a self-retrieval proxy
— query = a doc's LaTeX statement, target = that exact doc row. BM25 indexes
the statement text, and the new statement dense channel embeds it, so those
channels see (near-)exact query text in the target doc: their absolute numbers
are optimistic vs human prose queries. The dense slogan-channel number is the
honest cross-representation figure; deltas between configurations on the same
sample remain informative.

## 1. RRF k tune  →  default changed 60 → 10

`eval_retrieval_upgrades.py ranks` + `rrf` (dense top-100 + BM25 top-100 per
query, fused at each k; n=3000, full index):

| config            | R@1   | R@5   | R@10  | MRR   |
|-------------------|-------|-------|-------|-------|
| dense only        | 0.614 | 0.800 | 0.832 | 0.698 |
| bm25 only (leaky) | 0.966 | 0.998 | 0.999 | 0.981 |
| hybrid rrf_k=60   | 0.764 | 0.963 | 0.991 | 0.855 |
| hybrid rrf_k=20   | 0.767 | 0.983 | 0.998 | 0.862 |
| **hybrid rrf_k=10** | **0.771** | **0.991** | **0.999** | **0.869** |

k=10 wins every metric → `HybridRetriever` default `rrf_k` is now **10**
(changed in `mathlas/retrieve/hybrid.py` with a comment citing this eval).

## 2. Qwen3-Reranker-0.6B cross-encoder (`mathlas/retrieve/rerank.py`)

Model: `Qwen/Qwen3-Reranker-0.6B`, downloaded to `reference/downloads/hf`.
Card-exact prompt frame; task instruction *"Given a math problem or statement,
judge whether the theorem answers it"*; score = P(yes) over {yes,no} last-token
logits in float32; `logits_to_keep=1` so the 151k-vocab logits are computed for
the final position only (full-seq logits OOM a 32GB GPU at batch 32×2048).

Measured (n=1000 subset of the sample — note this subset is easier than the
full 3000: its dense-only R@1 is 0.731, not 0.614; compare within the block):

| config                                   | R@1   | R@5   | R@10  | MRR   |
|------------------------------------------|-------|-------|-------|-------|
| dense only                               | 0.731 | 0.933 | 0.950 | 0.822 |
| dense + rerank, **replacement**          | 0.650 | 0.931 | 0.976 | 0.771 |
| **RRF(dense, rerank) — shipped blend**   | **0.748** | **0.974** | **0.989** | **0.844** |
| hybrid k=10, no rerank (leaky)           | 0.849 | 0.998 | 1.000 | 0.919 |
| hybrid k=10 + rerank, replacement        | 0.471 | 0.868 | 0.953 | 0.639 |
| RRF(hybrid, rerank) blend                | 0.702 | 0.984 | 0.998 | 0.821 |

Findings (honest):
* **Replacement reranking HURTS R@1 on this proxy.** Failure analysis
  (529 top-1 misses → only 50 are near-duplicate statements; median text
  similarity of the miss top-1 is 0.25) shows the cross-encoder ranks closely
  RELATED theorems above the exact target row — relevance ≠ row identity, and
  this corpus holds many same-content theorem variants. An fp16 P(yes)
  saturation artifact (mass ties at 1.0 falling back to first-stage order) was
  initially masking part of this (R@1 0.566 vs the de-tied 0.471).
* **The shipped wiring is therefore an RRF blend, not a replacement**
  (`HybridRetriever._maybe_rerank`): rerank order fused with the first-stage
  order at `rrf_k`. In the honest cross-representation setting this is a
  genuine lift over dense-only: **R@1 +1.7pp, R@5 +4.1pp, R@10 +3.9pp**.
* On the leaky hybrid numbers even the blend trails no-rerank (0.702 vs
  0.849) — that ordering is dominated by the eval's exact-text BM25 advantage,
  which human queries do not have. Reranking stays **opt-in** (inject
  `reranker=Qwen3Reranker(...)`; `retrieve(..., rerank=True/False)`; honest
  stderr fallback + unreranked order when torch/weights are missing).

## 3. Dual-channel doc indexing (statement channel)

`scripts/build_statement_channel.py` — resumable single-GPU (GPU1) sharded
embed of every doc's `clean_statement`'d statement text with
Qwen3-Embedding-8B (row-aligned with the served index; 3470 of 3,683,428 rows
fall back to name+slogan where the statement is empty/figure-polluted).
Workdir `reference/downloads/statement_channel/` (921 shards × 4000;
`emb/emb_*.npy` fp16). The served files are untouched; output is a NEW
artifact `reference/downloads/index_full_statement.npz`
(+ `.partial.npz` prefix builds).

Mechanism validated on the first 200k embedded rows (`slice-queries` +
`dual` stages; n=1000 queries sampled from rows <200k, seed 0):

| config (200k prefix)        | R@1   | R@5   | R@10  | MRR   |
|-----------------------------|-------|-------|-------|-------|
| slogan channel only         | 0.833 | 0.975 | 0.989 | 0.895 |
| statement channel only      | 0.980 | 0.998 | 0.999 | 0.989 |
| **max-sim(slogan, statement)** | **0.981** | **0.998** | **0.999** | **0.989** |
| statement as 3rd RRF channel | 0.872 | 0.995 | 0.998 | 0.931 |

Max-sim decisively beats third-RRF-channel fusion → the shipped wiring folds
the statement channel into the DENSE ranking by per-doc max-sim
(`HybridRetriever._dense_rank`), opt-in via
`HybridRetriever.from_index(..., statement_index=...)` or
`HybridRetriever(..., statement_matrix=...)`; default behaviour unchanged.
(Statement-channel numbers carry the same exact-text caveat as BM25 here;
the design point is that a statement-shaped query gets a channel in its own
surface form while prose queries keep the slogan channel.)

## Status: COMPLETE (2026-06-10, shipped in v1.2.0)

* The statement-channel embed finished (921 shards, 3,683,428 rows,
  `logs/stmt_channel_embed2.log`); `finalize` wrote
  `reference/downloads/index_full_statement.npz` (3,683,428 x 4096 fp16,
  30.2 GB), row count verified equal to the served meta and every shard
  verified contiguous and row-aligned before finalize.
* `scripts/eval_retrieval_upgrades.py final` ran the FULL-corpus dual-channel
  eval on the n=3000 baseline sample; numbers in the headline section below.
* The TheoremSearch-110 regression-recovery check ran via
  `scripts/eval_source_weights.py dense110 --dual` + `eval110 --dual`
  (the dual ranks reproduce the shipped max-sim math exactly): dual-channel
  default recovers PART of the index-growth regression (paper 11.8% -> 12.7%,
  theorem 10.0% -> 10.9%) but NOT the full pre-growth 13.6%; the source-aware
  exclude knob remains the full mitigation on that benchmark. Full table:
  `docs/02_eval_vs_theoremsearch.md`.

## Headline (current best, honest accounting)

Full-corpus dual-channel numbers measured 2026-06-10 on the SAME cached n=3000
baseline sample as the 0.614 / 0.832 headline
(`scripts/eval_retrieval_upgrades.py final`, log `logs/eval_upgrades_final.log`):

| config (n=3000, full 3.68M index) | R@1 | R@10 | MRR |
|---|---|---|---|
| dense slogan channel (baseline) | 0.614 | 0.832 | 0.698 |
| production hybrid (dense+BM25, rrf_k=10) | 0.771 | 0.999 | 0.869 |
| **FULL dual-channel max-sim (dense only)** | **0.965** | **0.999** | **0.982** |
| FULL dual-dense + BM25, rrf_k=10 | 0.966 | 1.000 | 0.982 |

* **Dual channel: R@1 0.614 -> 0.965 on the full corpus** (the 200k-prefix
  mechanism check predicted 0.833 -> 0.981 on its easier subset; the full-scale
  measurement confirms the mechanism at 3.68M). Caveat, stated plainly: this is
  the self-retrieval proxy, and the statement channel indexes the very text the
  queries are drawn from, so like BM25 it carries an exact-text advantage here.
  The design claim it validates is narrower and real: a statement-shaped query
  now has a dense channel in its own surface form instead of relying on the
  slogan representation. The no-leak external check is the TheoremSearch-110
  human-query bench above (a real but partial lift).
* Honest cross-representation lift with no leak: shipped rerank blend over
  dense = **+1.7pp R@1 / +3.9pp R@10** (n=1000).

## Server wiring (APPLIED in v1.2.0, `mathlas/server.py`)

Both upgrades are now served, strictly opt-in via env vars (default behaviour
byte-identical to v1.1.2):

```bash
# second dense channel (statement matrix folded in by per-doc max-sim);
# value = a path, or "auto" for index_full_statement.npz beside the index
MATHLAS_STATEMENT_INDEX=/path/index_full_statement.npz python -m mathlas.server

# cross-encoder rerank blend stage (Qwen3-Reranker-0.6B)
MATHLAS_RERANK=1 python -m mathlas.server
```

Wiring tests: `tests/test_statement_channel.py`. `rrf_k=10` is the
constructor default (no wiring needed).

Why the statement channel is opt-in and NOT auto-detected (the honest memory
math, one machine, fp32-resident serving):

| dense tier | disk | resident RAM | self-recall R@1 (n=3000) |
|---|---|---|---|
| binary sidecar (`MATHLAS_QUANTIZED=binary`) | 1.9 GB (+15.1 rescore) | ~minimal (memmap) | 0.614 |
| int8 sidecar (`MATHLAS_QUANTIZED=int8`) | 15.1 GB | ~minimal (memmap) | 0.615 |
| fp16 single channel (default) | 30.2 GB | ~57.5 GB fp32 | 0.614 |
| fp16 dual channel (`MATHLAS_STATEMENT_INDEX`) | 60.4 GB | ~115 GB fp32 | 0.965 (proxy-leaky, see headline) |

A second 3,683,428 x 4096 matrix roughly DOUBLES resident RAM; merely having
the file on disk must not change the server's footprint, so auto-detect was
rejected. Measured at full scale on the build box (2026-06-10, logs/
dual_serving_smoke2.log): the dual-channel retriever loads in 264 s with a
150.1 GB process peak (the two fp32 matrices ~115 GB + 3.68M doc records +
BM25 cache), answers the dual max-sim dense scan at ~2.75 s/query median on
2 CPU threads, and hit 20/20 top-10 on real cached query vectors through the
genuine retrieve() path. So plan for a ~192 GB-RAM workstation for the dual
tier. Shipping this exposed a real loader bug, fixed in v1.2.0: from_index
used to materialise np.load + fp32-cast + normalisation temps per matrix
(transient >250 GB for the dual load; OOM-killed on the 251 GB build box).
Matrices are now streamed memmap -> chunked unit-norm fp32
(`_load_unit_fp32_matrix`), so peak load memory is essentially the resident
footprint; pinned by `tests/test_statement_channel.py`. The dual tier is
incompatible with `MATHLAS_QUANTIZED` (the statement channel is served fp32;
from_index raises an honest error). Reranking adds a 0.6B-model load +
~1-2 s/query on GPU (much slower on CPU), hence env-gated too. Query encoding
still needs the Qwen3-Embedding-8B encoder in all tiers.
