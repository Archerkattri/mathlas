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

## Status / what is still running

* **Statement-channel embed is still running on GPU1** (resumable;
  `logs/stmt_channel_embed2.log`; ~52-62 docs/s ≈ 17-19 h total for 3.68M).
  Relaunch after any interruption with:
  ```
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 HF_HUB_CACHE=reference/downloads/hf \
    PYTHONPATH=. python3 scripts/build_statement_channel.py embed \
      --workdir reference/downloads/statement_channel
  ```
  then `finalize --out reference/downloads/index_full_statement.npz`.
* Once `index_full_statement.npz` exists,
  `python3 scripts/eval_retrieval_upgrades.py final` automatically runs the
  FULL-corpus dual-channel eval on the n=3000 baseline sample and prints the
  headline table (everything else is already cached).

## Headline (current best, honest accounting)

* vs the 0.614 / 0.832 dense baseline (n=3000, full index): production hybrid
  (dense+BM25, new rrf_k=10) = **0.771 / 0.999** (carries the proxy's
  exact-text advantage on the BM25 side).
* Honest cross-representation lift with no leak: shipped rerank blend over
  dense = **+1.7pp R@1 / +3.9pp R@10** (n=1000).
* Dual-channel max-sim on the 200k prefix: slogan-only 0.833 → **0.981 R@1**;
  full-corpus number pending the embed.

## Server wiring (for the owner of `mathlas/server.py` — not applied here)

`_get_retriever()` currently does
`HybridRetriever.from_index(index_path, embedder=_embedder_for_index(index_path))`.
To pick up the upgrades:

```python
# inside _get_retriever(), replacing the from_index call:
stmt_path = os.environ.get("MATHLAS_STATEMENT_INDEX")  # opt-in 2nd dense channel
if stmt_path is None:
    cand = os.path.join(os.path.dirname(index_path), "index_full_statement.npz")
    stmt_path = cand if os.path.exists(cand) else None
reranker = None
if os.environ.get("MATHLAS_RERANK", "").strip().lower() in {"1", "true", "yes", "on"}:
    from .retrieve.rerank import Qwen3Reranker
    reranker = Qwen3Reranker()        # lazy; honest stderr fallback if no torch
retr = HybridRetriever.from_index(
    index_path,
    embedder=_embedder_for_index(index_path),
    statement_index=stmt_path,        # None -> exactly today's behaviour
    reranker=reranker,                # None -> no rerank stage
)
```

Notes for the server owner: `rrf_k=10` is now the constructor default (no
wiring needed); the statement matrix adds ~30GB fp16 → ~60GB fp32 resident
(same as the slogan matrix), so gate it on available RAM; reranking adds a
0.6B-model load + ~1-2 s/query on GPU (much slower on CPU) — keep it env-gated.
