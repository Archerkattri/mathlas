# Quantized laptop tier — 2026-06-10 (limitations item 2)

The flagship dense channel was unusable without a big box: a 3.68M x 4096
fp16 matrix (**30.2 GB on disk, ~60 GB as fp32 in RAM**) plus a
Qwen3-Embedding-**8B** GPU query encoder. This ships the document-side fix —
the pattern TheoremSearch proved at 9.2M scale (binary embeddings + exact
rescore): quantize the SAME index once, then serve it from disk-memmapped
artifacts on a CPU-only machine.

Everything below is reproducible with `scripts/eval_quantized_tier.py`
(stages cache under `reference/downloads/retrieval_upgrades/`, results merge
into `results.json` there); the serving mechanism is pinned by
`tests/test_quantized_tier.py` (15 tests: quantize round-trip, packed-bit
Hamming == brute-force popcount, two-stage == brute force, end-to-end
`from_index(quantized=...)`, honest errors on missing artifacts).

## What ships

`mathlas/retrieve/quantized.py` + an opt-in `HybridRetriever.from_index`
backend (default behaviour unchanged):

```python
r = HybridRetriever.from_index(index, embedder=emb, quantized="int8")    # ~15 GB memmap
r = HybridRetriever.from_index(index, embedder=emb, quantized="binary")  # ~1.9 GB + rescore rows
```

The MCP server flips it with one env var (`mathlas/server.py`):

```bash
MATHLAS_QUANTIZED=binary python -m mathlas.server   # or int8
```

* **int8** — per-dimension symmetric quantization (`q = round(x/scale)`,
  `scale[d] = absmax_d/127`, fp32 scale vector saved). Search = exact
  chunked dequantized dot over all rows, served from a memmap: resident
  memory is whatever the OS caches, not a 60 GB resident matrix.
* **binary** — sign bits packed 8/byte (512 bytes/doc). Search = two-stage:
  packed-bit Hamming (XOR + uint16 popcount table) selects a top-1000
  shortlist, then ONLY those 1000 rows are gathered and rescored with exact
  dot products (int8-dequant rows by default; falls back to fp16 rows from
  the original npz if the int8 sidecar is absent).

Artifacts are built ONCE, streamed in 100k-row slabs straight out of the npz
(`npz_member_memmap` — `np.savez` members are ZIP_STORED, so the 30 GB matrix
is never materialized):

```
OMP_NUM_THREADS=4 PYTHONPATH=. python3 scripts/eval_quantized_tier.py quantize
```

## Footprints (measured)

| artifact | file | size |
|---|---|---|
| fp16 flagship (unchanged) | `index_full_dense.npz` | 30.17 GB (≈60 GB fp32 resident) |
| int8 + scale | `index_full_dense.q8.npy` + `.q8.scale.npy` | 15.09 GB + 16 KB (memmap-served) |
| binary | `index_full_dense.qbin.npy` | 1.89 GB (memmap-served) |

## Recall (measured, full 3.68M index, n=3000)

Same protocol and the SAME cached n=3000 Qwen3-Embedding-8B query embeddings
as the published dense baseline (`retrieval_upgrades/queries_n3000_seed0.npz`;
body→slogan cross-representation self-recall — the honest hard regime; same
caveats as `docs/RETRIEVAL_UPGRADE_NOTES.md`). "fp16 agreement" compares each
ranking with the exact fp16 ranking (`dense_top100.npy`) — the real contract:
quantization should not change what gets retrieved.

| dense channel config | R@1 | R@5 | R@10 | MRR | top-1 = fp16 | top-10 overlap |
|---|---|---|---|---|---|---|
| fp16 exact (baseline) | 0.6140 | 0.7997 | 0.8323 | 0.6984 | 1.000 | 1.000 |
| int8 (exact dequant dot) | 0.6147 | 0.8003 | 0.8323 | 0.6986 | 0.9967 | 0.9944 |
| binary raw (Hamming only) | 0.6110 | 0.7950 | 0.8387 | 0.6957 | 0.8433 | 0.7564 |
| **binary top-1000 → int8 rescore** | **0.6143** | **0.8003** | **0.8323** | **0.6985** | **0.9963** | **0.9945** |
| binary top-1000 → fp16-row rescore | 0.6143 | 0.7997 | 0.8323 | 0.6986 | 0.9990 | 1.0000 |

**Headline: quantization is recall-lossless on this index.** int8 R@1 is
within +0.07% of fp16 (0.6147 vs 0.6140 — a 2-query difference at n=3000);
binary+rescore R@1 within −0.05% (0.6143) and R@10 *equal* (0.8323). Even raw
1-bit Hamming alone only costs −0.3% R@1. Per-query results barely move
either: int8 returns the identical top-1 to fp16 on 99.67% of queries with
99.4% top-10 overlap; binary+fp16-rescore reaches 99.9% / 100%. Raw Hamming's
lower agreement (84.3% / 75.6%) is what the rescore stage is for.

Eval note: the binary channel's ranking is scored with an exact algebraic
identity (Hamming ranking == descending ±1 sign-dot ranking, since
`dot = dim − 2·hamming`), which turns the 3000-query eval into the same
chunked GEMM as int8; the production packed-bit kernel itself is pinned equal
to brute-force popcount in the tests.

## Latency (measured, CPU only, OMP_NUM_THREADS=4)

Production `QuantizedDenseIndex.search` paths on the real artifacts
(single query, top-10; cold = first query after open, page cache empty-ish):

| path | cold (first query in process) | warm median (5 reps) |
|---|---|---|
| **binary Hamming top-1000 → int8 rescore** | 2.63 s | **2.40 s** |
| int8 full dequant dot | 56.98 s | 30.69 s |

The binary+rescore path is the laptop default: 2.4 s/query end-to-end over
3.68M docs on 4 CPU threads — the same class as TheoremSearch's reported ~3 s
at 9.2M with the identical architecture (binary ANN + full-precision rescore).
The int8 full scan is the high-fidelity offline option (it touches all 15 GB
per query); it is NOT the interactive path. Caveat: "cold" here still
benefits from the OS page cache of the build box; a true cold-disk first
query additionally pays the artifact read (~1.9 GB for binary on any SSD).

## The honest query-encoder caveat (8B tier)

Quantization shrinks the **document** side only. Queries must still be
embedded in the SAME space the index was built in — Qwen3-Embedding-**8B**.
A small encoder (Qwen3-Embedding-0.6B, ~1.2 GB, CPU-runnable) lives in a
**different embedding space**: its query vectors against this 8B-built index
would be garbage. The 8B quantized tier is therefore: quantized index on the
laptop + 8B query vectors from elsewhere (a GPU box you own, or a remote
embedding endpoint). The TRUE end-to-end laptop tier is the 0.6B sibling
index below (v1.3.0), which closes exactly this gap.

The BM25 channel is unaffected (CPU-native already); the opt-in statement
dense channel is not supported under `quantized=` (it stays fp32-resident —
use the default path if you need it).

---

# The 0.6B end-to-end laptop tier, 2026-06-10 (v1.3.0)

The remaining half of limitations item 2, now closed and measured. The SAME
3,683,428-doc corpus and the SAME served representation channel ("name --
slogan", statement fallback) re-embedded once with **Qwen3-Embedding-0.6B**
(dim 1024, fp16), so the query encoder itself runs on a laptop CPU. Built by
`scripts/build_06b_index.py` (resumable shards, single GPU, sourced from the
served meta so row i aligns EXACTLY with row i of the 8B index; alignment
spot-verified by re-embedding rows across the matrix, min cosine 0.998).
Measured end-to-end by `scripts/eval_06b_tier.py`
(queries/ranks/quantize/eval/latency; results merge into
`retrieval_upgrades/results.json`, keys `q06b_*`).

Serve it with one env var (`MATHLAS_ENCODER=0.6b`, composes with
`MATHLAS_QUANTIZED=int8|binary`; selection mechanism pinned by
`tests/test_encoder_tier.py`):

```bash
MATHLAS_ENCODER=0.6b MATHLAS_QUANTIZED=binary python -m mathlas.server
```

## Footprints (measured, dim 1024)

| artifact | file | size |
|---|---|---|
| 0.6B fp16 index | `index_full_dense_06b.npz` | 7.54 GB |
| int8 + scale | `index_full_dense_06b.q8.npy` (+ `.q8.scale.npy`) | 3.77 GB |
| binary | `index_full_dense_06b.qbin.npy` | 0.47 GB |
| query encoder | Qwen3-Embedding-0.6B (HF cache) | ~1.2 GB |

The dense-channel laptop install is **binary sidecar (0.47 GB) + the 0.6B
encoder (~1.2 GB) = ~1.7 GB**, plus the rescore source you choose (int8
3.77 GB recommended; the rescore touches only ~1000 rows per query, so it is
read traffic, not resident memory). The hybrid channel additionally uses the
shared meta jsonl (3.3 GB) + BM25 cache (1.6 GB), identical for both tiers.

## Recall (measured, full 3.68M index, n=3000)

Protocol identical to the 8B table above: the SAME n=3000 sampled rows,
query = the doc's LaTeX statement (body -> slogan cross-representation
self-recall, the honest hard regime), but query texts re-encoded with the
0.6B model (the cached 8B vectors are a different space and unusable).
"fp16 agreement" compares against the 0.6B fp16 exact ranking.

| dense channel config | R@1 | R@5 | R@10 | MRR | top-1 = fp16 | top-10 overlap |
|---|---|---|---|---|---|---|
| 0.6B fp16 exact (tier baseline) | 0.5437 | 0.7070 | 0.7450 | 0.6190 | 1.000 | 1.000 |
| 0.6B int8 (exact dequant dot) | 0.5443 | 0.7077 | 0.7450 | 0.6194 | 0.9730 | 0.9682 |
| 0.6B binary raw (Hamming only) | 0.4990 | 0.6610 | 0.6983 | 0.5735 | 0.6790 | 0.5520 |
| **0.6B binary top-1000 -> int8 rescore** | **0.5450** | **0.7077** | **0.7450** | **0.6197** | 0.9723 | 0.9654 |
| 0.6B binary top-1000 -> fp16 rescore | 0.5433 | 0.7070 | 0.7457 | 0.6188 | 0.9730 | 0.9670 |

Two findings, stated plainly:

1. **Quantization is again recall-lossless** within the tier: binary+rescore
   matches the 0.6B fp16 exact scan to the third decimal (R@1 0.5450 vs
   0.5437, R@10 equal at 0.7450). Raw 1-bit Hamming alone costs more here
   than at dim 4096 (R@1 -4.5pp, top-1 agreement 67.9%): 1024 sign bits carry
   less signal, so the rescore stage is load-bearing in this tier, not
   cosmetic.
2. **The encoder downgrade costs ~7-9pp recall vs the 8B tier** (R@1 0.544 vs
   0.614, R@10 0.745 vs 0.832 on the identical protocol). That is the honest
   price of fitting the query encoder on a laptop CPU. The dual-channel 8B
   configuration (0.965 / 0.999) remains the quality ceiling and remains
   big-box-only.

| tier (same n=3000 protocol) | R@1 | R@10 | query encoder runs on |
|---|---|---|---|
| 0.6B laptop tier (binary + int8 rescore) | 0.545 | 0.745 | laptop CPU |
| 8B quantized tier (binary + int8 rescore) | 0.614 | 0.832 | GPU box / remote |
| 8B dual-channel (opt-in, fp32-resident) | 0.965 | 0.999 | GPU box, 150 GB RAM |

**TheoremSearch-110 corpus-only probe** (`benchmarks/webaug_110_bench.py
baseline` on the 0.6B index): Hit@20 theorem 8.2% (9/110), paper 10.0%
(11/110), vs the 8B tier's 10.0% / 11.8%. Both are licensing-bounded floors
(only 15/110 targets are in the permissive corpus); the 0.6B tier loses 2
queries of the 8B floor, consistent with the self-recall gap.

## Latency: the laptop headline (measured, pure CPU)

End-to-end = 0.6B query encode on CPU + production
`QuantizedDenseIndex.search` (binary Hamming top-1000 + int8 rescore),
single query, top-10, warm medians over 5 reps
(`scripts/eval_06b_tier.py latency`; CUDA masked off in-process):

| threads | query encode | binary search | **end-to-end** | int8 full scan |
|---|---|---|---|---|
| 4 | 0.26 s | 0.40 s | **0.67 s/query** | 6.26 s/query |
| 2 | 0.49 s | 0.39 s | **0.88 s/query** | 6.43 s/query |

Sub-second per query over 3.68M documents with everything, encoder included,
on 2-4 CPU threads; one-time model load 10-17 s. Honesty notes: (1) threads
were capped via `torch.set_num_threads` / `OMP_NUM_THREADS` but the cores are
server-class (Threadripper PRO 7975WX); a laptop core is slower, though the
same order of magnitude. (2) Warm numbers benefit from the OS page cache; a
true cold-disk first query additionally pays the artifact read (~0.47 GB
binary + touched rescore rows on any SSD). The 8B quantized tier's 2.4
s/query (above) does not include query encoding at all, because it cannot:
its encoder does not fit on the machine.
