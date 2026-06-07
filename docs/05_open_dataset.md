# The open dataset & index (shipped 1.635M, roadmap to ~11M)

## What ships now — the 1,635,233-doc exact index

The **served** index (`reference/downloads/index_full_dense.npz`, default in
`server.py`) is a single **exact** (PQ-free) dense matrix of **1,635,233** documents,
Qwen3-Embedding-8B (4096-d, fp16) + Okapi-BM25 + RRF:

| Source | Docs | License | Embedded text |
|---|---|---|---|
| TheoremSearch permissive subset | 1,341,083 | CC-BY / CC0 | NL slogan |
| Dolma arXiv-math (this corpus, sloganized) | 294,150 | permissive arXiv | NL slogan |
| Stacks Project | 12,693 | CC-BY-SA | NL slogan |
| ProofWiki | 23,871 | CC-BY-SA | NL slogan |

Held-out 81,833-doc test split (`reference/downloads/splits/`): slogan **R@1 0.977 /
R@10 0.998**, statement **R@10 0.923** (`scripts/eval_benchmark.py all`). The dolma
slice was given NL slogans (the cross-representation fix below) before the merge, so
the whole 1.635M index lives in one slogan-dense space.

## Roadmap — extending to ~11M theorems (the full dolma corpus)

To widen coverage further we extracted our own corpus from arXiv full-text and will
index the remaining ~9.5M **synced** with the 1.635M into one ~11.3M index.

## The corpus
- **Source**: `emozilla/dolma-v1_7-arxiv` (~1.55M papers), regex theorem-environment extraction
  (`scripts/build_arxiv_fulltext_corpus.py`). **9,970,177** raw theorems.
- **Cleaned + deduped → 9,773,744** indexable docs (`scripts/postprocess_corpus.py` logic:
  strip `\label`, `\cite`/`\ref`→`[REF]`, unwrap `\textcolor`, drop figure-leak bodies, per-paper
  dedup that also kills the env-name-collision extraction dups).
- **Provenance**: a content-hash `doc_id`, **not** an arXiv id. Recovering arXiv ids needed either
  a ~5 TB bulk-tar download or a multi-day per-paper fetch — dropped as not worth it. We care about
  the theorems, not the paper number; the hash still identifies the source paper.

## Indexing, synced with the 1.34M — `scripts/index_dolma_corpus.py`
Same embedder (Qwen3-Embedding-8B, 4096-d, fp16), same meta schema, so vectors share the 1.34M's
space and combine row-for-row. Each step resumable; only `embed` needs a GPU.
1. **shard** (CPU, **DONE**): dolma → `docs_*.jsonl` in the `build_index_mp` schema
   (`doc_id` namespaced `dolma:<hash>`, `embed_text` = cleaned statement). 2,444 shards.
2. **embed** (GPU): reuse `build_index_mp.py embed --workdir dolma_index_build` (multi-GPU,
   resumable). Deferred until the GPUs free (benbi owns them).
3. **merge** (CPU): base 1.34M + dolma vectors → a **faiss IVF+PQ** union index (~11.3M). The flat
   exact dot-product retriever does not fit RAM past ~1.34M (92 GB fp16 / 185 GB fp32), so the
   big-corpus serving is ANN+PQ. A small `HybridRetriever.from_faiss` wires it into serving.

## The cross-representation fix (already applied to the shipped dolma slice)
Raw dolma docs embed their **statement** (no NL slogan), vs the base slogans — so they
retrieve slightly worse on NL queries (the cross-representation gap). The fix is to
**generate NL slogans** (an offline local-LLM pass; `scripts/gen_slogans_local.py` +
`apply_slogans.py`) and re-embed. The 294,150-doc dolma slice in the shipped 1.635M
index was sloganized this way; the remaining ~9.5M of the full dolma corpus is the
pending follow-up for the ~11.3M index.

## Multi-source (already fetched, future fold-in)
`datasets/{mathlib (Apache), stacks (CC-BY-SA), proofwiki, arxiv_math}` — additional permissive
sources to add once the dolma half is indexed.
