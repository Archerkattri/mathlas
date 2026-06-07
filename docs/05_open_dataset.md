# The open dataset & index

`search_existing_math` is served from a single open, self-hosted index. This document
describes what that index contains, how it is built, and how it is evaluated.

## The served index

The served index (`reference/downloads/index_full_dense.npz`, the default in
`server.py`) is an **exact** (PQ-free) dense matrix of **1,635,233** documents,
embedded with Qwen3-Embedding-8B (4096-d, fp16) and fused with Okapi-BM25 by
Reciprocal Rank Fusion:

| Source | Docs | License | Embedded text |
|---|---|---|---|
| TheoremSearch permissive subset | 1,341,083 | CC-BY / CC0 | NL slogan |
| arXiv-math (Dolma) | 294,150 | permissive arXiv | NL slogan |
| Stacks Project | 12,693 | CC-BY-SA | NL slogan |
| ProofWiki | 23,871 | CC-BY-SA | NL slogan |

Every source is openly licensed and redistributable. Each document is embedded by its
natural-language **slogan** — the meaning of the theorem, not its LaTeX — so the whole
index lives in one consistent slogan-dense space and queries match on concept rather
than notation.

## Retrieval accuracy

On a held-out **81,833-document** test split, querying each theorem by its
natural-language slogan retrieves its own entry at **R@1 0.977 / R@10 0.998**; querying
by the raw formal statement (the harder cross-representation test, formal LaTeX in,
NL-slogan entry out) retrieves it at **R@10 0.923**. Reproduce with
`scripts/eval_benchmark.py all`.

## How the corpus is built

The arXiv-math slice is extracted from the open `emozilla/dolma-v1_7-arxiv` corpus by
regex theorem-environment extraction (`scripts/build_arxiv_fulltext_corpus.py`),
cleaned and deduplicated (`scripts/postprocess_corpus.py`: strip `\label`,
normalise `\cite`/`\ref` to `[REF]`, unwrap `\textcolor`, drop figure-leak bodies, and
per-paper dedup that removes environment-name-collision duplicates). Each document
carries a content-hash `doc_id` that identifies its source paper.

Because raw arXiv documents would otherwise embed their formal **statement** rather
than an NL slogan — and so retrieve worse on natural-language queries — the arXiv-math
slice is given NL slogans before indexing (`scripts/gen_slogans_local.py` +
`apply_slogans.py`, an offline local-LLM pass) and re-embedded, keeping it in the same
slogan-dense space as the rest of the corpus.

## Building the index

The index is built offline by `scripts/build_index_mp.py` (multi-GPU, resumable):
shard the corpus into the index schema, embed with Qwen3-Embedding-8B, and write the
served `index.npz` plus its sidecar metadata. `scripts/reindex_findings.py` embeds the
backlog of web-added findings (see `add_finding`). Only the embed step needs a GPU; all
other steps run on CPU and are resumable.
