#!/usr/bin/env python3
"""Assemble and upload the mathlas document corpus to Hugging Face.

What ships (see docs/HF_DATASET_LICENSING.md for the audited per-source
licensing matrix):

  * config ``theoremsearch`` (1,341,083 rows): the TheoremSearch permissive
    subset as served (arXiv + ProofWiki + Stacks + open textbooks), CC BY-SA 4.0.
  * config ``dolma`` (2,342,345 rows): our theorem-statement excerpts from the
    open Dolma v1.7 arXiv corpus (ODC-BY) with OUR generated NL slogans.
  * config ``findings`` (small): the live ``web_added`` findings store, with
    embedded ``dense_vec`` vectors STRIPPED (embeddings are out of scope).

What never ships: the 30 GB embedding matrices, the local benchmark slices
(``reference/downloads/datasets/``), and anything listed as excluded in the
licensing doc. This script reads ONLY the served corpus artifacts:

  * ``reference/downloads/index_full.meta.jsonl``  (the served 3.68M corpus)
  * ``reference/downloads/findings.jsonl``         (the live findings store)

Usage:

  # build + validate locally, print stats, push nothing
  python3 scripts/hf_upload_corpus.py --dry-run

  # the one command once a token exists (builds if needed, then uploads
  # to a PRIVATE dataset repo; flip public on the Hub when ready)
  HF_TOKEN=hf_... python3 scripts/hf_upload_corpus.py --push

The built dataset lands in ``reference/downloads/hf_corpus/`` (gitignored via
``reference/``); the parquet never goes anywhere near git.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mathlas.retrieve.corpus import _coerce_citations, source_key  # noqa: E402

# ---------------------------------------------------------------------------
# Layout + provenance constants
# ---------------------------------------------------------------------------

DEFAULT_META = os.path.join(_REPO_ROOT, "reference", "downloads",
                            "index_full.meta.jsonl")
DEFAULT_FINDINGS = os.path.join(_REPO_ROOT, "reference", "downloads",
                                "findings.jsonl")
DEFAULT_OUT = os.path.join(_REPO_ROOT, "reference", "downloads", "hf_corpus")
DEFAULT_REPO = "Archerkattri/mathlas-corpus"

#: source_key -> upload config. The TheoremSearch subset groups four source
#: keys; dolma stands alone. ``findings`` is routed separately (different file
#: AND different schema).
CONFIG_OF_SOURCE = {
    "arxiv": "theoremsearch",
    "proofwiki": "theoremsearch",
    "stacks": "theoremsearch",
    "other": "theoremsearch",
    "dolma": "dolma",
}
MAIN_CONFIGS = ("theoremsearch", "dolma")
ALL_CONFIGS = MAIN_CONFIGS + ("findings",)

#: Audited counts (docs/HF_DATASET_LICENSING.md). The build refuses to ship a
#: corpus whose per-config counts drift from the audited served index unless
#: ``--no-strict`` is passed (e.g. for a future rebuild; re-audit first).
EXPECTED_COUNTS = {"theoremsearch": 1_341_083, "dolma": 2_342_345}

#: Fields a findings record may carry that must NEVER ship (embeddings and
#: serving flags are out of scope for the document corpus).
FINDINGS_STRIP = ("dense", "dense_vec")

ZENODO_DOI = "10.5281/zenodo.20634787"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _main_schema():
    import pyarrow as pa
    return pa.schema([
        ("doc_id", pa.string()),
        ("name", pa.string()),
        ("slogan", pa.string()),
        ("statement", pa.string()),
        ("source", pa.string()),
        ("title", pa.string()),
        ("label", pa.string()),
        ("citations", pa.int64()),
        ("category", pa.string()),
        ("source_key", pa.string()),
    ])


def _findings_schema():
    import pyarrow as pa
    return pa.schema([
        ("doc_id", pa.string()),
        ("name", pa.string()),
        ("slogan", pa.string()),
        ("statement", pa.string()),
        ("source", pa.string()),
        ("provenance", pa.string()),
        ("added_ts", pa.float64()),
    ])


def _opt_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("none", "nan") else None


def normalize_row(m: dict) -> dict:
    """One served-meta record -> one upload row (main schema)."""
    return {
        "doc_id": str(m.get("doc_id")),
        "name": _opt_str(m.get("name")),
        "slogan": _opt_str(m.get("slogan")) or "",
        "statement": _opt_str(m.get("statement")) or "",
        "source": _opt_str(m.get("source")),
        "title": _opt_str(m.get("title")),
        "label": _opt_str(m.get("label")),
        "citations": _coerce_citations(m.get("citations")),
        "category": _opt_str(m.get("category")),
        "source_key": source_key(m.get("doc_id"), m.get("source")),
    }


def normalize_finding(m: dict) -> dict:
    """One findings record -> one upload row. Embedded vectors are stripped by
    construction: only the whitelisted fields below ever leave this function."""
    for k in FINDINGS_STRIP:
        if k in m:  # explicit, so a future schema change cannot leak vectors
            m = {kk: vv for kk, vv in m.items() if kk not in FINDINGS_STRIP}
            break
    ts = m.get("added_ts")
    return {
        "doc_id": str(m.get("doc_id")),
        "name": _opt_str(m.get("name")),
        "slogan": _opt_str(m.get("slogan")) or "",
        "statement": _opt_str(m.get("statement")) or "",
        "source": _opt_str(m.get("source")),
        "provenance": _opt_str(m.get("provenance")) or "web_added",
        "added_ts": float(ts) if ts is not None else None,
    }


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


@dataclass
class _ShardWriter:
    """Buffered multi-shard parquet writer for one config."""
    out_dir: str
    config: str
    schema: object
    rows_per_shard: int
    batch_rows: int = 8192
    rows: int = 0
    shards: List[str] = field(default_factory=list)
    _buf: List[dict] = field(default_factory=list)
    _writer: object = None
    _in_shard: int = 0

    def _shard_path(self) -> str:
        d = os.path.join(self.out_dir, "data", self.config)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.config}-{len(self.shards):05d}.parquet")

    def _flush_buf(self) -> None:
        if not self._buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        if self._writer is None:
            path = self._shard_path()
            self.shards.append(path)
            self._writer = pq.ParquetWriter(path, self.schema,
                                            compression="zstd")
            self._in_shard = 0
        cols = {f.name: [r.get(f.name) for r in self._buf]
                for f in self.schema}
        self._writer.write_batch(
            pa.record_batch([pa.array(cols[f.name], type=f.type)
                             for f in self.schema], schema=self.schema))
        self._in_shard += len(self._buf)
        self._buf.clear()
        if self._in_shard >= self.rows_per_shard:
            self._writer.close()
            self._writer = None

    def add(self, row: dict) -> None:
        self._buf.append(row)
        self.rows += 1
        if len(self._buf) >= self.batch_rows:
            self._flush_buf()

    def close(self) -> None:
        self._flush_buf()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def bytes(self) -> int:
        return sum(os.path.getsize(p) for p in self.shards
                   if os.path.exists(p))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(out_dir: str = DEFAULT_OUT,
          meta_path: str = DEFAULT_META,
          findings_path: str = DEFAULT_FINDINGS,
          rows_per_shard: int = 400_000,
          exclude_sources: Sequence[str] = (),
          strict: bool = True,
          progress: bool = True,
          batch_rows: int = 8192) -> dict:
    """Assemble the upload tree: ``data/<config>/*.parquet`` + README.md +
    stats.json. Returns the stats dict.

    ``exclude_sources`` removes whole source keys (the licensing-exclusion
    switch: if a source's terms ever turn out non-redistributable, exclude it
    here and the card auto-reflects the shipped counts).
    """
    exclude = {s.strip().lower() for s in exclude_sources if s.strip()}
    unknown = exclude - set(CONFIG_OF_SOURCE) - {"findings"}
    if unknown:
        raise ValueError(f"unknown --exclude-source keys: {sorted(unknown)} "
                         f"(known: {sorted(CONFIG_OF_SOURCE)} + 'findings')")
    os.makedirs(out_dir, exist_ok=True)

    writers = {c: _ShardWriter(out_dir, c, _main_schema(), rows_per_shard,
                               batch_rows=batch_rows)
               for c in MAIN_CONFIGS}
    per_source: Dict[str, int] = {}
    excluded_rows = 0
    t0 = time.time()
    for i, m in enumerate(_iter_jsonl(meta_path)):
        row = normalize_row(m)
        sk = row["source_key"]
        if sk in exclude:
            excluded_rows += 1
            continue
        per_source[sk] = per_source.get(sk, 0) + 1
        writers[CONFIG_OF_SOURCE[sk]].add(row)
        if progress and (i + 1) % 500_000 == 0:
            print(f"  ... {i + 1:,} meta rows in {time.time() - t0:.0f}s",
                  flush=True)
    for w in writers.values():
        w.close()

    fw = None
    if "findings" not in exclude and os.path.exists(findings_path):
        fw = _ShardWriter(out_dir, "findings", _findings_schema(),
                          rows_per_shard, batch_rows=batch_rows)
        for m in _iter_jsonl(findings_path):
            fw.add(normalize_finding(m))
        fw.close()

    stats = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta_path": meta_path,
        "findings_path": findings_path,
        "excluded_sources": sorted(exclude),
        "excluded_rows": excluded_rows,
        "per_source": dict(sorted(per_source.items())),
        "configs": {},
    }
    for c, w in list(writers.items()) + ([("findings", fw)] if fw else []):
        if w.rows:  # fully excluded / empty configs do not ship
            stats["configs"][c] = {"rows": w.rows, "shards": len(w.shards),
                                   "bytes": w.bytes}

    if strict and not exclude:
        for c, want in EXPECTED_COUNTS.items():
            got = stats["configs"].get(c, {}).get("rows", 0)
            if got != want:
                raise RuntimeError(
                    f"config '{c}' built {got:,} rows but the audited served "
                    f"corpus has {want:,} (docs/HF_DATASET_LICENSING.md). "
                    f"Re-audit the licensing doc + EXPECTED_COUNTS for the new "
                    f"build, or pass --no-strict.")

    card = build_card(stats)
    _assert_no_em_dash(card, "dataset card")
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------


def _assert_no_em_dash(text: str, what: str) -> None:
    for ch, name in (("—", "em dash"), ("–", "en dash")):
        if ch in text:
            line = next(i for i, l in enumerate(text.splitlines(), 1)
                        if ch in l)
            raise ValueError(f"{what} contains an {name} (line {line}); "
                             f"public-facing text must not use them")


def build_card(stats: dict) -> str:
    cfg = stats["configs"]
    ps = stats["per_source"]
    n_ts = cfg.get("theoremsearch", {}).get("rows", 0)
    n_dolma = cfg.get("dolma", {}).get("rows", 0)
    n_find = cfg.get("findings", {}).get("rows", 0)
    total = n_ts + n_dolma
    gb = sum(c.get("bytes", 0) for c in cfg.values()) / 1e9

    def row(label, key):
        return f"| {label} | {ps.get(key, 0):,} |"

    excl = ""
    if stats.get("excluded_sources"):
        excl = ("\n> Note: the following source groups were EXCLUDED from this "
                "build for licensing reasons: "
                + ", ".join(stats["excluded_sources"]) + ".\n")

    configs_yaml = []
    for c in ALL_CONFIGS:
        if c not in cfg:
            continue
        configs_yaml.append(
            f"  - config_name: {c}\n"
            + ("    default: true\n" if c == "theoremsearch" else "")
            + f"    data_files:\n"
            f"      - split: train\n"
            f"        path: data/{c}/*.parquet")

    return f"""---
license:
  - cc-by-sa-4.0
  - odc-by
  - cc-by-4.0
language:
  - en
tags:
  - mathematics
  - theorems
  - semantic-search
  - information-retrieval
  - LaTeX
pretty_name: mathlas corpus
size_categories:
  - 1M<n<10M
configs:
{chr(10).join(configs_yaml)}
---

# mathlas corpus

The document corpus behind [mathlas](https://github.com/Archerkattri/mathlas),
an open MCP tool that lets an AI find the existing mathematics that solves a
problem and verify the match. This dataset is the full text + metadata side of
the served retrieval index: {total:,} theorem-level documents, each carrying a
natural-language **slogan** (the meaning of the theorem in plain English), the
real **statement** (LaTeX), and full provenance.
{excl}
It is NOT the embedding matrix. mathlas embeds each document's slogan with
Qwen3-Embedding-8B and serves dense + BM25 + reciprocal-rank-fusion retrieval;
this release lets anyone rebuild that index (or their own) from the texts.

## Configs

| Config | Rows | What it is | License |
|---|---|---|---|
| `theoremsearch` (default) | {n_ts:,} | The permissively licensed TheoremSearch subset: theorem statements + DeepSeek-V3.1 slogans from arXiv, ProofWiki, the Stacks Project, and open textbooks. Redistributed from [uw-math-ai/theorem-search-dataset](https://huggingface.co/datasets/uw-math-ai/theorem-search-dataset). | CC BY-SA 4.0 |
| `dolma` | {n_dolma:,} | Theorem-environment excerpts we extracted from the open Dolma v1.7 arXiv corpus ([emozilla/dolma-v1_7-arxiv](https://huggingface.co/datasets/emozilla/dolma-v1_7-arxiv)), cleaned and deduplicated, with NL slogans we generated locally (Qwen3.6-35B-A3B). | statements ODC-BY 1.0, slogans CC BY 4.0 |
| `findings` | {n_find:,} | The live findings store: web-research records added through the mathlas `add_finding` tool (our slogans + short quoted statements + source URLs). Embedding vectors are stripped. | CC BY 4.0 |

Per-source breakdown inside the main configs (`source_key` column):

| source_key | Rows |
|---|---|
{row("arxiv (TheoremSearch)", "arxiv")}
{row("proofwiki (TheoremSearch)", "proofwiki")}
{row("stacks (TheoremSearch)", "stacks")}
{row("other open textbooks (TheoremSearch)", "other")}
{row("dolma (ours)", "dolma")}

Total parquet size: about {gb:.1f} GB (zstd).

## Schema

`theoremsearch` and `dolma` configs share one schema:

| Column | Type | Description |
|---|---|---|
| `doc_id` | string | Stable document id (TheoremSearch theorem_id, or `dolma:<sha>#<n>`) |
| `name` | string or null | Theorem display name, e.g. "Lemma 46.3.3." |
| `slogan` | string | Natural-language denotation of the theorem (what mathlas embeds) |
| `statement` | string | The real theorem statement (LaTeX) |
| `source` | string or null | URL or identifier of the original source |
| `title` | string or null | Paper or chapter title |
| `label` | string or null | LaTeX label tag from the source |
| `citations` | int64 or null | Paper citation count where known |
| `category` | string or null | Primary arXiv category, e.g. `math.AG` |
| `source_key` | string | One of `arxiv`, `proofwiki`, `stacks`, `other`, `dolma` |

`findings` schema: `doc_id`, `name`, `slogan`, `statement`, `source`,
`provenance` (always `web_added`), `added_ts` (unix time).

## Load

```python
from datasets import load_dataset

ts = load_dataset("{DEFAULT_REPO}", "theoremsearch", split="train")
dolma = load_dataset("{DEFAULT_REPO}", "dolma", split="train")
```

## How mathlas uses it

Every document is embedded by its slogan (name + slogan, never the raw LaTeX),
so the whole index lives in one slogan-dense space and queries match on concept
rather than notation. BM25 runs over name + slogan + statement + label so exact
symbol and term hits still land. The two channels are fused by reciprocal rank
fusion. Honest headline recall at the full {total:,}-doc scale: R@1 0.614 and
R@10 0.832 querying by a document's raw body against its slogan-embedded entry
(the hard cross-representation regime).

## Licensing and provenance

Full audit: [docs/HF_DATASET_LICENSING.md](https://github.com/Archerkattri/mathlas/blob/main/docs/HF_DATASET_LICENSING.md).

* `theoremsearch`: redistributed from the upstream CC BY-SA 4.0 dataset by the
  UW Math AI Lab. That subset exists precisely because every underlying paper
  carries a permissive license (CC BY 4.0 for the large majority, plus
  CC BY-SA, CC BY 3.0, CC0 and public domain); the upstream card documents the
  extraction. Sub-sources keep their own terms: Stacks Project (GNU FDL),
  ProofWiki (CC BY-SA 3.0).
* `dolma`: the statements are short theorem-statement excerpts of arXiv papers,
  taken from the open ODC-BY Dolma v1.7 corpus released by the Allen Institute
  for AI; copyright of the underlying papers remains with their authors, and
  every record keeps its provenance. The slogans are our own generated
  annotations, released CC BY 4.0.
* `findings`: our own records, CC BY 4.0; quoted statements are brief excerpts
  with full source URLs.

If you are an author and want a record removed, open an issue on the mathlas
repository and we will remove it promptly.

## Citation

If you use this corpus, please cite mathlas:

```bibtex
@software{{attri_mathlas,
  author = {{Attri, Krishi}},
  title  = {{mathlas: airtight math tools an AI uses over MCP}},
  url    = {{https://github.com/Archerkattri/mathlas}},
  doi    = {{{ZENODO_DOI}}}
}}
```

Please also credit the upstream sources you use:

```bibtex
@article{{theoremsearch2026,
  title   = {{Semantic Search over 9 Million Mathematical Theorems}},
  author  = {{UW Math AI Lab}},
  journal = {{arXiv preprint arXiv:2602.05216}},
  year    = {{2026}}
}}

@article{{dolma2024,
  title   = {{Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research}},
  author  = {{Soldaini, Luca and others}},
  journal = {{arXiv preprint arXiv:2402.00159}},
  year    = {{2024}}
}}
```
"""


# ---------------------------------------------------------------------------
# Validate (dry-run)
# ---------------------------------------------------------------------------


def validate(out_dir: str = DEFAULT_OUT, sample: bool = True) -> dict:
    """Load every built config with ``datasets`` from the local path and verify
    row counts against stats.json; spot-check one record per source_key."""
    from datasets import load_dataset
    with open(os.path.join(out_dir, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    cache = os.path.join(out_dir, ".validate_cache")
    report = {}
    for c, info in stats["configs"].items():
        files = sorted(
            os.path.join(out_dir, "data", c, p)
            for p in os.listdir(os.path.join(out_dir, "data", c))
            if p.endswith(".parquet"))
        ds = load_dataset("parquet", data_files={"train": files},
                          split="train", cache_dir=cache)
        ok = len(ds) == info["rows"]
        entry = {"rows_loaded": len(ds), "rows_built": info["rows"],
                 "match": ok, "columns": ds.column_names, "samples": {}}
        if sample and len(ds):
            seen = set()
            # first + last + a middle row catch shard boundaries cheaply
            for i in {0, len(ds) // 2, len(ds) - 1}:
                r = ds[int(i)]
                key = r.get("source_key", c)
                if key not in seen:
                    seen.add(key)
                    entry["samples"][key] = {
                        k: (v[:120] if isinstance(v, str) else v)
                        for k, v in r.items()}
        report[c] = entry
        if not ok:
            raise RuntimeError(
                f"validation FAILED for config '{c}': loaded "
                f"{len(ds):,} rows, built {info['rows']:,}")
    return report


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


def push(out_dir: str = DEFAULT_OUT, repo_id: str = DEFAULT_REPO,
         private: bool = True) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("--push needs HF_TOKEN in the environment "
                         "(HF_TOKEN=hf_... python3 scripts/hf_upload_corpus.py --push)")
    if not os.path.exists(os.path.join(out_dir, "stats.json")):
        raise SystemExit(f"nothing built at {out_dir}; run --dry-run first "
                         f"(or let --push build by not passing --no-build)")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private,
                    exist_ok=True)
    api.upload_folder(
        folder_path=out_dir, repo_id=repo_id, repo_type="dataset",
        ignore_patterns=[".validate_cache/*", ".validate_cache*"],
        commit_message="mathlas document corpus: theoremsearch + dolma + findings")
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"pushed to {url} (private={private}); flip public in repo settings "
          f"when ready")
    return url


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="build locally + validate with datasets + print stats")
    ap.add_argument("--push", action="store_true",
                    help="upload to the Hub (needs HF_TOKEN; builds first if "
                         "no local build exists)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--public", action="store_true",
                    help="create the repo public (default: private)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--findings", default=DEFAULT_FINDINGS)
    ap.add_argument("--rows-per-shard", type=int, default=400_000)
    ap.add_argument("--exclude-source", action="append", default=[],
                    metavar="KEY",
                    help="exclude a source key for licensing reasons "
                         "(arxiv|proofwiki|stacks|other|dolma|findings); "
                         "repeatable")
    ap.add_argument("--no-strict", action="store_true",
                    help="skip the audited-count gate (re-audit the licensing "
                         "doc first)")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.push):
        ap.error("pick --dry-run and/or --push")

    have_build = os.path.exists(os.path.join(args.out, "stats.json"))
    if args.dry_run or not have_build or args.force_rebuild:
        print(f"[build] {args.meta} -> {args.out}", flush=True)
        stats = build(args.out, meta_path=args.meta,
                      findings_path=args.findings,
                      rows_per_shard=args.rows_per_shard,
                      exclude_sources=args.exclude_source,
                      strict=not args.no_strict)
        print(json.dumps(stats, indent=2))

    if args.dry_run:
        print("[validate] loading every config with datasets ...", flush=True)
        report = validate(args.out)
        print(json.dumps(report, indent=2))
        print("[dry-run OK] all configs load; counts match build stats")

    if args.push:
        push(args.out, repo_id=args.repo, private=not args.public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
