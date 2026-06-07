#!/usr/bin/env python3
"""Extract theorem-like environments from arXiv full-text LaTeX.

Source: emozilla/dolma-v1_7-arxiv (~1.87M arXiv papers, full LaTeX in 'text' field)
Output: reference/downloads/datasets/arxiv_fulltext_theorems/theorems_NNNNN.jsonl
        shards of ~500k records each.

Safety constraints:
- streaming=True: never materialises the full dataset in RAM
- One example at a time, RSS monitored
- OMP_NUM_THREADS=2
- Resumable via progress checkpoint file

Run:
    OMP_NUM_THREADS=2 python3 scripts/build_arxiv_fulltext_corpus.py [--max-papers N]
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── env ──────────────────────────────────────────────────────────────────────
ME = Path(__file__).resolve().parent.parent   # math_engine root
OUT_DIR = ME / "reference/downloads/datasets/arxiv_fulltext_theorems"
PROGRESS_FILE = OUT_DIR / "progress.json"
HF_CACHE = ME / "reference/downloads/hf_datasets"

os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))
os.environ.setdefault("OMP_NUM_THREADS", "2")

DATASET_ID = "emozilla/dolma-v1_7-arxiv"
SHARD_SIZE = 500_000   # records per output shard
MAX_STATEMENT_CHARS = 3000
RSS_WARN_MB = 5500     # warn if RSS > 5.5GB
RSS_ABORT_MB = 6000    # abort if RSS > 6GB

# ── theorem environments to capture ──────────────────────────────────────────
BUILTIN_ENVS: set[str] = {
    "theorem", "thm",
    "lemma", "lem",
    "proposition", "prop",
    "corollary", "cor",
    "definition", "defn",
    "conjecture", "conj",
    "claim",
    "remark", "rmk",
    "example", "ex",
    "fact",
    "observation",
    "note",
    "notation",
    "axiom",
    "hypothesis",
}

# Map abbreviated env names → canonical type label
ENV_CANONICAL = {
    "thm": "theorem",
    "lem": "lemma",
    "prop": "proposition",
    "cor": "corollary",
    "defn": "definition",
    "conj": "conjecture",
    "rmk": "remark",
    "ex": "example",
}

# ── regex helpers ─────────────────────────────────────────────────────────────
# Detect \newtheorem{envname}{Kind} declarations
_NEWTHEOREM_RE = re.compile(
    r'\\newtheorem\*?\{([A-Za-z][A-Za-z0-9@_*]*)\}\{([^}]+)\}',
    re.MULTILINE,
)
# Also \newtheorem{env}[counter]{Kind}
_NEWTHEOREM_COUNTER_RE = re.compile(
    r'\\newtheorem\*?\{([A-Za-z][A-Za-z0-9@_*]*)\}\[[^\]]+\]\{([^}]+)\}',
    re.MULTILINE,
)

def _discover_custom_envs(latex: str) -> dict[str, str]:
    """Return {env_name: Kind} for every \\newtheorem declaration in the source."""
    custom: dict[str, str] = {}
    for m in _NEWTHEOREM_RE.finditer(latex):
        custom[m.group(1).lower()] = m.group(2).strip()
    for m in _NEWTHEOREM_COUNTER_RE.finditer(latex):
        custom[m.group(1).lower()] = m.group(2).strip()
    return custom


def _build_env_pattern(extra_envs: set[str]) -> re.Pattern:
    """Build a begin{...} pattern for all known theorem-like envs."""
    all_envs = BUILTIN_ENVS | extra_envs
    alt = "|".join(re.escape(e) for e in sorted(all_envs, key=len, reverse=True))
    # Match \begin{env} or \begin{env*} optionally followed by [...][...]
    return re.compile(
        r'\\begin\{(' + alt + r')\*?\}',
        re.IGNORECASE,
    )


def _extract_env_body(latex: str, begin_pos: int, env_name: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract the body of \\begin{env}[optional name]...\\end{env} starting at begin_pos.
    Returns (body_text, optional_name) or (None, None) if malformed.
    Caps body at MAX_STATEMENT_CHARS.
    """
    end_re = re.compile(
        re.escape(r'\end{') + re.escape(env_name) + r'\*?\}',
        re.IGNORECASE,
    )
    # Move past \begin{env}
    start = latex.find(r'\begin{', begin_pos)
    if start == -1:
        return None, None
    # Skip past the } of \begin{env_name}
    open_close = latex.find('}', start + len(r'\begin{'))
    if open_close == -1:
        return None, None
    body_start = open_close + 1

    # Extract optional [name] argument: \begin{theorem}[Pythagoras]
    # Filter out citation/reference noise: skip if it looks like \cite{...}
    opt_name: Optional[str] = None
    after_begin = latex[body_start:body_start + 200]
    opt_m = re.match(r'^\s*\[([^\]]{1,120})\]', after_begin)
    if opt_m:
        raw_name = opt_m.group(1).strip()
        # Accept only if it looks like a descriptive name, not a citation
        if raw_name and '\\cite' not in raw_name and '\\ref' not in raw_name:
            opt_name = raw_name
        body_start += opt_m.end()

    m = end_re.search(latex, body_start)
    if m is None:
        return None, opt_name
    body = latex[body_start:m.start()].strip()
    return body[:MAX_STATEMENT_CHARS], opt_name


def _extract_label(body: str) -> Optional[str]:
    """Pull the first \\label{...} from the environment body."""
    m = re.search(r'\\label\{([^}]+)\}', body)
    return m.group(1).strip() if m else None


def extract_theorems(doc_id: str, latex: str) -> list[dict]:
    """
    Parse one paper's LaTeX and return a list of theorem records.
    Each record: {doc_id, paper_id, env_type, name, statement, label, source}
    """
    results = []
    # 1. Discover custom theorem environments declared in this paper
    custom_env_map = _discover_custom_envs(latex)  # {lower_env: Kind}
    extra_envs = set(custom_env_map.keys())

    # Build pattern
    pattern = _build_env_pattern(extra_envs)

    for m in pattern.finditer(latex):
        env_raw = m.group(1).lower().rstrip('*')
        # Determine canonical type
        if env_raw in ENV_CANONICAL:
            env_type = ENV_CANONICAL[env_raw]
        elif env_raw in BUILTIN_ENVS:
            env_type = env_raw
        elif env_raw in custom_env_map:
            env_type = custom_env_map[env_raw].lower()
        else:
            env_type = env_raw

        body, opt_name = _extract_env_body(latex, m.start(), m.group(1))
        if body is None or len(body.strip()) < 10:
            continue

        label = _extract_label(body)
        # Remove the \label command from the statement text
        statement = re.sub(r'\\label\{[^}]+\}', '', body).strip()
        if len(statement) < 10:
            continue

        results.append({
            "doc_id": doc_id,
            "paper_id": None,          # filled in caller
            "env_type": env_type,
            "name": opt_name,          # optional title e.g. \begin{theorem}[Pythagoras]
            "statement": statement,
            "label": label,
            "source": DATASET_ID,
        })
    # Within-paper dedup: a `\newtheorem{thm}{...}` custom env and the builtin `thm` can match
    # the SAME body twice (env-name collision), producing exact intra-paper duplicates. Keep the
    # first occurrence per distinct statement.
    seen_stmts: set[str] = set()
    deduped = []
    for r in results:
        if r["statement"] not in seen_stmts:
            seen_stmts.add(r["statement"])
            deduped.append(r)
    return deduped


# ── arxiv id extraction from dolma id field ──────────────────────────────────
_ARXIV_ID_RE = re.compile(r'arxiv\.org/abs/([0-9]{4}\.[0-9]+(?:v\d+)?)', re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(r'arxiv\.org/abs/([a-z\-]+/\d+(?:v\d+)?)', re.IGNORECASE)

def _parse_paper_id(doc_id_field: str) -> Optional[str]:
    if not doc_id_field:
        return None
    m = _ARXIV_ID_RE.search(doc_id_field)
    if m:
        return m.group(1)
    m = _ARXIV_OLD_RE.search(doc_id_field)
    if m:
        return m.group(1)
    return doc_id_field


# ── memory monitoring ─────────────────────────────────────────────────────────
def _rss_mb() -> float:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


# ── shard writer ──────────────────────────────────────────────────────────────
class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int = SHARD_SIZE):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._shard_idx = 0
        self._buf: list[str] = []
        self._total = 0
        self._fh = None
        self._open_next()

    def _shard_path(self, idx: int) -> Path:
        return self.out_dir / f"theorems_{idx:05d}.jsonl"

    def _open_next(self):
        if self._fh:
            self._fh.close()
        # Find the first shard index that doesn't exist yet
        while self._shard_path(self._shard_idx).exists():
            # Count existing records in this shard
            existing = sum(1 for _ in open(self._shard_path(self._shard_idx)))
            if existing < self.shard_size:
                # Resume appending to this shard
                self._fh = open(self._shard_path(self._shard_idx), 'a', buffering=1)
                self._total += existing
                return
            self._shard_idx += 1
        self._fh = open(self._shard_path(self._shard_idx), 'w', buffering=1)

    @property
    def total(self) -> int:
        return self._total

    def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + '\n')
        self._total += 1
        if self._total % self.shard_size == 0:
            self._shard_idx += 1
            self._open_next()

    def flush(self):
        if self._fh:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


# ── progress checkpoint ───────────────────────────────────────────────────────
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"papers_processed": 0, "theorems_written": 0, "start_time": None}


def save_progress(papers: int, theorems: int, start_time: float):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            "papers_processed": papers,
            "theorems_written": theorems,
            "start_time": start_time,
            "last_update": time.time(),
        }, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-papers', type=int, default=None,
                   help='Stop after this many papers (for testing)')
    p.add_argument('--log-every', type=int, default=5000,
                   help='Print progress every N papers')
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    # Load progress checkpoint
    prog = load_progress()
    resume_from = prog["papers_processed"]
    start_time = prog.get("start_time") or time.time()

    print(f"[corpus] Source: {DATASET_ID}")
    print(f"[corpus] Output: {OUT_DIR}")
    if resume_from > 0:
        print(f"[corpus] RESUMING from paper #{resume_from:,}, theorems so far: {prog['theorems_written']:,}")
    else:
        print("[corpus] Starting fresh run")

    # Import datasets lazily
    try:
        import datasets as hf_datasets
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    hf_datasets.disable_progress_bar()

    # Open shard writer (will auto-resume to last shard)
    writer = ShardWriter(OUT_DIR)
    # Sync writer total to checkpoint
    if writer.total < prog["theorems_written"]:
        print(f"[corpus] WARNING: writer total ({writer.total}) < checkpoint ({prog['theorems_written']})")

    print(f"[corpus] Writer starting at total={writer.total:,} theorems")

    # Load streaming dataset
    print(f"[corpus] Loading {DATASET_ID} (streaming)...")
    ds = hf_datasets.load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
    )

    papers_processed = 0
    papers_in_session = 0
    last_rss_check = 0

    try:
        for ex in ds:
            # Skip already-processed papers
            if papers_processed < resume_from:
                papers_processed += 1
                continue

            text = ex.get("text", "")
            doc_id = ex.get("id", f"doc_{papers_processed}")
            paper_id = _parse_paper_id(str(doc_id))

            theorems = extract_theorems(doc_id, text)
            for t in theorems:
                t["paper_id"] = paper_id
                writer.write(t)

            papers_processed += 1
            papers_in_session += 1

            # Periodic logging + checkpoint
            if papers_in_session % args.log_every == 0:
                rss = _rss_mb()
                elapsed = time.time() - start_time
                rate = papers_processed / elapsed if elapsed > 0 else 0
                eta_h = (1_870_000 - papers_processed) / (rate * 3600) if rate > 0 else float('inf')
                print(
                    f"[corpus] papers={papers_processed:,}  theorems={writer.total:,}  "
                    f"RSS={rss:.0f}MB  rate={rate:.0f}/s  ETA={eta_h:.1f}h"
                )
                writer.flush()
                save_progress(papers_processed, writer.total, start_time)

                # Memory guard
                if rss > RSS_ABORT_MB:
                    print(f"[corpus] ABORT: RSS {rss:.0f}MB > {RSS_ABORT_MB}MB limit", file=sys.stderr)
                    break
                elif rss > RSS_WARN_MB:
                    print(f"[corpus] WARN: RSS {rss:.0f}MB > {RSS_WARN_MB}MB — GC", file=sys.stderr)
                    gc.collect()

                last_rss_check = papers_processed

            if args.max_papers and papers_processed >= resume_from + args.max_papers:
                print(f"[corpus] Reached --max-papers={args.max_papers}, stopping")
                break

    except KeyboardInterrupt:
        print("\n[corpus] Interrupted — saving checkpoint")
    finally:
        writer.flush()
        writer.close()
        save_progress(papers_processed, writer.total, start_time)

    elapsed = time.time() - start_time
    print()
    print(f"[corpus] DONE")
    print(f"[corpus] Papers processed: {papers_processed:,}")
    print(f"[corpus] Theorems written: {writer.total:,}")
    print(f"[corpus] Elapsed: {elapsed/3600:.1f}h")
    print(f"[corpus] Output: {OUT_DIR}")

    # Disk usage
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', str(OUT_DIR)], capture_output=True, text=True)
        print(f"[corpus] Disk used: {result.stdout.strip()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
    # Use os._exit to bypass Python GIL teardown crash with C extension modules
    # (Python 3.11 + pyarrow/torch teardown deadlock). Script completed successfully.
    os._exit(0)
