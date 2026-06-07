"""Integer-sequence identification against a LOCAL copy of OEIS -- NO LLM.

mathlas is a tool an AI *uses*, never one that uses an AI. This module gives the
calling AI an airtight integer-sequence lookup: hand it a list of integers and it
returns the matching OEIS entries (A-number, name, URL) by **EXACT term match**
against a local copy of the OEIS data -- no model, no API key, no guessing. Either
the terms occur (as a contiguous run) in a stored sequence or they do not; the
verdict is mechanical, not an opinion.

Why local + exact (the airtight discipline, same spirit as the numeric tier)
----------------------------------------------------------------------------
The numeric beachhead's rule is *airtight or nothing* -- a match is a real,
independently-checkable fact, not a plausible-looking guess. Sequence lookup gets
the same treatment: we match terms against the actual OEIS data verbatim. There is
no fuzzy scoring, no embedding, no LLM -- a returned A-number provably contains the
queried run of terms. (For the same airtightness we do exact integer comparison on
Python ``int`` -- arbitrary precision -- so big OEIS terms never lose digits.)

Data (downloaded once, parsed once, cached)
-------------------------------------------
Two gzip files published by OEIS under their end-user license:

  * ``stripped.gz``  -- ``A000045 ,0,1,1,2,3,5,8,13,21,...``  (terms, comma-led)
  * ``names.gz``     -- ``A000045 Fibonacci numbers: F(n) = F(n-1) + F(n-2) ...``

Place them in a directory (default search includes
``reference/downloads/oeis``). They total ~tens of MB and are gitignored. The
parse builds, ONCE per process (data-flow discipline -- every per-call load is
cached), an n-gram index from short runs of consecutive terms to the sequences
containing them, so lookups are fast even over ~400k sequences and never re-read
the files.

Matching (airtight, with reasonable subsequence/offset handling)
----------------------------------------------------------------
A query matches a stored sequence iff the query's terms appear as a **contiguous
sub-run** anywhere in that sequence (so a leading-term/offset difference -- e.g.
``[1,1,2,3,5,8,13,21]`` vs Fibonacci's stored ``0,1,1,2,3,5,8,...`` -- still
matches). Matches are ranked by how early the run starts (prefix matches first)
then by A-number, and the offset at which the run was found is reported so the AI
can see whether it was an exact prefix or an interior/offset hit.
"""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Configuration.
# --------------------------------------------------------------------------- #
#: Length of the consecutive-term key used to index sequences. A query must
#: supply at least this many terms for an indexed lookup (fewer terms are far too
#: ambiguous to be an "airtight" identification anyway). 4 consecutive integers is
#: already highly selective across OEIS while keeping the index light.
NGRAM = 4

#: Minimum number of query terms we will attempt to identify. Below this the
#: result would not be an identification -- it would be a near-universal match.
MIN_QUERY_TERMS = 4

#: Default places to look for the OEIS data files, in order. The first directory
#: containing ``stripped.gz`` (and ideally ``names.gz``) wins. Override with the
#: ``MATHLAS_OEIS_DIR`` env var or the ``data_dir`` argument.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_DATA_DIRS: Tuple[str, ...] = (
    os.environ.get("MATHLAS_OEIS_DIR", ""),
    os.path.join(_REPO_ROOT, "reference", "downloads", "oeis"),
    os.path.join(os.getcwd(), "reference", "downloads", "oeis"),
    os.path.join(os.getcwd(), "oeis"),
)

OEIS_STRIPPED = "stripped.gz"
OEIS_NAMES = "names.gz"

#: How to turn an A-number into its canonical OEIS page.
OEIS_URL = "https://oeis.org/{anum}"

#: A line of the stripped file: ``A000045 ,0,1,1,2,...`` -- captures the A-number
#: and the comma-separated body (which itself begins with a comma).
_STRIPPED_RE = re.compile(r"^(A\d{6,})\s+,(.*?),?\s*$")
_NAME_RE = re.compile(r"^(A\d{6,})\s+(.*)$")


# --------------------------------------------------------------------------- #
# Result types.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SequenceMatch:
    """One OEIS sequence whose stored terms contain the queried run exactly."""
    a_number: str           # e.g. "A000045"
    name: str               # the OEIS name/description (may be "")
    url: str                # canonical OEIS page
    offset: int             # index in the stored terms where the query run starts
    exact_prefix: bool      # True iff offset == 0 (query is a leading prefix)
    matched_terms: int      # how many query terms were matched (== len(query))


@dataclass(frozen=True)
class SequenceResult:
    """The outcome of an integer-sequence identification (airtight, NO LLM)."""
    query: List[int]
    matches: List[SequenceMatch]
    data_dir: Optional[str]
    note: str

    @property
    def identified(self) -> bool:
        return bool(self.matches)


# --------------------------------------------------------------------------- #
# Parsing helpers.
# --------------------------------------------------------------------------- #
def _open_text(path: str) -> Iterable[str]:
    """Yield decoded lines from a (possibly gzip) text file, tolerant of stray
    bytes. OEIS files are gzip; a plain-text copy is also accepted."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line


def _parse_terms(body: str) -> List[int]:
    """Parse the comma-separated integer body of a stripped-file line.

    Airtight: uses Python ``int`` (arbitrary precision) so large OEIS terms keep
    every digit. A malformed token aborts that line's parse (returns what parsed
    cleanly up to the bad token) rather than guessing."""
    out: List[int] = []
    for tok in body.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            break
    return out


def _find_data_file(name: str, data_dir: Optional[str]) -> Optional[str]:
    """Locate an OEIS data file by name, honoring an explicit dir then defaults."""
    dirs: List[str] = []
    if data_dir:
        dirs.append(data_dir)
    dirs.extend(d for d in DEFAULT_DATA_DIRS if d)
    for d in dirs:
        cand = os.path.join(d, name)
        if os.path.isfile(cand):
            return cand
    return None


# --------------------------------------------------------------------------- #
# The local OEIS index -- built once, cached (data-flow discipline).
# --------------------------------------------------------------------------- #
class OEISIndex:
    """An in-memory, exact-match index over a local copy of OEIS.

    Built once from ``stripped.gz`` (+ optional ``names.gz``) and reused for every
    lookup. Stores, per sequence, its full parsed term list; and an n-gram index
    mapping each length-``NGRAM`` run of consecutive terms to the A-numbers that
    contain that run, so candidate sequences are found without scanning all ~400k.
    """

    def __init__(self, ngram: int = NGRAM) -> None:
        self.ngram = ngram
        # A-number -> full list of stored terms.
        self._terms: Dict[str, List[int]] = {}
        # A-number -> name.
        self._names: Dict[str, str] = {}
        # length-`ngram` consecutive-term key -> list of A-numbers containing it.
        self._ngram_index: Dict[Tuple[int, ...], List[str]] = {}
        self.data_dir: Optional[str] = None
        self.n_sequences = 0

    # -- construction -- #
    def load(self, data_dir: Optional[str] = None) -> "OEISIndex":
        """Load and index the OEIS data. Raises ``FileNotFoundError`` if the
        ``stripped.gz`` terms file cannot be found (names are optional)."""
        stripped = _find_data_file(OEIS_STRIPPED, data_dir)
        if not stripped:
            searched = [data_dir] if data_dir else []
            searched += [d for d in DEFAULT_DATA_DIRS if d]
            raise FileNotFoundError(
                f"OEIS terms file '{OEIS_STRIPPED}' not found. Looked in: "
                f"{', '.join(searched)}. Download it (and names.gz) from "
                "https://oeis.org/stripped.gz / https://oeis.org/names.gz into "
                "reference/downloads/oeis/ (see docs/methods.md).")
        self.data_dir = os.path.dirname(stripped)

        # Parse terms + build the n-gram index.
        ng = self.ngram
        for line in _open_text(stripped):
            if not line or line[0] != "A":   # skip comments / blanks fast
                continue
            m = _STRIPPED_RE.match(line)
            if not m:
                continue
            anum, body = m.group(1), m.group(2)
            terms = _parse_terms(body)
            if not terms:
                continue
            self._terms[anum] = terms
            seen_keys = set()  # avoid duplicate posting for repeating runs
            if len(terms) >= ng:
                for i in range(len(terms) - ng + 1):
                    key = tuple(terms[i:i + ng])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    self._ngram_index.setdefault(key, []).append(anum)
        self.n_sequences = len(self._terms)

        # Parse names (optional but expected).
        names = _find_data_file(OEIS_NAMES, data_dir or self.data_dir)
        if names:
            for line in _open_text(names):
                if not line or line[0] != "A":
                    continue
                m = _NAME_RE.match(line)
                if m:
                    self._names[m.group(1)] = m.group(2).strip()
        return self

    # -- lookup -- #
    def name_of(self, anum: str) -> str:
        return self._names.get(anum, "")

    def _candidate_anums(self, query: Sequence[int]) -> List[str]:
        """A-numbers that contain the query's leading length-``ngram`` run.

        We key on the FIRST window of the query; the full contiguous match is then
        verified term-by-term in :meth:`find`. (One key suffices to gather every
        candidate, because any sequence containing the whole query necessarily
        contains its first window.)"""
        ng = self.ngram
        if len(query) < ng:
            return []
        key = tuple(query[:ng])
        return self._ngram_index.get(key, [])

    @staticmethod
    def _match_offset(stored: Sequence[int], query: Sequence[int]) -> Optional[int]:
        """Lowest index where ``query`` occurs as a contiguous sub-run of
        ``stored``, or None. Exact integer comparison (airtight)."""
        n, q = len(stored), len(query)
        if q == 0 or q > n:
            return None
        first = query[0]
        last_start = n - q
        i = 0
        while i <= last_start:
            if stored[i] == first and list(stored[i:i + q]) == list(query):
                return i
            i += 1
        return None

    def find(self, query: Sequence[int], max_results: int = 5) -> List[SequenceMatch]:
        """All OEIS sequences containing ``query`` as a contiguous sub-run.

        Ranked by **A-number ascending** -- OEIS assigns low A-numbers to the
        canonical / most-fundamental sequences, so the foundational entry (e.g.
        Fibonacci A000045) sorts to the top rather than being buried under
        coincidental prefix-sharers with high A-numbers. This mirrors how OEIS
        itself orders results. ``offset``/``exact_prefix`` are reported per match
        (so the AI sees prefix vs interior/offset hits) but do NOT reorder away
        the canonical sequence. Airtight -- every match is verified term-by-term.
        """
        q = [int(x) for x in query]
        matches: List[SequenceMatch] = []
        for anum in self._candidate_anums(q):
            stored = self._terms.get(anum)
            if stored is None:
                continue
            off = self._match_offset(stored, q)
            if off is None:
                continue
            matches.append(SequenceMatch(
                a_number=anum, name=self.name_of(anum),
                url=OEIS_URL.format(anum=anum), offset=off,
                exact_prefix=(off == 0), matched_terms=len(q)))
        # Sort by numeric A-number ascending (canonical entries first). The
        # A-number is "A" + zero-padded digits, so lexicographic == numeric.
        matches.sort(key=lambda m: m.a_number)
        return matches[:max_results]


# --------------------------------------------------------------------------- #
# Process-wide cache of loaded indexes (one per data_dir) -- data-flow discipline.
# --------------------------------------------------------------------------- #
_INDEX_CACHE: Dict[str, OEISIndex] = {}


def get_index(data_dir: Optional[str] = None) -> OEISIndex:
    """Return a cached, loaded :class:`OEISIndex` for ``data_dir`` (or the first
    default dir that has the data). Built once per process and reused -- the files
    are never re-read after the first load (the 50x-speedup data-flow lesson)."""
    # Resolve to the actual data directory so different spellings of the same dir
    # share a cache entry.
    stripped = _find_data_file(OEIS_STRIPPED, data_dir)
    key = os.path.dirname(stripped) if stripped else (data_dir or "<none>")
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = OEISIndex().load(data_dir)
        _INDEX_CACHE[key] = idx
        # also key by the resolved dir so a later None/relative call hits the cache
        if idx.data_dir:
            _INDEX_CACHE[idx.data_dir] = idx
    return idx


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def identify_sequence(terms: Sequence[int], max_results: int = 5,
                      data_dir: Optional[str] = None) -> SequenceResult:
    """Identify an integer sequence against the local OEIS data -- airtight, NO LLM.

    Parameters
    ----------
    terms : sequence of int
        The integer sequence to identify (>= ``MIN_QUERY_TERMS`` terms).
    max_results : int
        Cap on returned matches (default 5).
    data_dir : str, optional
        Directory holding ``stripped.gz``/``names.gz``; defaults to the standard
        search path (``reference/downloads/oeis`` etc.).

    Returns a :class:`SequenceResult`: the A-number, name and OEIS URL of every
    sequence whose stored terms contain the queried run exactly (contiguous,
    offset/subsequence-tolerant). If the data files are missing, returns a result
    with a clear note (never a fake match).
    """
    q: List[int] = []
    for x in terms:
        # accept ints, or strings/floats that are exact integers
        if isinstance(x, bool):
            raise TypeError("sequence terms must be integers, not booleans")
        if isinstance(x, int):
            q.append(x)
        else:
            iv = int(x)
            if iv != x:                       # reject non-integer floats
                raise TypeError(f"non-integer term in sequence: {x!r}")
            q.append(iv)

    if len(q) < MIN_QUERY_TERMS:
        return SequenceResult(
            query=q, matches=[], data_dir=None,
            note=(f"Need >= {MIN_QUERY_TERMS} terms for an airtight identification "
                  f"(got {len(q)}); fewer terms match too many sequences to be an "
                  "identification."))

    try:
        idx = get_index(data_dir)
    except FileNotFoundError as e:
        return SequenceResult(
            query=q, matches=[], data_dir=None,
            note=("OEIS data not available -> UNDETERMINED (honest, not a fake "
                  f"match). {e}"))

    matches = idx.find(q, max_results=max_results)
    if matches:
        note = (f"Airtight EXACT term-match against local OEIS "
                f"({idx.n_sequences} sequences). Each match's stored terms contain "
                "your run verbatim (contiguous; offset reported). NO LLM.")
    else:
        note = (f"No OEIS sequence contains these exact terms as a contiguous run "
                f"(searched {idx.n_sequences} local sequences) -> UNIDENTIFIED "
                "(honest). Try giving more/fewer terms, or check for typos. NO LLM.")
    return SequenceResult(query=q, matches=matches, data_dir=idx.data_dir, note=note)


__all__ = [
    "identify_sequence", "SequenceResult", "SequenceMatch", "OEISIndex",
    "get_index", "NGRAM", "MIN_QUERY_TERMS", "OEIS_URL",
]
