"""FUNSEARCH HARNESS — the AI is the program GENERATOR; mathlas is the HARNESS.

FunSearch (Romera-Paredes et al., *Nature* 2024) pairs an LLM that proposes
programs with a *deterministic evaluator* that scores them, kept in an
evolutionary program database; iterating the loop discovers programs that beat
the prior best (it found larger cap sets and better bin-packing heuristics).

mathlas implements the **HARNESS half ONLY — NO LLM**. The CALLING AI writes the
candidate programs (that is the "language model" in FunSearch); mathlas:

  * ``evaluate(program_src, problem_id)`` — runs a candidate in a SANDBOXED
    subprocess (OS-level isolation: RLIMIT_NOFILE=3 blocks all new fd allocations
    including sockets and file opens; RLIMIT_NPROC=0 blocks fork; hard CPU/memory
    rlimits; wall-clock timeout; stripped env; throwaway cwd) against the
    registered scorer and returns the numeric score or the error string.
    Deterministic, no model.
  * ``register(program_src, score, problem_id)`` — stores the program in an
    island / MAP-Elites database ON DISK (under
    ``reference/downloads/funsearch/``, gitignored) and reports whether it is a
    new best (globally and in its behaviour cell).
  * ``status(problem_id)`` — returns the current best program(s) + score + the
    FEW-SHOT CONTEXT (problem spec + top programs) the AI uses to write its next,
    better variant. That few-shot block is exactly FunSearch's "best-shot"
    prompt, assembled as DATA — mathlas never calls a model with it.

Shipped problems (so the loop runs end-to-end with zero setup):
  * ``cap_set`` — size of a cap set in Z_3^n (no 3 points collinear); the headline
    FunSearch result. Program returns a list of vectors; scorer validates the
    no-line property and returns the size (invalid -> -inf).
  * ``online_bin_packing`` — an online best-fit-style heuristic; program returns
    a priority over bins for an incoming item; scorer runs it on fixed streams and
    returns -(#bins used) (so larger = better, FunSearch's objective).

MAP-Elites (Mouret & Clune 2015) keeps not just the single best but the best
program in each cell of a behaviour grid, preserving diversity the AI can branch
from — the islands-of-elites structure FunSearch uses to avoid local optima.

Reference: Romera-Paredes, Barekatain, Novikov, Balog, Kumar, Dupont, Ruiz,
Ellenberg, Wang, Fawzi, Kohli & Fawzi, "Mathematical discoveries from program
search with large language models," *Nature* 625, 468-475 (2024). Open prior
art: OpenEvolve (open-source FunSearch/AlphaEvolve re-implementation).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Where the program DB lives (gitignored — reference/ is ignored wholesale).
# --------------------------------------------------------------------------- #
def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def db_dir() -> str:
    """The on-disk program-DB directory. Overridable via ``MATHLAS_FUNSEARCH_DIR``
    (e.g. for tests / ephemeral runs). Created on demand."""
    d = os.environ.get("MATHLAS_FUNSEARCH_DIR") or os.path.join(
        _repo_root(), "reference", "downloads", "funsearch")
    os.makedirs(d, exist_ok=True)
    return d


def _db_path(problem_id: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in problem_id)
    return os.path.join(db_dir(), f"{safe}.json")


# --------------------------------------------------------------------------- #
# Problem registry: each problem ships (a) a description, (b) the function name
# the AI's program must define, (c) a SCORER source string run INSIDE the sandbox
# (so scoring is isolated too), and (d) a behaviour descriptor for MAP-Elites.
#
# The scorer is plain Python text appended after the candidate in the sandbox; it
# must set ``__SCORE__`` (a float; -inf for invalid) and may set
# ``__BEHAVIOR__`` (a small tuple of ints/floats -> the MAP-Elites cell).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Problem:
    problem_id: str
    description: str
    entry_point: str               # the function name the candidate must define
    scorer_src: str                # runs in-sandbox; sets __SCORE__ / __BEHAVIOR__
    starter_src: str               # a trivial valid program (the seed few-shot)
    higher_is_better: bool = True
    note: str = ""


# ---- problem 1: cap set in Z_3^n -------------------------------------------- #
_CAPSET_SCORER = r'''
# Scorer for cap_set: the candidate must define solve(n) -> list of n-tuples over
# {0,1,2}. A CAP SET has NO three distinct points a,b,c with a+b+c == 0 mod 3
# (equivalently no 3 collinear). Score = size of the set if valid, else -inf.
import itertools
_N = globals().get("__N__", 4)
try:
    pts = solve(_N)
    pts = [tuple(int(x) % 3 for x in p) for p in pts]
    if any(len(p) != _N for p in pts):
        raise ValueError("a point has the wrong dimension")
    S = set(pts)
    if len(S) != len(pts):
        raise ValueError("duplicate points")
    ok = True
    pts_list = list(S)
    # check no 3 distinct points sum to 0 mod 3 (the cap / no-line condition)
    for a, b in itertools.combinations(pts_list, 2):
        c = tuple((-(a[i] + b[i])) % 3 for i in range(_N))
        if c != a and c != b and c in S:
            ok = False
            break
    __SCORE__ = float(len(S)) if ok else float("-inf")
    __BEHAVIOR__ = (len(S) % 8,)   # coarse behaviour cell
except Exception as _e:
    __SCORE__ = float("-inf")
    __ERROR__ = repr(_e)
'''

_CAPSET_STARTER = textwrap.dedent('''\
    def solve(n):
        # Trivial valid cap set: the n standard basis vectors + the zero vector.
        # (No three of these sum to zero mod 3 for n>=1.) The AI improves this.
        pts = [tuple(0 for _ in range(n))]
        for i in range(n):
            v = [0] * n
            v[i] = 1
            pts.append(tuple(v))
        return pts
''')


# ---- problem 2: online bin packing ------------------------------------------ #
_BINPACK_SCORER = r'''
# Scorer for online_bin_packing: the candidate defines priority(item, bins) ->
# a list of scores (one per OPEN bin) giving the remaining capacity each bin has;
# we place the item in the HIGHEST-priority bin that fits, else open a new bin.
# Items arrive online from fixed streams. Score = -(total bins used), averaged.
import random
_CAP = globals().get("__CAP__", 1.0)
_STREAMS = globals().get("__STREAMS__", None)
if _STREAMS is None:
    rng = random.Random(0)
    _STREAMS = [[round(rng.uniform(0.1, 0.7), 3) for _ in range(40)]
                for _ in range(5)]
try:
    total_bins = 0
    for stream in _STREAMS:
        bins = []   # remaining capacity of each open bin
        for item in stream:
            fit = [i for i, r in enumerate(bins) if r + 1e-9 >= item]
            if fit:
                pr = priority(item, [bins[i] for i in fit])
                # choose the fitting bin with the largest priority value
                best = max(range(len(fit)), key=lambda k: pr[k])
                bins[fit[best]] -= item
            else:
                bins.append(_CAP - item)
        total_bins += len(bins)
    avg = total_bins / len(_STREAMS)
    __SCORE__ = float(-avg)              # fewer bins -> higher score
    __BEHAVIOR__ = (int(avg) % 8,)
except Exception as _e:
    __SCORE__ = float("-inf")
    __ERROR__ = repr(_e)
'''

_BINPACK_STARTER = textwrap.dedent('''\
    def priority(item, bins):
        # Best-fit baseline: prefer the bin with the LEAST remaining capacity that
        # still fits (tightest fit). Returns a priority per candidate bin.
        # (Higher priority = chosen.) The AI improves this heuristic.
        return [-rem for rem in bins]
''')


_PROBLEMS: Dict[str, Problem] = {
    "cap_set": Problem(
        problem_id="cap_set",
        description=(
            "Find the LARGEST possible cap set in Z_3^n: a set of points in "
            "{0,1,2}^n containing NO three distinct points a,b,c with "
            "a+b+c == 0 (mod 3) (equivalently, no 3 on a line). Define "
            "`solve(n)` returning a list of n-tuples. Bigger set = higher score. "
            "This is FunSearch's headline result (it beat the prior best for n=8)."),
        entry_point="solve",
        scorer_src=_CAPSET_SCORER,
        starter_src=_CAPSET_STARTER,
        higher_is_better=True,
        note="Default n=4 (override via the evaluate `params` {'__N__': k})."),
    "online_bin_packing": Problem(
        problem_id="online_bin_packing",
        description=(
            "Design an ONLINE bin-packing heuristic. Define "
            "`priority(item, bins)` returning one score per OPEN bin (whose "
            "remaining capacities are given in `bins`); the item is placed in the "
            "highest-priority bin that fits, else a new bin opens. Items arrive "
            "online from fixed streams (bin capacity 1.0). Score = -(avg bins "
            "used), so FEWER bins = higher score. FunSearch found heuristics that "
            "beat best-fit here."),
        entry_point="priority",
        scorer_src=_BINPACK_SCORER,
        starter_src=_BINPACK_STARTER,
        higher_is_better=True,
        note="Override the streams/capacity via evaluate `params`."),
}


def problems() -> List[str]:
    return sorted(_PROBLEMS)


def get_problem(problem_id: str) -> Problem:
    p = _PROBLEMS.get(problem_id)
    if p is None:
        raise KeyError(
            f"unknown problem_id {problem_id!r}; available: {problems()}")
    return p


# --------------------------------------------------------------------------- #
# The SANDBOX: run candidate + scorer in a fresh subprocess, hard timeout, with
# OS-level resource limits in a throwaway cwd. We do NOT trust the program (it
# is AI-written); the containment layers are:
#
#   1. Process isolation — a fresh ``python -I -S`` child; parent is unaffected
#      regardless of what the child does.
#   2. RLIMIT_NOFILE=3 — after Python initialises and all needed stdlib modules
#      are loaded into sys.modules, we close every fd above 2 (stdin/stdout/
#      stderr) and cap the file-descriptor table at 3. This blocks at the OS
#      level: socket() (network), open() (filesystem reads/writes), and dynamic
#      library loads (numpy, ctypes, etc.) all require a new fd and fail with
#      EMFILE. This is NOT a monkey-patch; it is an OS-enforced hard limit.
#   3. RLIMIT_NPROC=0 — blocks os.fork() at the kernel level, so subprocess,
#      multiprocessing, and os.system() cannot spawn child processes.
#   4. RLIMIT_CPU — hard CPU-second budget; excess use triggers SIGXCPU.
#   5. RLIMIT_AS — address-space cap; prevents unbounded memory allocation.
#   6. Stripped environment — child inherits only PATH + tuning vars; no tokens,
#      credentials, or proxy settings leak through.
#   7. Wall-clock timeout (subprocess.run timeout) — primary guard, catches
#      cases where SIGXCPU is masked or the CPU limit is generous.
#
# What this sandbox does NOT prevent (be honest):
#   - Code that uses only already-loaded stdlib modules (itertools, math, json,
#     random, etc. listed in _SANDBOX_PREIMPORTS) can still run freely — that is
#     intentional; the scorers need them.
#   - CPU-bound infinite loops in pure Python: killed only by RLIMIT_CPU /
#     the wall-clock timeout, not instantaneously.
#   - Memory consumption up to RLIMIT_AS before the kill.
#   - Exfiltration via stdout (the parent only reads the __FUNSEARCH_RESULT__
#     line, but the child can write arbitrary data that is captured and discarded
#     by capture_output=True — no network path, so this is contained).
#   - On systems where RLIMIT_NPROC enforcement is per-UID and the UID already
#     has many processes, the fork block may not engage. The RLIMIT_NOFILE=3
#     guard remains effective independently.
#
# This replaces the previous socket monkey-patch, which was bypassable by
# importing the underlying C extension directly. The new approach enforces
# isolation at the OS syscall level via the Linux rlimit interface.
# --------------------------------------------------------------------------- #

# Stdlib modules pre-imported inside the child before fd lockdown.
# Anything listed here is usable by candidate programs after lockdown.
# Extend this list conservatively — every addition is a capability grant.
_SANDBOX_PREIMPORTS = (
    "resource", "os", "sys", "gc",
    "json", "math", "itertools", "random", "functools", "collections",
    "operator", "heapq", "bisect", "copy", "re", "string", "io",
    "struct", "array", "textwrap", "typing", "dataclasses", "abc",
    "hashlib", "enum", "fractions", "decimal", "statistics",
    # socket is pre-imported so the module object exists in sys.modules, but
    # socket() calls are still blocked because they allocate a new fd.
    "socket",
)

# The preamble is injected at the top of every child script.
# It runs BEFORE the candidate's code and BEFORE the scorer.
_SANDBOX_PREAMBLE = r'''
import resource as _r, os as _os, sys as _sys, gc as _gc
# ---- Phase 1: pre-import all stdlib modules the candidate / scorer may need.
# Must happen BEFORE fd lockdown because importing a module opens its .pyc file.
_PREIMPORTS = (
    "json", "math", "itertools", "random", "functools", "collections",
    "operator", "heapq", "bisect", "copy", "re", "string", "io",
    "struct", "array", "textwrap", "typing", "dataclasses", "abc",
    "hashlib", "enum", "fractions", "decimal", "statistics", "socket",
)
for _mod in _PREIMPORTS:
    try:
        __import__(_mod)
    except Exception:
        pass
del _PREIMPORTS
_gc.collect()

# ---- Phase 2: close every fd above 2 (stdin/stdout/stderr).
# This cleans up any stray fds (e.g. the dirfd opened by listdir itself).
try:
    _open_fds = [int(_f) for _f in _os.listdir("/proc/self/fd") if int(_f) > 2]
    for _fd in sorted(_open_fds, reverse=True):
        try:
            _os.close(_fd)
        except OSError:
            pass
    del _open_fds
except Exception:
    pass

# ---- Phase 3: set OS-level resource limits. All errors are silently ignored
# so the sandbox degrades gracefully on non-Linux or restricted environments.
try:
    _cpu = int(_os.environ.get("FUNSEARCH_CPU_SECONDS", "8"))
    _r.setrlimit(_r.RLIMIT_CPU, (_cpu, _cpu))
except Exception:
    pass
try:
    _mem = int(_os.environ.get("FUNSEARCH_MEM_BYTES", str(1024 * 1024 * 1024)))
    _r.setrlimit(_r.RLIMIT_AS, (_mem, _mem))
except Exception:
    pass
try:
    # RLIMIT_NPROC=0: no new processes may be forked (blocks subprocess, os.fork).
    _r.setrlimit(_r.RLIMIT_NPROC, (0, 0))
except Exception:
    pass
try:
    # RLIMIT_NOFILE=3: only fds 0/1/2 (stdin/stdout/stderr) remain usable.
    # Any syscall that allocates a new fd (socket, open, pipe, accept, ...) will
    # fail with EMFILE. This is the primary network + filesystem isolation layer.
    _r.setrlimit(_r.RLIMIT_NOFILE, (3, 3))
except Exception:
    pass
del _r, _gc
# _os and _sys intentionally left for candidate/scorer use (os.path, sys.exit, ...)
'''


def _runner_src(program_src: str, scorer_src: str,
                params: Dict[str, Any]) -> str:
    """Assemble the full program run in the sandbox: preamble + parameter
    injection + the AI's candidate + the problem's scorer + result emit.

    The scorer reads ``__N__``/``__CAP__``/... from globals (we inject ``params``)
    and sets ``__SCORE__`` / optional ``__BEHAVIOR__`` / ``__ERROR__``. We print a
    single JSON line for the parent to parse — nothing else goes to stdout from
    here (the candidate's own prints are swallowed by capturing separately)."""
    inject = "\n".join(f"{k} = {v!r}" for k, v in (params or {}).items())
    return (
        _SANDBOX_PREAMBLE
        + "\n# ---- injected parameters ----\n" + inject
        + "\n# ---- candidate program (AI-written) ----\n" + program_src
        + "\n# ---- problem scorer (mathlas) ----\n" + scorer_src
        + textwrap.dedent('''
        # ---- emit result ----
        import json as _json, math as _math
        _score = globals().get("__SCORE__", float("-inf"))
        try:
            _score = float(_score)
        except Exception:
            _score = float("-inf")
        _out = {
            "score": (None if _math.isinf(_score) and _score < 0 else _score),
            "raw_score": ("-inf" if (_math.isinf(_score) and _score < 0)
                          else _score),
            "behavior": list(globals().get("__BEHAVIOR__", []) or []),
            "error": globals().get("__ERROR__"),
        }
        import sys as _sys
        _sys.stdout.write("__FUNSEARCH_RESULT__" + _json.dumps(_out) + "\\n")
        ''')
    )


@dataclass(frozen=True)
class EvalResult:
    problem_id: str
    ok: bool                       # ran AND produced a finite score
    score: Optional[float]         # None when invalid / errored
    behavior: Tuple[float, ...]
    error: Optional[str]
    timed_out: bool
    seconds: float
    stdout_tail: str = ""


def evaluate(program_src: str, problem_id: str,
             timeout_s: float = 10.0,
             params: Optional[Dict[str, Any]] = None) -> EvalResult:
    """Run ``program_src`` against ``problem_id``'s scorer in a sandboxed
    subprocess and return its score (or the error). NO LLM. Deterministic.

    Containment (OS-enforced, not monkey-patched):
      - Process isolation: a fresh ``python -I -S`` child with a stripped
        environment (no credentials/tokens can leak).
      - RLIMIT_NOFILE=3: after Python initialises and all needed stdlib modules
        are pre-loaded, the fd table is capped at stdin/stdout/stderr only.
        Any syscall allocating a new fd — socket(), open(), pipe() — fails with
        EMFILE at the OS level. Network access and filesystem reads/writes are
        both blocked by this single limit.
      - RLIMIT_NPROC=0: fork() is blocked; subprocess, multiprocessing, and
        os.system() cannot spawn child processes.
      - RLIMIT_CPU: hard CPU-second budget; excess triggers SIGXCPU.
      - RLIMIT_AS: address-space cap against memory bombs.
      - Wall-clock timeout (``timeout_s``): primary guard for infinite loops.
      - Throwaway cwd in a temp dir that is deleted after the run.

    The candidate is AI-written and NOT trusted. See the module-level sandbox
    comment for a full list of what is and is not prevented.
    """
    prob = get_problem(problem_id)
    runner = _runner_src(program_src, prob.scorer_src, params or {})
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="funsearch_") as tmp:
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(runner)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "FUNSEARCH_CPU_SECONDS": str(int(max(1, timeout_s))),
            # keep the child light: no implicit thread fan-out
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", script],   # -I isolated, -S no site
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            raw = e.stdout or b""
            tail = (raw.decode("utf-8", "replace") if isinstance(raw, bytes)
                    else str(raw))[-500:]
            return EvalResult(problem_id=problem_id, ok=False, score=None,
                              behavior=(), error="timeout", timed_out=True,
                              seconds=time.time() - t0, stdout_tail=tail)
        dt = time.time() - t0
    # parse the single result line we emit; the candidate's own stdout is ignored.
    result_line = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("__FUNSEARCH_RESULT__"):
            result_line = line[len("__FUNSEARCH_RESULT__"):]
            break
    if result_line is None:
        # A negative returncode means the child was killed by a signal — most
        # often SIGXCPU (RLIMIT_CPU) or SIGKILL (RLIMIT_AS): that is the rlimit
        # guard firing, i.e. an effective timeout/resource-kill, not a clean error.
        killed = proc.returncode is not None and proc.returncode < 0
        if killed:
            sig = -proc.returncode
            err = f"killed by signal {sig} (CPU/memory rlimit — effective timeout)"
            return EvalResult(problem_id=problem_id, ok=False, score=None,
                              behavior=(), error=err, timed_out=True, seconds=dt,
                              stdout_tail=(proc.stdout or "")[-500:])
        err = (proc.stderr or "").strip()[-800:] or "no result emitted"
        return EvalResult(problem_id=problem_id, ok=False, score=None,
                          behavior=(), error=err, timed_out=False, seconds=dt,
                          stdout_tail=(proc.stdout or "")[-500:])
    try:
        data = json.loads(result_line)
    except json.JSONDecodeError:
        return EvalResult(problem_id=problem_id, ok=False, score=None,
                          behavior=(), error="malformed result", timed_out=False,
                          seconds=dt)
    score = data.get("score")
    behavior = tuple(data.get("behavior") or [])
    err = data.get("error")
    ok = score is not None and isinstance(score, (int, float)) and math.isfinite(score)
    return EvalResult(problem_id=problem_id, ok=ok,
                      score=(float(score) if ok else None),
                      behavior=behavior, error=err, timed_out=False, seconds=dt,
                      stdout_tail="")


# --------------------------------------------------------------------------- #
# The on-disk program DB: islands of MAP-Elites cells. We keep, per behaviour
# cell, the best-scoring program; plus the global best. Append-only writes are
# atomic (write tmp + os.replace) so concurrent agents don't corrupt it.
# --------------------------------------------------------------------------- #
def _load_db(problem_id: str) -> Dict[str, Any]:
    path = _db_path(problem_id)
    if not os.path.exists(path):
        return {"problem_id": problem_id, "cells": {}, "best": None,
                "n_registered": 0}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"problem_id": problem_id, "cells": {}, "best": None,
                "n_registered": 0}


def _save_db(problem_id: str, db: Dict[str, Any]) -> None:
    path = _db_path(problem_id)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp, path)


@dataclass(frozen=True)
class RegisterResult:
    problem_id: str
    accepted: bool                 # stored (new cell or beat the cell's best)
    new_global_best: bool
    score: float
    cell: str
    global_best_score: Optional[float]
    n_cells: int
    n_registered: int


def register(program_src: str, score: float, problem_id: str,
             behavior: Optional[Tuple] = None,
             meta: Optional[Dict[str, Any]] = None) -> RegisterResult:
    """Store a scored program in the on-disk MAP-Elites DB and report whether it
    is a new best (in its behaviour cell and/or globally). NO LLM.

    A program is ACCEPTED into a cell iff it beats that cell's current best (or
    the cell is empty) — the MAP-Elites rule, preserving one elite per behaviour
    niche so the AI can branch from diverse strong programs, not just the top one.
    """
    get_problem(problem_id)  # validate
    db = _load_db(problem_id)
    cell = ",".join(str(b) for b in (behavior or [])) or "_"
    cells: Dict[str, Any] = db.setdefault("cells", {})
    entry = {"program": program_src, "score": float(score),
             "behavior": list(behavior or []), "meta": meta or {},
             "ts": time.time()}
    prev = cells.get(cell)
    accepted = prev is None or float(score) > float(prev.get("score", -math.inf))
    if accepted:
        cells[cell] = entry
    # global best
    best = db.get("best")
    new_global = best is None or float(score) > float(best.get("score", -math.inf))
    if new_global:
        db["best"] = entry
    db["n_registered"] = int(db.get("n_registered", 0)) + 1
    _save_db(problem_id, db)
    return RegisterResult(
        problem_id=problem_id, accepted=accepted, new_global_best=new_global,
        score=float(score), cell=cell,
        global_best_score=(db["best"]["score"] if db.get("best") else None),
        n_cells=len(cells), n_registered=db["n_registered"])


def _few_shot(prob: Problem, top: List[Dict[str, Any]]) -> str:
    """Assemble FunSearch's best-shot prompt as DATA (NOT sent to any model here).
    Problem spec + the entry-point contract + the top program(s) so the AI can
    write a strictly-better variant."""
    lines = [
        f"PROBLEM: {prob.problem_id}",
        prob.description,
        f"Your program MUST define `{prob.entry_point}(...)` as in the examples.",
        ("Write a NEW version that scores HIGHER than the best below. Change the "
         "strategy, do not just reformat. Return ONLY the program source."),
        "",
    ]
    if not top:
        lines += ["# starter (seed) program — beat it:", prob.starter_src]
    else:
        for i, e in enumerate(top, 1):
            lines += [f"# program {i} — score {e['score']}:", e["program"], ""]
    return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class StatusResult:
    problem_id: str
    description: str
    entry_point: str
    best_score: Optional[float]
    best_program: Optional[str]
    n_cells: int
    n_registered: int
    elites: List[Dict[str, Any]]   # per-cell elites (program + score + cell)
    few_shot_context: str          # the next-variant prompt, as DATA
    starter_program: str


def status(problem_id: str, top_k: int = 3) -> StatusResult:
    """Return the current best program(s) + score + the FEW-SHOT CONTEXT the AI
    uses to write the next, better variant. NO LLM — the few-shot block is DATA
    the calling AI consumes; mathlas never sends it to a model."""
    prob = get_problem(problem_id)
    db = _load_db(problem_id)
    cells: Dict[str, Any] = db.get("cells", {})
    elites = sorted(cells.values(), key=lambda e: float(e.get("score", -math.inf)),
                    reverse=True)
    top = elites[:max(1, top_k)]
    best = db.get("best")
    few = _few_shot(prob, top)
    return StatusResult(
        problem_id=problem_id, description=prob.description,
        entry_point=prob.entry_point,
        best_score=(best["score"] if best else None),
        best_program=(best["program"] if best else None),
        n_cells=len(cells), n_registered=int(db.get("n_registered", 0)),
        elites=[{"cell": ",".join(map(str, e.get("behavior", []))) or "_",
                 "score": e.get("score"), "program": e.get("program")}
                for e in top],
        few_shot_context=few, starter_program=prob.starter_src)


__all__ = [
    "evaluate", "EvalResult", "register", "RegisterResult",
    "status", "StatusResult", "problems", "get_problem", "Problem",
    "db_dir",
]
