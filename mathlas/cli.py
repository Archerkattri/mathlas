"""``mathlas`` command-line entry point.

Two modes, auto-detected from the argument (or forced with a flag):

  NUMERIC  ``mathlas 1.6449340668482264``   -> the airtight constant->closed-form
           path (engine.identify). No LLM, no network. The default when the
           argument parses as a number.

  PROBLEM  ``mathlas "smooth variety over a separably closed field has a k-point"``
           -> the retrieve -> map -> verify -> provenance loop (solve.solve) over
           a corpus you point it at (``--corpus DIR``, the open dataset). Needs an
           LLM for MAP/VERIFY; without one it runs retrieval-only.

Examples
--------
  mathlas 1.6449340668482264
  mathlas --basis pi,e,catalan 0.915965594177219
  mathlas "filtered colimit commutes with an adequate functor" \\
             --corpus reference/theorem-search-dataset --limit 5000 --no-verify
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _run_numeric(value: str, args) -> int:
    import mpmath
    from .engine import identify
    from .identify import DEFAULT_BASIS
    basis = tuple(args.basis.split(",")) if args.basis else DEFAULT_BASIS
    mpmath.mp.dps = max(args.dps_verify + 10, 60)
    v = mpmath.mpf(value)  # str -> full precision if more digits were given
    res = identify(v, dps_search=args.dps_search, dps_verify=args.dps_verify,
                   min_digits=args.min_digits, basis=basis)
    print(res)
    if args.verbose and res.candidates:
        for c in res.candidates:
            print(f"    candidate {c.expr}: verified {c.verify.digits_agreed} digits "
                  f"[{c.provenance.novelty.value}]")
    return 0 if res.identified else 1


def _build_retriever(args):
    from .retrieve.corpus import load_documents
    from .retrieve.hybrid import HybridRetriever
    embedder = None
    if args.embedder == "qwen3":
        from .embed import Qwen3Embedder
        embedder = Qwen3Embedder(dim=args.embed_dim or None)
    docs = load_documents(args.corpus, limit=args.limit)
    if args.verbose:
        print(f"# corpus: {len(docs)} documents from {args.corpus}", file=sys.stderr)
    return HybridRetriever(docs, embedder=embedder)


def _make_llm(args):
    if args.llm == "anthropic":
        from .llm import AnthropicLLM
        return AnthropicLLM(model=args.model) if args.model else AnthropicLLM()
    if args.llm == "echo":
        from .llm import EchoLLM
        return EchoLLM()
    return None  # retrieval-only


def _run_problem(problem: str, args) -> int:
    retriever = _build_retriever(args)
    llm = _make_llm(args)

    if llm is None:
        # Retrieval-only: no LLM available -> just show candidate existing results.
        print(f"# retrieval-only (no LLM); top {args.k} candidate results:")
        for c in retriever.retrieve(problem, k=args.k):
            print(f"  [{c.score:.4f}] {c.name or '(unnamed)'}  -- {c.source or ''}")
            print(f"      {(c.meta or {}).get('slogan') or c.statement[:160]}")
        return 0

    from .solve import solve
    sol = solve(problem, retriever, llm, k=args.k,
                min_confidence=args.min_confidence,
                verify=not args.no_verify, verify_passes=args.verify_passes)
    print(sol)
    return 0 if sol.found else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mathlas",
        description="Map a problem to the EXISTING math that solves it, and verify it.")
    p.add_argument("query", help="a number (numeric mode) or a problem description")
    p.add_argument("--mode", choices=["auto", "numeric", "problem"], default="auto")
    p.add_argument("-v", "--verbose", action="store_true")

    g = p.add_argument_group("numeric mode")
    g.add_argument("--basis", help="comma-separated constant basis (e.g. pi,e,catalan)")
    g.add_argument("--dps-search", type=int, default=30)
    g.add_argument("--dps-verify", type=int, default=50)
    g.add_argument("--min-digits", type=int, default=20)

    g = p.add_argument_group("problem mode")
    g.add_argument("--corpus", help="dir with the open theorem dataset parquets")
    g.add_argument("--limit", type=int, default=5000,
                   help="cap corpus size (keep small without a GPU)")
    g.add_argument("--k", type=int, default=10, help="candidates to retrieve")
    g.add_argument("--embedder", choices=["hashing", "qwen3"], default="hashing")
    g.add_argument("--embed-dim", type=int, default=0, help="dense dim (0=model default)")
    g.add_argument("--llm", choices=["none", "anthropic", "echo"], default="none")
    g.add_argument("--model", help="LLM model id (for --llm anthropic)")
    g.add_argument("--min-confidence", type=float, default=0.5)
    g.add_argument("--no-verify", action="store_true",
                   help="skip the adversarial applicability verification")
    g.add_argument("--verify-passes", type=int, default=1,
                   help="independent skeptic passes (worst verdict wins)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    mode = args.mode
    if mode == "auto":
        mode = "numeric" if _looks_numeric(args.query) else "problem"
    if mode == "numeric":
        return _run_numeric(args.query, args)
    if not args.corpus:
        print("error: problem mode needs --corpus DIR (the open theorem dataset)",
              file=sys.stderr)
        return 2
    return _run_problem(args.query, args)


if __name__ == "__main__":
    raise SystemExit(main())
