"""``mathlas`` command-line entry point — NO LLM, NO API key.

mathlas is a tool an AI *uses*; the CLI is a thin, human-facing view of the same
no-LLM capabilities the AI gets over MCP. Modes:

  NUMERIC  ``mathlas 1.6449340668482264``   -> the airtight constant->closed-form
           path (engine.identify). Airtight, no LLM, no network. The default when
           the argument parses as a number.

  PROBLEM  ``mathlas "Banach contraction gives a unique fixed point"``
           -> search EXISTING math (our own index) and, for the top candidate,
           print the needs<->guarantees SCAFFOLD + applicability CHECKLIST as
           data — the structure an AI (or you) reasons over. NO LLM is called.
           Uses a small built-in seed corpus by default; point ``--corpus DIR`` at
           the open theorem dataset for the real index.

  MCP      ``mathlas mcp``  (or ``python -m mathlas.server``)
           -> run the MCP server so an AI client can call the tools. Register in
           Claude Code with:  ``claude mcp add mathlas -- python -m mathlas.server``

Examples
--------
  mathlas 1.6449340668482264364724151666460251892
  mathlas --basis pi,e,catalan 0.915965594177219
  mathlas "a bounded sequence has a convergent subsequence" --k 5
  mathlas "<problem>" --corpus reference/theorem-search-dataset --limit 5000
  mathlas mcp
"""
from __future__ import annotations

import argparse
import json
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


def _run_problem(problem: str, args) -> int:
    """AI-driven, NO-LLM path: search existing math, then print the scaffold +
    checklist the calling AI reasons over. mathlas itself never judges."""
    from .server import (tool_search_existing_math, tool_mapping_scaffold,
                         tool_applicability_checklist)
    search = tool_search_existing_math(problem, k=args.k, corpus_dir=args.corpus,
                                       corpus_limit=args.limit)
    if args.json:
        out = {"search": search}
        if search["candidates"]:
            top = search["candidates"][0]["statement"]
            out["mapping_scaffold"] = tool_mapping_scaffold(problem, top)
            out["applicability_checklist"] = tool_applicability_checklist(top)
        print(json.dumps(out, indent=2, default=str))
        return 0 if search["candidates"] else 1

    print(f"# search ({search['corpus']}) — top {args.k} candidate existing results:")
    for c in search["candidates"]:
        score = c["score"]
        print(f"  [{score:.4f}] {c['name'] or '(unnamed)'}  -- {c['source'] or ''}")
        print(f"      {(c['slogan'] or c['statement'])[:160]}")
    if not search["candidates"]:
        print("  (no candidates; try --corpus DIR for a larger index)")
        return 1

    top = search["candidates"][0]
    print(f"\n# needs<->guarantees SCAFFOLD for the top candidate "
          f"({top['name'] or 'unnamed'}) — the AI reasons over this (NO LLM):")
    sc = tool_mapping_scaffold(problem, top["statement"])
    for q in sc["questions"]:
        print(f"  - {q}")
    print("\n# applicability CHECKLIST (mark each against your problem):")
    cl = tool_applicability_checklist(top["statement"])
    for cond in cl["preconditions"]:
        print(f"  [ ] {cond['text']}")
    print(f"  => guarantees: {cl['conclusion']}")
    print("\n(mathlas provided search + scaffold + checklist; an AI does the "
          "judging. Run `mathlas mcp` to expose these as MCP tools.)")
    return 0


def _run_mcp(_args) -> int:
    from .server import main as server_main
    return server_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mathlas",
        description="A tool FOR an AI: search existing math + airtight "
                    "verification + needs<->guarantees scaffolds. No API key.")
    p.add_argument("query", nargs="?",
                   help="a number (numeric mode), a problem description (problem "
                        "mode), or 'mcp' to run the MCP server")
    p.add_argument("--mode", choices=["auto", "numeric", "problem", "mcp"],
                   default="auto")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON (problem mode)")

    g = p.add_argument_group("numeric mode")
    g.add_argument("--basis", help="comma-separated constant basis (e.g. pi,e,catalan)")
    g.add_argument("--dps-search", type=int, default=30)
    g.add_argument("--dps-verify", type=int, default=50)
    g.add_argument("--min-digits", type=int, default=20)

    g = p.add_argument_group("problem mode")
    g.add_argument("--corpus", help="dir with the open theorem dataset parquets "
                                    "(omit to use the built-in seed corpus)")
    g.add_argument("--limit", type=int, default=5000,
                   help="cap corpus size (keep small without a GPU)")
    g.add_argument("--k", type=int, default=10, help="candidates to retrieve")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    query = args.query
    mode = args.mode
    if mode == "auto":
        if query is None:
            print("error: provide a number, a problem description, or 'mcp'",
                  file=sys.stderr)
            return 2
        if query == "mcp":
            mode = "mcp"
        else:
            mode = "numeric" if _looks_numeric(query) else "problem"
    if mode == "mcp":
        return _run_mcp(args)
    if query is None:
        print("error: a query is required for this mode", file=sys.stderr)
        return 2
    if mode == "numeric":
        return _run_numeric(query, args)
    return _run_problem(query, args)


if __name__ == "__main__":
    raise SystemExit(main())
