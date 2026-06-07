#!/usr/bin/env python3
"""Validation for the two remaining mathlas tools: FUNSEARCH harness + WEB-AUG.

FunSearch (mathlas = the sandboxed EVALUATOR, the AI is the generator):
  * correctness  -- the starter cap_set / bin_packing programs score as expected;
                    an INVALID program (collinear points) scores -inf.
  * containment  -- AI-written programs are NOT trusted; verify the sandbox blocks
                    network, kills an infinite loop (timeout), and caps memory.
  * MAP-Elites   -- register/status keep the best per behaviour cell + global best.

Web-augmentation (mathlas tells the AI WHAT to search, ingests what it brings back):
  * search_directive returns structured arXiv queries + named results + tool hints.
  * add_finding -> the finding is retrievable via BM25 with NO embedding-model load.

All runs use throwaway temp dirs (no pollution of the real DB / live corpus).
Run:  PYTHONPATH=. python3 benchmarks/tools_bench.py
"""
from __future__ import annotations

import os
import tempfile

# Throwaway DB / findings dirs BEFORE importing the tools (they read env at call time).
_TMP = tempfile.mkdtemp(prefix="mathlas_tools_bench_")
os.environ["MATHLAS_FUNSEARCH_DIR"] = os.path.join(_TMP, "funsearch")
os.environ["MATHLAS_FINDINGS"] = os.path.join(_TMP, "findings.jsonl")

from mathlas import funsearch as fs
from mathlas import webaug as wa


INVALID_CAPSET = ("def solve(n):\n"
                  "    # three collinear points (a+b+c == 0 mod 3) -> NOT a cap set\n"
                  "    return [tuple(0 for _ in range(n)), tuple(1 for _ in range(n)), "
                  "tuple(2 for _ in range(n))]\n")
MALICIOUS_NET = ("def solve(n):\n"
                 "    import socket\n"
                 "    socket.socket().connect(('1.1.1.1', 53))\n"
                 "    return [tuple(0 for _ in range(n))]\n")
MALICIOUS_LOOP = "def solve(n):\n    while True:\n        pass\n"
MALICIOUS_MEM = ("def solve(n):\n"
                 "    x = bytearray(10**10)  # 10 GB -> RLIMIT_AS should kill it\n"
                 "    return [tuple(0 for _ in range(n))]\n")


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def bench_funsearch():
    section("FUNSEARCH harness")
    passed = total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        print(f"  [{'OK  ' if cond else 'FAIL'}] {name}: {detail}")

    # --- correctness ---
    starter = fs.get_problem("cap_set").starter_src
    r = fs.evaluate(starter, "cap_set", params={"__N__": 4})
    check("cap_set starter valid", r.ok and r.score == 5.0,
          f"score={r.score} (expect 5 = zero+4 basis)")
    r2 = fs.evaluate(INVALID_CAPSET, "cap_set", params={"__N__": 4})
    check("invalid cap_set rejected", (not r2.ok) and r2.score is None,
          f"score={r2.score}, error={r2.error}")
    rb = fs.evaluate(fs.get_problem("online_bin_packing").starter_src, "online_bin_packing")
    check("bin_packing starter valid", rb.ok and rb.score is not None and rb.score < 0,
          f"score={rb.score} (= -avg bins, <0)")

    # --- containment (the AI's programs are untrusted) ---
    rn = fs.evaluate(MALICIOUS_NET, "cap_set", params={"__N__": 4})
    check("network blocked", (not rn.ok) and rn.error is not None,
          f"error={str(rn.error)[:60]}")
    rl = fs.evaluate(MALICIOUS_LOOP, "cap_set", timeout_s=3, params={"__N__": 4})
    check("infinite loop killed", (not rl.ok) and rl.timed_out,
          f"timed_out={rl.timed_out}, secs={rl.seconds:.1f}")
    rm = fs.evaluate(MALICIOUS_MEM, "cap_set", timeout_s=8, params={"__N__": 4})
    check("memory bomb contained", not rm.ok, f"error={str(rm.error)[:50]}")

    # --- MAP-Elites DB ---
    reg1 = fs.register(starter, 5.0, "cap_set", behavior=(5,))
    check("register new global best", reg1.accepted and reg1.new_global_best,
          f"global_best={reg1.global_best_score}")
    reg2 = fs.register("def solve(n): return []", 9.0, "cap_set", behavior=(1,))
    check("higher score is new best", reg2.new_global_best and reg2.global_best_score == 9.0,
          f"global_best={reg2.global_best_score}, cells={reg2.n_cells}")
    st = fs.status("cap_set")
    check("status returns few-shot context", st.best_score == 9.0 and "PROBLEM: cap_set" in st.few_shot_context,
          f"best={st.best_score}, elites={len(st.elites)}")
    print(f"  -> FunSearch {passed}/{total}")
    return passed, total


def bench_webaug():
    section("WEB-AUGMENTATION (search_directive + add_finding)")
    passed = total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        print(f"  [{'OK  ' if cond else 'FAIL'}] {name}: {detail}")

    d = wa.search_directive(
        "Find a closed form for the constant 1.6180339887498949 and the named theorem "
        "characterising contraction mappings on a complete metric space.")
    tool_names = [t["tool"] for t in d.mathlas_tools]
    check("directive: numeric tool suggested", "identify_constant" in tool_names,
          f"tools={tool_names}")
    check("directive: arxiv queries built", len(d.arxiv_queries) > 0,
          f"{len(d.arxiv_queries)} queries, e.g. {d.arxiv_queries[:1]}")
    check("directive: named results found", any("Banach" in n for n in d.named_results),
          f"named={d.named_results[:3]}")

    # add a finding with a UNIQUE marker term, then retrieve it (BM25, no model load).
    marker = "Zzqqxx"
    res = wa.add_finding(
        statement=f"The {marker} theorem: every widget admits a unique gizmo.",
        slogan=f"{marker} theorem on widgets and gizmos", source="arXiv:2606.00001")
    check("add_finding persisted (no model load)", res.ok and not res.dense_added,
          f"n_findings={res.n_findings}, dense_added={res.dense_added}")
    hits = wa.search_findings(marker, k=5)
    check("finding retrievable via BM25", len(hits) >= 1 and marker.lower() in
          (hits[0].get("statement", "").lower() if hits else ""),
          f"{len(hits)} hits, top source={hits[0]['source'] if hits else '-'}")
    print(f"  -> Web-aug {passed}/{total}")
    return passed, total


def main():
    print("=" * 70)
    print("mathlas tools validation — FunSearch harness + Web-augmentation")
    print("=" * 70)
    fp, ft = bench_funsearch()
    wp, wt = bench_webaug()
    print("\n" + "=" * 70)
    print(f"SUMMARY: FunSearch {fp}/{ft}   Web-aug {wp}/{wt}   "
          f"TOTAL {fp + wp}/{ft + wt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
