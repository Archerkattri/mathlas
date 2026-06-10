#!/usr/bin/env python3
"""Agent-in-the-loop bench: the SAME model with vs without the mathlas MCP server.

This reconstructs the "live MCP server exercised end-to-end" acceptance test
(RESULTS.md §2b) as a headless, reproducible with/without comparison: a fixed
set of math tasks with deterministic ground truth is given to a coding agent
(`claude -p`, headless) twice —

  * WITH    — `--mcp-config <mathlas stdio config>` (server: `python -m
              mathlas.server`, NO API key, no LLM inside the server) and ONLY
              the mathlas tools allowed (no web, no bash) so the comparison
              isolates the tool's contribution;
  * WITHOUT — same model, same prompts, NO tools at all (pure parametric).

Scoring is deterministic string/number matching against known ground truth
(graded by this script, not by a model). Tasks deliberately include
hallucination traps (a constant with NO closed form; sqrt2+sqrt3 ~ pi;
e^pi - pi ~ 20) where the honest answer is "no/unidentified".

Run (server needs no API key; the DRIVING CLI is whatever `claude` is logged
in as — record the model with --model and in the output):

    PYTHONPATH=. python3 benchmarks/agent_bench.py --model claude-fable-5

Outputs a per-task table + totals, and writes raw transcripts under
logs/agent_bench/ for audit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (id, prompt, grader) -- grader: regex that must match (case-insens.) for credit
TASKS = [
    ("const_catalan",
     "Identify this constant exactly (closed form or standard name): "
     "0.915965594177219015054603514932384110774 . Answer concisely.",
     r"catalan"),
    ("const_sqrt23_trap",
     "Identify this constant exactly: 3.146264369941972342329135065715 . "
     "Is it pi? Answer concisely with the exact closed form.",
     r"sqrt\(?\s*2\s*\)?\s*\+\s*sqrt\(?\s*3|2\*?sqrt\(6\)\s*\+\s*5"
     r"|\\sqrt\{?2\}?\s*\+\s*\\sqrt\{?3|√\s*2\s*\+\s*√\s*3|√\(?5\s*\+\s*2√6"),
    ("const_no_form_trap",
     "Identify this constant exactly if a known closed form exists: "
     "1.637426929473257645941283898786 . If none is known, say UNIDENTIFIED. "
     "Answer concisely.",
     r"unidentified|no (known|verified) closed form"),
    ("const_zeta3",
     "Identify this constant exactly: 1.202056903159594285399738161511 . "
     "Answer concisely.",
     r"zeta\s*\(?\s*3|apery|apéry"),
    ("seq_motzkin",
     "Which OEIS sequence is 1, 1, 2, 4, 9, 21, 51, 127? Give the A-number. "
     "Answer concisely.",
     r"a001006|motzkin"),
    ("seq_bell",
     "Which OEIS sequence is 1, 1, 2, 5, 15, 52, 203, 877? Give the A-number. "
     "Answer concisely.",
     r"a000110|bell"),
    ("verify_epi_trap",
     "Is e^pi - pi exactly equal to 20? Verify numerically to high precision "
     "and answer YES or NO with the actual value. Answer concisely.",
     r"\bno\b.*19\.999|19\.999.*\bno\b|\bNO\b"),
    ("verify_zeta4",
     "Verify to at least 30 digits whether 1.082323233711138191516003696541 "
     "equals pi^4/90. Answer YES or NO concisely.",
     r"\byes\b"),
    ("search_bw",
     "Find the existing named theorem guaranteeing every bounded sequence in "
     "R^n has a convergent subsequence, and list its exact preconditions. "
     "Answer concisely.",
     r"bolzano",),
    ("lean_refute",
     "In Lean 4, does `rfl` prove `2 + 2 = 5`? If you can, check with a real "
     "Lean kernel and report the verdict. Answer concisely.",
     r"\bno\b|refuted|not a definitional|fails"),
]

MCP_CONFIG = {
    "mcpServers": {
        "mathlas": {
            "command": sys.executable,
            "args": ["-m", "mathlas.server"],
            "env": {"PYTHONPATH": _REPO, "MATHLAS_SEED": "1",
                    "OMP_NUM_THREADS": "2"},
        }
    }
}


def run_one(prompt: str, with_tool: bool, model: str, cfg_path: str,
            log_path: str, timeout: int = 600):
    cmd = ["claude", "-p", prompt, "--model", model,
           "--output-format", "stream-json", "--verbose",
           "--strict-mcp-config"]
    if with_tool:
        cmd += ["--mcp-config", cfg_path,
                "--allowedTools", "mcp__mathlas__*",
                "--disallowedTools", "Bash,WebSearch,WebFetch,Read,Write,Edit,Glob,Grep,Task"]
    else:
        cmd += ["--disallowedTools",
                "Bash,WebSearch,WebFetch,Read,Write,Edit,Glob,Grep,Task"]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                         cwd="/tmp")  # neutral cwd: no repo context leakage
    dt = time.time() - t0
    text, tool_calls = "", []
    for line in out.stdout.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "assistant":
            for blk in ev.get("message", {}).get("content", []):
                if blk.get("type") == "tool_use":
                    tool_calls.append(blk.get("name", "?"))
        if ev.get("type") == "result":
            text = ev.get("result", "") or ""
    with open(log_path, "w") as f:
        f.write(out.stdout + "\n--- STDERR ---\n" + out.stderr)
    return text, tool_calls, dt


def parse_transcript(log_path):
    """Re-parse a saved stream-json transcript (stdout section) -> (text, calls).

    Lets --reuse-logs re-grade every completed cell with the CURRENT grader
    instead of rerunning it (graders are deterministic regexes; the transcript
    is the ground truth)."""
    text, tool_calls = "", []
    with open(log_path) as f:
        for line in f:
            if line.startswith("--- STDERR ---"):
                break
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "assistant":
                for blk in ev.get("message", {}).get("content", []):
                    if blk.get("type") == "tool_use":
                        tool_calls.append(blk.get("name", "?"))
            if ev.get("type") == "result":
                text = ev.get("result", "") or ""
    return text, tool_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--logdir", default=os.path.join(_REPO, "logs", "agent_bench"))
    ap.add_argument("--tasks", default="",
                    help="comma-separated task ids to run (default: all)")
    ap.add_argument("--reuse-logs", action="store_true",
                    help="re-grade from saved transcripts when present; only "
                         "run cells whose transcript is missing")
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    cfg_path = os.path.join(args.logdir, "mcp_config.json")
    with open(cfg_path, "w") as f:
        json.dump(MCP_CONFIG, f, indent=2)

    only = {t.strip() for t in args.tasks.split(",") if t.strip()}
    rows = []
    for tid, prompt, pat in TASKS:
        if only and tid not in only:
            continue
        row = {"task": tid}
        for mode in ("with", "without"):
            log = os.path.join(args.logdir, f"{tid}.{mode}.jsonl")
            if args.reuse_logs and os.path.exists(log):
                text, calls = parse_transcript(log)
                dt = -1
                ok = bool(re.search(pat, text, re.IGNORECASE | re.DOTALL))
            else:
                try:
                    text, calls, dt = run_one(prompt, mode == "with", args.model,
                                              cfg_path, log)
                    ok = bool(re.search(pat, text, re.IGNORECASE | re.DOTALL))
                except Exception as e:  # timeout etc.
                    text, calls, dt, ok = f"<ERROR {e}>", [], -1, False
            row[mode] = ok
            row[f"{mode}_calls"] = len(calls)
            row[f"{mode}_tools"] = ",".join(sorted(set(calls)))
            row[f"{mode}_ans"] = re.sub(r"\s+", " ", text)[:160]
            print(f"[{tid:>18}] {mode:>7}: {'PASS' if ok else 'FAIL'} "
                  f"({len(calls)} tool calls, {dt:.0f}s) :: {row[f'{mode}_ans'][:90]}",
                  flush=True)
        rows.append(row)

    nw = sum(r["with"] for r in rows)
    no = sum(r["without"] for r in rows)
    print(f"\nWITH mathlas: {nw}/{len(rows)}    WITHOUT: {no}/{len(rows)}  "
          f"(model={args.model})")
    with open(os.path.join(args.logdir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "with": nw, "without": no,
                   "n": len(rows), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
