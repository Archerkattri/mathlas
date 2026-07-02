#!/usr/bin/env python3
"""Capture REAL mathlas tool outputs for the terminal demo GIF.

Runs the actual library in-process — the REAL Lean 4.31.0 kernel for
`verify_formal` and PSLQ + independent sympy re-eval for `identify_constant`.
No LLM, no mockups. Writes the verbatim JSON the MCP tools return to
``captured_outputs.json``; ``gen_terminal_gif.py`` renders the animation from
that file, so the GIF can only ever show what the tools actually returned.

Reproduce (from the repo root, Lean toolchain under reference/downloads/elan):
    PYTHONPATH=. python3 assets/gen/capture_outputs.py
"""
import json
import os
import sys

# Point verify_formal at the vendored Lean 4.31.0 toolchain if LEAN is unset.
_ELAN = os.path.join(os.getcwd(), "reference", "downloads", "elan",
                     "toolchains", "leanprover--lean4---v4.31.0", "bin", "lean")
if os.path.exists(_ELAN):
    os.environ.setdefault("LEAN", _ELAN)

import mpmath
from mathlas.server import tool_verify_formal, tool_identify_constant
from mathlas.verify_apply import find_lean

lean = find_lean()
if not lean:
    sys.exit("no Lean toolchain found — set LEAN or install elan; refusing to "
             "fake a verdict")
print(f"real Lean kernel: {lean}", file=sys.stderr)

# identify_constant input: zeta(2) to 50 significant digits (PSLQ needs >16).
with mpmath.workdps(60):
    zeta2 = mpmath.nstr(mpmath.zeta(2), 50)

calls = [
    {"tool": "verify_formal",
     "args": {"statement": "2 + 2 = 4", "proof": "rfl"},
     "result": tool_verify_formal(statement="2 + 2 = 4", proof="rfl")},
    {"tool": "verify_formal",
     "args": {"statement": "2 + 2 = 5", "proof": "rfl"},
     "result": tool_verify_formal(statement="2 + 2 = 5", proof="rfl")},
    {"tool": "verify_formal",
     "args": {"statement": "2 + 2 = 4", "proof": "by sorry"},
     "result": tool_verify_formal(statement="2 + 2 = 4", proof="by sorry")},
    {"tool": "identify_constant",
     "args": {"value": zeta2},
     "result": tool_identify_constant(zeta2)},
]

payload = {
    "provenance": {
        "lean": lean,
        "note": "captured in-process from the real Lean 4.31.0 kernel and "
                "PSLQ + independent sympy re-eval — no LLM inside",
    },
    "calls": calls,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "captured_outputs.json")
with open(out, "w") as fh:
    json.dump(payload, fh, indent=2, ensure_ascii=False)
print(f"wrote {out}", file=sys.stderr)
for c in calls:
    r = c["result"]
    verdict = r.get("proof_status") or ("identified" if r.get("identified")
                                        else "unidentified")
    print(f"  {c['tool']}({c['args']}) -> {verdict}", file=sys.stderr)
