#!/usr/bin/env python3
"""architecture.png / .svg — how mathlas plugs into any MCP agent.

MCP client (any agent) -> 12 pure data-returning tools grouped into
RETRIEVE / VERIFY / DISCOVER -> data returned. The AI is the brain; mathlas is
the hands. No LLM inside. Text-first, restrained, mono for tool names.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import _style as S

S.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, os.pardir, "architecture.png")
OUT_SVG = os.path.join(HERE, os.pardir, "architecture.svg")

fig = plt.figure(figsize=(12.6, 7.0))
fig.patch.set_facecolor(S.SURFACE)
T = fig.transFigure


def rbox(x, y, w, h, fc, ec, lw=1.2, z=1, rounding=0.02):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.002,rounding_size={rounding}",
        mutation_aspect=w / h, transform=T, fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, size, color, weight="normal", family="DejaVu Sans",
        ha="left", va="baseline", style="normal", z=5):
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight,
             fontfamily=family, ha=ha, va=va, style=style, zorder=z)


# --- title ---
txt(0.030, 0.945, "The AI is the brain; mathlas is the hands", 18, S.NAVY,
    "bold")
txt(0.030, 0.905, "any MCP client calls 12 pure data-returning tools — "
    "no LLM inside, no API key", 10.5, S.INK_2)

# --- agent (left) ---
ax0, ay0, aw, ah = 0.022, 0.315, 0.160, 0.42
rbox(ax0, ay0, aw, ah, "#eef3fb", S.BLUE, 1.6, z=2, rounding=0.035)
acx = ax0 + aw / 2
txt(acx, ay0 + ah - 0.075, "MCP client", 14, S.NAVY, "bold", ha="center")
txt(acx, ay0 + ah - 0.120, "(any agent)", 11, S.BLUE_DK, ha="center")
txt(acx, ay0 + ah * 0.46, "Claude Code", 9.4, S.INK_2,
    family="DejaVu Sans Mono", ha="center")
txt(acx, ay0 + ah * 0.46 - 0.043, "Cursor · any client", 9.4, S.INK_2,
    family="DejaVu Sans Mono", ha="center")
txt(acx, ay0 + 0.050, "the brain — reasons & judges", 8.6, S.MUTED,
    ha="center", style="italic")

# --- mathlas container (right) ---
mx, my, mw, mh = 0.300, 0.130, 0.685, 0.740
rbox(mx, my, mw, mh, "#ffffff", S.BASELINE, 1.4, z=1, rounding=0.02)
txt(mx + 0.018, my + mh - 0.052, "mathlas", 14, S.NAVY, "bold")
txt(mx + 0.145, my + mh - 0.052, "MCP server · 12 tools · every result "
    "provenance-labeled · no LLM inside", 9.0, S.MUTED)

# --- arrows between agent and container (wide, clear gap) ---
gap_mid = (ax0 + aw + mx) / 2
arr_call = FancyArrowPatch((ax0 + aw + 0.004, 0.585), (mx - 0.004, 0.585),
                           arrowstyle="-|>", mutation_scale=15, lw=2.0,
                           color=S.BLUE, transform=T, zorder=6)
fig.patches.append(arr_call)
txt(gap_mid, 0.610, "tool call", 8.6, S.BLUE_DK, "bold", ha="center",
    family="DejaVu Sans Mono")

arr_ret = FancyArrowPatch((mx - 0.004, 0.455), (ax0 + aw + 0.004, 0.455),
                          arrowstyle="-|>", mutation_scale=15, lw=2.0,
                          color=S.GOOD_DK, transform=T, zorder=6)
fig.patches.append(arr_ret)
txt(gap_mid, 0.480, "data back", 8.6, S.GOOD_DK, "bold", ha="center",
    family="DejaVu Sans Mono")
txt(gap_mid, 0.418, "candidates\nverdicts\nchecklists\nscaffolds", 7.4,
    S.MUTED, ha="center", va="top", family="DejaVu Sans Mono")

# --- three group columns inside the container ---
groups = [
    ("RETRIEVE", "find the real math", [
        ("search_existing_math", "hybrid dense + BM25 → RRF → rerank"),
        ("search_formal_math", "Loogle + LeanSearch (proxied)"),
    ]),
    ("VERIFY", "check it, airtight", [
        ("verify_numeric", "PSLQ + independent sympy re-eval"),
        ("identify_constant", "closed form, 50-digit re-eval"),
        ("identify_sequence", "exact OEIS term-match"),
        ("verify_formal", "real Lean kernel typecheck / proof"),
    ]),
    ("DISCOVER", "scaffold & grow", [
        ("conjecture_relation", "Ramanujan PSLQ + CF"),
        ("applicability_checklist", "atomic preconditions"),
        ("mapping_scaffold", "needs ↔ guarantees"),
        ("funsearch", "sandboxed program search"),
        ("search_directive", "web-search plan"),
        ("add_finding", "self-augmenting loop ↺"),
    ]),
]
inner_l, inner_r = mx + 0.016, mx + mw - 0.016
gap = 0.016
colw = (inner_r - inner_l - 2 * gap) / 3
col_top, col_bot = my + mh - 0.095, my + 0.028   # column-box extents
PITCH = 0.088                                     # fixed row pitch, top-aligned

for gi, (name, tag, tools) in enumerate(groups):
    cx0 = inner_l + gi * (colw + gap)
    rbox(cx0, col_bot, colw, col_top - col_bot, "#f7f9fc", S.GRID, 1.0,
         z=2, rounding=0.035)
    txt(cx0 + 0.012, col_top - 0.035, name, 11.5, S.BLUE_DK, "bold")
    txt(cx0 + 0.012, col_top - 0.060, tag, 8.2, S.MUTED, style="italic")
    fig.patches.append(FancyBboxPatch(
        (cx0 + 0.012, col_top - 0.073), colw - 0.024, 0.004,
        boxstyle="square,pad=0", transform=T, fc=S.BLUE, ec="none", zorder=3))
    rtop = col_top - 0.105
    for ti, (tool, mech) in enumerate(tools):
        ry = rtop - ti * PITCH
        txt(cx0 + 0.014, ry, tool, 9.6, S.NAVY, "bold",
            family="DejaVu Sans Mono")
        txt(cx0 + 0.014, ry - 0.027, mech, 7.9, S.INK_2, style="italic")

# --- footer flow line ---
txt(0.030, 0.052, "typical flow:", 9.0, S.MUTED, "bold",
    family="DejaVu Sans Mono")
txt(0.132, 0.052,
    "search_existing_math → mapping_scaffold + applicability_checklist → "
    "( the AI judges ) → verify_numeric / verify_formal",
    8.8, S.INK_2, family="DejaVu Sans Mono")

fig.savefig(OUT_PNG, facecolor=S.SURFACE)
fig.savefig(OUT_SVG, facecolor=S.SURFACE)
print("wrote", os.path.normpath(OUT_PNG), "and", os.path.normpath(OUT_SVG))
