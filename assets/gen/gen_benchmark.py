#!/usr/bin/env python3
"""benchmark_hit20.png — TheoremSearch 110-query benchmark, theorem Hit@20.

Numbers verbatim from the committed table in README.md / RESULTS.md §3c
(re-measured 2026-06-10 on the served 3,683,428-doc index). No mockups.
"""
import os
import matplotlib.pyplot as plt
import _style as S

S.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "benchmark_hit20.png")

# (label, value, kind)  kind: 'hero' | 'floor' | 'ext' | 'paper'
# theorem Hit@20 unless flagged paper-level (Google reports paper only).
rows = [
    ("mathlas + self-augmenting web loop", 59.1, "hero"),
    ("TheoremSearch  (Qwen3-8B, private 9.2M)", 45.0, "ext"),
    ("Google  (site:arxiv.org) *", 37.8, "paper"),
    ("Gemini 3 Pro", 27.0, "ext"),
    ("ChatGPT 5.2  w/ Search", 19.8, "ext"),
    ("mathlas — baseline (corpus-only)", 10.0, "floor"),
]
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
kinds = [r[2] for r in rows]
y = list(range(len(rows)))[::-1]  # first row at top

fig, ax = plt.subplots(figsize=(9.6, 5.4))
fig.subplots_adjust(left=0.375, right=0.965, top=0.78, bottom=0.27)

for yi, val, kind in zip(y, vals, kinds):
    fc = S.BLUE if kind == "hero" else S.GREY
    hatch = "////" if kind == "paper" else None
    ax.barh(yi, val, height=0.62, color=fc, hatch=hatch,
            edgecolor=S.SURFACE, linewidth=1.8, zorder=3)
    tcol = S.NAVY if kind == "hero" else S.INK
    tw = "bold" if kind == "hero" else "normal"
    ax.text(val + 1.0, yi, f"{val:.1f}", va="center", ha="left",
            fontsize=12.5, color=tcol, fontweight=tw,
            fontfamily="DejaVu Sans Mono")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10.5)
# colour the two mathlas tick labels blue (identity without a 2nd fill colour)
for lab, tick in zip(labels[::-1], ax.get_yticklabels()):
    if lab.startswith("mathlas"):
        tick.set_color(S.BLUE_DK)
        tick.set_fontweight("bold")

ax.set_xlim(0, 66)
ax.set_xticks(range(0, 61, 10))
ax.set_xlabel("theorem Hit@20  (%)   —   higher is better", fontsize=10,
              color=S.INK_2, labelpad=6)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(S.BASELINE)
ax.xaxis.grid(True, color=S.GRID, lw=0.7, zorder=0)
ax.set_axisbelow(True)

# clean callout in the empty lower-right space (auto-sized text bbox)
ax.text(48.0, 1.6,
        "The self-augmenting web loop:\n"
        "10.0 → 59.1   (+49.1 pts)\n"
        "the AI web-finds each missing\n"
        "statement; add_finding() fuses it\n"
        "through the dense channel at runtime",
        fontsize=8.4, color=S.BLUE_DK, va="center", ha="center",
        linespacing=1.45, zorder=6,
        bbox=dict(boxstyle="round,pad=0.6", fc="#eef3fb", ec=S.BLUE, lw=1.1))

fig.text(0.02, 0.945, "Beating the prior open SOTA on its own benchmark",
         fontsize=15, fontweight="bold", color=S.NAVY, fontfamily="DejaVu Sans")
fig.text(0.02, 0.885,
         "TheoremSearch's 110 human-written queries  ·  theorem Hit@20  ·  "
         "mathlas served from the 3.68M-doc index",
         fontsize=9.6, color=S.INK_2, fontfamily="DejaVu Sans")

cap = ("Honest caveat — this is the LOOP's value, not a native-corpus claim.  The 10.0% floor is licensing-bounded: TheoremSearch "
       "withheld ~85% of their 9.2M\ncorpus, so 95/110 target papers are unreachable for any open system; the loop repairs that at AI-runtime.   "
       "* Google is paper-level Hit@20 (no\ntheorem number reported); every other bar is theorem Hit@20.")
fig.text(0.02, 0.015, cap, fontsize=7.7, color=S.MUTED, va="bottom",
         fontfamily="DejaVu Sans", linespacing=1.5)

fig.savefig(OUT)
print("wrote", os.path.normpath(OUT))
