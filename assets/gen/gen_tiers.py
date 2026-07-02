#!/usr/bin/env python3
"""verification_tiers.png — the zero-false-positive scoreboard.

Every number verified against RESULTS.md §1 (verification/identification tiers),
§2 (applicability moat), §2b (discovery tools). Recovery is 100% on knowns and
false-positives are 0 across every measured tier.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import _style as S

S.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "verification_tiers.png")

# tier, tool, recovery, sub, mechanism, badge
tiles = [
    ("NUMERIC", "identify_constant", "8/8", "recovered @ known",
     "independent 50-digit re-eval", "0 / 3  false positives"),
    ("SEQUENCE", "identify_sequence", "8/8", "recovered  (7 top-1)",
     "exact OEIS term-match", "0 / 3  false positives"),
    ("RAMANUJAN", "conjecture_relation", "6/6", "relations re-verified",
     "PSLQ + CF, re-verified ≥25 digits", "0 / 2  false positives"),
    ("FORMAL", "verify_formal", "7/7", "correct kernel verdicts",
     "real Lean 4.31.0 kernel typecheck", "0  fake passes"),
    ("APPLICABILITY", "applicability_checklist", "15/15", "hypotheses surfaced",
     "atomic precondition decomposition", "6 / 6  traps caught"),
    ("DISCOVERY", "funsearch · add_finding", "14/14", "harness + web-aug checks",
     "sandbox: network / timeout / memory", "3 / 3  escapes contained"),
]

fig = plt.figure(figsize=(11.4, 6.7))
fig.patch.set_facecolor(S.SURFACE)

# --- title band ---
fig.text(0.035, 0.945, "Zero false positives across every tier",
         fontsize=19, fontweight="bold", color=S.NAVY,
         fontfamily="DejaVu Sans")
fig.text(0.035, 0.895,
         "airtight-or-nothing — every result is independently checkable, "
         "never a plausible guess",
         fontsize=10.5, color=S.INK_2, fontfamily="DejaVu Sans")
# headline stat pill (top-right)
fig.text(0.965, 0.925,
         "0 / 8  false hits on structureless inputs\n"
         "100% recovery on knowns  ·  RESULTS.md §1–2b",
         fontsize=9.3, color=S.GOOD_DK, ha="right", va="center",
         fontfamily="DejaVu Sans Mono", linespacing=1.6,
         bbox=dict(boxstyle="round,pad=0.6", fc="#e9f6e9", ec=S.GOOD, lw=1.1))

# --- grid of cards ---
ncol, nrow = 3, 2
x0, x1, ytop, ybot = 0.028, 0.972, 0.815, 0.045
gx, gy = 0.022, 0.055
cw = (x1 - x0 - (ncol - 1) * gx) / ncol
ch = (ytop - ybot - (nrow - 1) * gy) / nrow


def card(ax_fig, cx, cy, tier, tool, rec, sub, mech, badge):
    box = FancyBboxPatch((cx, cy), cw, ch,
                         boxstyle="round,pad=0.004,rounding_size=0.014",
                         mutation_aspect=cw / ch,
                         transform=fig.transFigure, fc="#ffffff",
                         ec=S.GRID, lw=1.2, zorder=1)
    fig.patches.append(box)
    # blue accent rail on the left edge
    rail = FancyBboxPatch((cx, cy), 0.006, ch,
                          boxstyle="square,pad=0", transform=fig.transFigure,
                          fc=S.BLUE, ec="none", zorder=2)
    fig.patches.append(rail)
    pad = 0.020
    lx = cx + pad
    # tier label + tool
    fig.text(lx, cy + ch - 0.050, tier, fontsize=11.5, fontweight="bold",
             color=S.NAVY, fontfamily="DejaVu Sans")
    fig.text(lx, cy + ch - 0.088, tool, fontsize=9.2, color=S.MUTED,
             fontfamily="DejaVu Sans Mono")
    # big recovery number, descriptor stacked beneath (no side collision)
    fig.text(lx, cy + ch - 0.175, rec, fontsize=31, fontweight="bold",
             color=S.BLUE_DK, fontfamily="DejaVu Sans Mono", va="center")
    fig.text(lx + 0.004, cy + ch - 0.222, sub, fontsize=9.0,
             color=S.INK_2, fontfamily="DejaVu Sans", va="center")
    # 100% meter
    my = cy + ch * 0.245
    mw = cw - 2 * pad
    fig.patches.append(FancyBboxPatch((lx, my), mw, 0.012,
                       boxstyle="round,pad=0,rounding_size=0.006",
                       transform=fig.transFigure, fc=S.GRID, ec="none",
                       zorder=2))
    fig.patches.append(FancyBboxPatch((lx, my), mw, 0.012,
                       boxstyle="round,pad=0,rounding_size=0.006",
                       transform=fig.transFigure, fc=S.BLUE, ec="none",
                       zorder=3))
    fig.text(cx + cw - pad, my + 0.028, "100%", fontsize=8.2, color=S.BLUE_DK,
             ha="right", fontfamily="DejaVu Sans Mono", fontweight="bold")
    fig.text(lx, my + 0.028, mech, fontsize=7.9, color=S.MUTED,
             fontfamily="DejaVu Sans", style="italic")
    # green badge
    fig.text(lx, cy + 0.052, "✔  " + badge, fontsize=9.0, color=S.GOOD_DK,
             fontfamily="DejaVu Sans", fontweight="bold", va="center",
             bbox=dict(boxstyle="round,pad=0.42", fc="#e9f6e9", ec=S.GOOD,
                       lw=1.0))


for i, t in enumerate(tiles):
    r, c = divmod(i, ncol)
    cx = x0 + c * (cw + gx)
    cy = ytop - ch - r * (ch + gy)
    card(fig, cx, cy, *t)

fig.savefig(OUT, facecolor=S.SURFACE)
print("wrote", os.path.normpath(OUT))
