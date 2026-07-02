"""Shared mathlas visual-pack palette + matplotlib defaults.

Brand colours sampled from assets/banner.png (navy #071a4f, cyan #04b2d1) on the
near-white dataviz reference surface (#fcfcfb). Colour roles follow the dataviz
skill: a single validated blue series, neutral grey for comparators, reserved
status-green for the zero-false-positive badges, cyan only as a labelled accent.
"""
import matplotlib as mpl

# --- surfaces & ink -------------------------------------------------------
SURFACE = "#fcfcfb"   # chart surface (matches banner background)
PAGE = "#f7f7f4"
INK = "#0b0b0b"       # primary ink
NAVY = "#071a4f"      # brand navy (headings / hero text)
INK_2 = "#52514e"     # secondary ink
MUTED = "#898781"     # axis / muted labels
GRID = "#e1e0d9"      # hairline gridline
BASELINE = "#c3c2b7"  # axis line

# --- data marks -----------------------------------------------------------
BLUE = "#1c5cab"      # the mathlas series (validated: in band, >=4.5:1 on light)
BLUE_DK = "#12408f"
GREY = "#a8a7a1"      # neutral comparator fill
GREY_DK = "#7d7c76"
CYAN = "#04b2d1"      # brand accent — labelled use only
GOOD = "#0ca30c"      # status: good / zero-FP (icon + label, never colour alone)
GOOD_DK = "#006300"

# --- terminal (dark) ------------------------------------------------------
TERM_BG = "#12151c"
TERM_PANEL = "#181c25"
TERM_INK = "#e6e6e0"
TERM_MUTED = "#8b93a3"
TERM_BLUE = "#5aa2f0"
TERM_CYAN = "#3ad0ea"
TERM_GREEN = "#4ec97a"
TERM_RED = "#f0736b"
TERM_AMBER = "#e8b64c"

MONO = ["DejaVu Sans Mono", "monospace"]
SANS = ["DejaVu Sans", "sans-serif"]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "svg.fonttype": "path",
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })
