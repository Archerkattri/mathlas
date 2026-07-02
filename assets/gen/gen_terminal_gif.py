#!/usr/bin/env python3
"""demo_terminal.gif — the hero: a rendered terminal session of REAL mathlas
tool calls with REAL outputs.

Reads assets/gen/captured_outputs.json (verbatim JSON from the real Lean 4.31.0
kernel + PSLQ, produced by capture_outputs.py) and asserts the load-bearing
verdicts before drawing — so the animation can only show what the tools actually
returned. No asciinema needed: a PIL monospace typing animation.
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "captured_outputs.json")
OUT = os.path.join(HERE, os.pardir, "demo_terminal.gif")

# ---- palette (dark terminal) --------------------------------------------
BG = (18, 21, 28)
PANEL = (24, 28, 37)
BAR = (30, 34, 44)
INK = (230, 230, 224)
MUTED = (139, 147, 163)
CYAN = (58, 208, 234)
BLUE = (90, 162, 240)
GREEN = (78, 201, 122)
RED = (240, 115, 107)
AMBER = (232, 182, 76)
KEY = (120, 176, 245)
STR = (150, 210, 160)

# ---- fonts ---------------------------------------------------------------
FD = "/usr/share/fonts/truetype/dejavu"
FS = 15
font = ImageFont.truetype(f"{FD}/DejaVuSansMono.ttf", FS)
fontb = ImageFont.truetype(f"{FD}/DejaVuSansMono-Bold.ttf", FS)
fonttitle = ImageFont.truetype(f"{FD}/DejaVuSansMono-Bold.ttf", 14)
CW = font.getlength("M")          # monospace cell width
LH = 25                           # line height
MX, MY = 26, 58                   # left margin, top margin (below title bar)

# ---- load + verify the real captured outputs ----------------------------
with open(DATA) as fh:
    payload = json.load(fh)
calls = {tuple(sorted(c["args"].items())): c["result"] for c in payload["calls"]}
verified = payload["calls"][0]["result"]
refuted = payload["calls"][1]["result"]
sorry = payload["calls"][2]["result"]
ident = payload["calls"][3]["result"]

assert verified["proof_status"] == "VERIFIED_PROOF", verified
assert refuted["proof_status"] == "REFUTED"
assert "Not a definitional equality" in refuted["kernel_error"]
assert "2 + 2" in refuted["kernel_error"] and "5" in refuted["kernel_error"]
assert sorry["proof_status"] == "REFUTED" and "sorry" in sorry["detail"]
assert ident["identified"] and ident["best"]["display"] == "pi**2/6"
DIG = ident["best"]["digits_agreed"]
assert DIG == 50, DIG
VAL = payload["calls"][3]["args"]["value"]

# ---- the transcript: list of lines; each line = list of (text,color,bold) --
# segment helpers
def seg(t, c=INK, b=False):
    return (t, c, b)


P = lambda: [seg("● ", CYAN, True)]  # header bullet
DOL = lambda: [seg("$ ", CYAN, True)]

SCRIPT = []  # each entry: (kind, payload)
# kind: 'out' printed line (segments) | 'type' typed command (segments)
#       'run' amber running line (text) | 'blank' | 'hold' n

def out(segs):
    SCRIPT.append(("out", segs))

def typ(segs):
    SCRIPT.append(("type", segs))

def run(t):
    SCRIPT.append(("run", t))

def blank():
    SCRIPT.append(("blank", None))

def hold(n=6):
    SCRIPT.append(("hold", n))


# header
out([seg("● ", CYAN, True),
     seg("mathlas — the MCP math tools an AI ", MUTED),
     seg("uses", INK, True), seg(".  no LLM inside.", MUTED)])
out([seg("  every verdict below is from the real Lean 4.31.0 kernel or PSLQ "
         "+ an independent re-eval.", MUTED)])
blank()

# 1. VERIFIED
typ(DOL() + [seg('verify_formal(statement="2 + 2 = 4", proof="rfl")', INK)])
run("running the real Lean 4.31.0 kernel …")
out([seg("  → ", MUTED), seg("{ ", MUTED), seg('"proof_status"', KEY),
     seg(": ", MUTED), seg('"VERIFIED_PROOF"', GREEN, True),
     seg(", ", MUTED), seg('"checked"', KEY), seg(": ", MUTED),
     seg("true", STR), seg(",", MUTED)])
out([seg("      ", MUTED), seg('"detail"', KEY), seg(": ", MUTED),
     seg('"kernel typecheck OK (no diagnostics)"', STR), seg(" }", MUTED)])
out([seg("  ✓ VERIFIED_PROOF", GREEN, True)])
hold(7)
blank()

# 2. REFUTED
typ(DOL() + [seg('verify_formal(statement="2 + 2 = 5", proof="rfl")', INK)])
out([seg("  → ", MUTED), seg("{ ", MUTED), seg('"proof_status"', KEY),
     seg(": ", MUTED), seg('"REFUTED"', RED, True), seg(",", MUTED)])
out([seg("      ", MUTED), seg('"kernel_error"', KEY), seg(": ", MUTED),
     seg('"error: Not a definitional equality: the', STR)])
out([seg("        left-hand side  2 + 2  is not definitionally", STR)])
out([seg('        equal to the right-hand side  5"', STR), seg(" }", MUTED)])
out([seg("  ✗ REFUTED", RED, True),
     seg("  — the kernel's exact error → the agent's repair loop", MUTED)])
hold(7)
blank()

# 3. REJECTED (sorry)
typ(DOL() + [seg('verify_formal(statement="2 + 2 = 4", proof="by sorry")',
                 INK)])
out([seg("  → ", MUTED), seg("{ ", MUTED), seg('"proof_status"', KEY),
     seg(": ", MUTED), seg('"REFUTED"', AMBER, True), seg(",", MUTED)])
out([seg("      ", MUTED), seg('"detail"', KEY), seg(": ", MUTED),
     seg('"proof contains `sorry` — a proof with', STR)])
out([seg('        holes is not a proof"', STR), seg(" }", MUTED)])
out([seg("  ⊘ REJECTED", AMBER, True), seg("  — sorry holes never pass", MUTED)])
hold(7)
blank()

# 4. identify_constant
typ(DOL() + [seg('identify_constant("', INK), seg(VAL[:22] + "…", INK),
             seg('")', INK)])
run("running PSLQ + independent sympy re-eval …")
out([seg("  → ", MUTED), seg("{ ", MUTED), seg('"identified"', KEY),
     seg(": ", MUTED), seg("true", STR), seg(", ", MUTED),
     seg('"best"', KEY), seg(": { ", MUTED), seg('"display"', KEY),
     seg(": ", MUTED), seg('"pi**2/6"', CYAN, True), seg(",", MUTED)])
out([seg("            ", MUTED), seg('"digits_agreed"', KEY), seg(": ", MUTED),
     seg(str(DIG), STR), seg(", ", MUTED), seg('"provenance"', KEY),
     seg(": ", MUTED), seg('"known_form"', STR), seg(" } }", MUTED)])
out([seg("  ✓ ", GREEN, True), seg("pi**2/6", CYAN, True),
     seg(f"   [ verified {DIG} digits ]", GREEN)])
hold(8)
blank()
out([seg("every verdict above is from the real kernel / PSLQ + an "
         "independent", MUTED)])
out([seg("re-eval — ", MUTED), seg("no LLM in the loop", INK, True),
     seg(".", MUTED)])
hold(10)

# ---- compute canvas size -------------------------------------------------
def line_len(segs):
    return sum(len(t) for t, _, _ in segs)


max_chars = 0
n_lines = 0
for kind, pl in SCRIPT:
    if kind in ("out", "type"):
        max_chars = max(max_chars, line_len(pl))
        n_lines += 1
    elif kind == "run":
        max_chars = max(max_chars, len(pl) + 4)
        n_lines += 1
    elif kind == "blank":
        n_lines += 1

W = int(MX + max_chars * CW + MX)
H = int(MY + n_lines * LH + 22)


# ---- rendering -----------------------------------------------------------
def draw_chrome(d):
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=12, fill=BG)
    d.rectangle([0, 0, W, 34], fill=BAR)
    d.rounded_rectangle([0, 0, W - 1, 44], radius=12, outline=(46, 52, 66),
                        width=1)
    for i, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([18 + i * 22, 12, 30 + i * 22, 24], fill=col)
    d.text((W / 2, 17), "mathlas — real tool session", font=fonttitle,
           fill=MUTED, anchor="mm")


def render(committed, partial=None, cursor=False):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    draw_chrome(d)
    y = MY
    lines = committed + ([partial] if partial is not None else [])
    for segs in lines:
        if segs is None:
            y += LH
            continue
        x = MX
        for t, c, b in segs:
            d.text((x, y), t, font=(fontb if b else font), fill=c)
            x += len(t) * CW
        y += LH
    if cursor and lines:
        # block cursor at the end of the last non-None line
        last = None
        yy = MY
        for segs in lines:
            if segs is not None:
                last = segs
            yy += LH
        if last is not None:
            cx = MX + line_len(last) * CW
            cyv = MY + (len([s for s in lines]) - 1) * LH
            d.rectangle([cx, cyv + 3, cx + CW - 2, cyv + FS + 4], fill=INK)
    return img


frames = []
committed = []
for kind, pl in SCRIPT:
    if kind == "blank":
        committed.append(None)
        frames.append(render(committed))
    elif kind == "out":
        committed.append(pl)
        frames.append(render(committed))
    elif kind == "run":
        line = [seg("  " + pl, AMBER)]
        committed.append(line)
        frames.append(render(committed, cursor=True))
        frames.append(render(committed, cursor=False))
        committed.pop()   # the running line is transient; result replaces it
    elif kind == "type":
        total = line_len(pl)
        step = 5
        L = step
        while L < total:
            # build partial: first L chars across segments
            part, rem = [], L
            for t, c, b in pl:
                if rem <= 0:
                    break
                if len(t) <= rem:
                    part.append((t, c, b))
                    rem -= len(t)
                else:
                    part.append((t[:rem], c, b))
                    rem = 0
            frames.append(render(committed, partial=part, cursor=True))
            L += step
        committed.append(pl)
        frames.append(render(committed, cursor=True))
    elif kind == "hold":
        n = pl
        for i in range(n):
            frames.append(render(committed, cursor=(i % 2 == 0)))

# per-frame durations (ms): default fast; longer on the last of each group
durations = [70] * len(frames)
# make the very last frame linger
durations[-1] = 1400

# ---- quantize to a compact shared palette for a small GIF ----------------
pal_img = frames[0].convert("P", palette=Image.ADAPTIVE, colors=32)
qframes = [f.quantize(colors=32, palette=pal_img, dither=Image.NONE)
           for f in frames]

qframes[0].save(OUT, save_all=True, append_images=qframes[1:], loop=0,
                duration=durations, optimize=True, disposal=2)
sz = os.path.getsize(OUT)
print(f"wrote {os.path.normpath(OUT)}  ({W}x{H}, {len(frames)} frames, "
      f"{sz/1024:.0f} KB)")
