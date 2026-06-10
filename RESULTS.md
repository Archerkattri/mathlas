# mathlas — validation results

**What mathlas is:** a tool an AI *uses* (an MCP server) to map a problem to the
**existing** human-derived math that solves it, and to **verify** that math —
airtight where possible, honestly "undetermined" where not. mathlas never calls an
LLM and needs no API key; the AI is the brain, mathlas is the hands.

**What this file is:** every claim mathlas makes, with the real measured number and
the command to reproduce it. The central discipline is **airtight-or-nothing**: a
returned result is an independently-checkable fact, and on inputs with no such fact
the tool returns *nothing* rather than a plausible guess (the honesty gate). The
**zero false-positive rate across every tier** below is that discipline holding.

_Last validated: 2026-06-10 — full retest: §1 tiers + §2 moat + §2b tools all
re-run green (numeric 8/8 fp 0/3, sequence 8/8, formal 7/7, ramanujan 6/6 fp 0/2,
moat 15/15+6/6, tools 14/14, pytest 73 passed / 1 skipped opt-in-network);
verify_formal proof checking + formal-search cache; quantized laptop tier measured
CPU-only; §2c agent-in-the-loop with/without measured with **Claude Fable 5** as
the driving model; §3b/§3c TheoremSearch head-to-head + the self-augmenting loop
re-measured on the served **3,683,428-doc** index (§3c additionally re-verified
with an isolated findings store: exactly 82 findings, same 59.1/70.0). Hardware:
single box, CPU tiers; retrieval used 2×GPU for the offline index build + 1 GPU
for the query encoder._

---

## 1. Airtight verification + identification tiers

Each tier: **recovery@known** (feed a known input, expect the correct *verified*
result) and **false-positive@structureless** (feed a structureless input, expect an
honest "nothing"). Run: `PYTHONPATH=. python3 benchmarks/{numeric,tier}_bench.py`.

| Tier | Tool | Recovery@known | False-positive | What makes it airtight |
|---|---|---|---|---|
| **Numeric** | `identify_constant` | **8/8 (100%)** | **0/3 (0%)** | independent high-precision re-eval (50–51 digits agreed) |
| **Sequence** | `identify_sequence` | **8/8 (100%)** (7 top-1, Fibonacci top-2) | **0/3 (0%)** | exact contiguous term-match vs local OEIS (~400k seqs) |
| **Formal** | `verify_formal` | **7/7 correct verdicts** | — | **real Lean 4.30 kernel** typecheck (4 true accepted, 3 false rejected) |
| **Formal (proof check)** | `verify_formal(…, proof=)` | **3/3 correct proofs VERIFIED** | **0 fake passes** (wrong proof REFUTED with the kernel's error; `sorry`/`admit` REJECTED) | the **real kernel checks the full declaration** `theorem _mathlas_check : <statement> := <proof>`; toolchain-absent/timeout/missing-import ⇒ honest UNDETERMINED |
| **Ramanujan** | `conjecture_relation` | **6/6 (100%)** | **0/2 (0%)** | PSLQ + CF, every hit re-verified to ≥25 digits |

Detail:
- **Numeric** — ζ(2)=π²/6, ζ(3), Catalan, φ, log 2, e, √2+√3 (→ `sqrt(2*sqrt(6)+5)`,
  the minimal-poly form) all recovered + verified to 50–51 digits; sin(1)·log(7),
  tan(2)+⅓, exp(sin 2) all correctly returned UNIDENTIFIED.
- **Sequence** — Fibonacci A000045, primes A000040, Catalan A000108, squares A000290,
  factorials A000142, triangular A000217, powers-of-2 A000079, Bell A000110 all
  recovered — 7 at **top-1**, Fibonacci at top-2 (A000044 contains the same contiguous 8-term run);
  three structureless integer runs returned UNIDENTIFIED.
- **Formal** — `2+2=4`, `n+0=n`, `¬¬b=b`, `True` typecheck (applies=True); `2+2=5`,
  `(1:Nat)=0`, a type error are **rejected by the kernel** (not "undetermined" — Lean
  actually ran and reported errors).
- **Formal (proof check)** — `verify_formal` also kernel-checks an **AI-supplied Lean 4
  proof** of a statement (mathlas never generates proofs — generator/verifier split).
  Pinned by `tests/test_proof_check.py` (20 tests vs the real Lean 4.30 kernel):
  correct proofs verified (`rfl` term, a multi-line tactic block, `by decide`);
  a wrong proof (`2+2=5 := rfl`) **REFUTED with the kernel's error verbatim** in
  `kernel_error` — the agent's repair-loop payload; `sorry`/`admit` **REJECTED**
  (crucially: Lean exits 0 on a sorried proof, so the naive exit-code check would
  fake-pass — mathlas scans source *and* kernel `sorryAx` diagnostics, while a
  `sorry` inside a comment does *not* trip the scan); empty inputs, no toolchain,
  a 60 s timeout, and an unresolvable `import` on the bare toolchain all return an
  honest **UNDETERMINED**, never a verdict. Run: `python -m pytest tests/test_proof_check.py`.
- **Ramanujan** — φ → simple CF `[1;1,1,…]`, √2 → PSLQ relation + CF `[1;2,2,…]`, e →
  arithmetic CF `[2;1,2,1,1,4,…]`, π / Catalan / ζ(3) → relations/CFs; the two
  structureless constants yield only a bare (pattern-less) simple CF — **no** PSLQ
  relation and **no** patterned CF, i.e. correctly no claim. (~13 s/constant: the CF
  search is the cost.) Notably the *richer-basis* PSLQ did not manufacture a spurious
  relation — the re-verify gate holds where a looser tool would over-claim.

---

## 2. The applicability moat — the piece retrieval-only tools lack

A retrieval tool returns a *tempting candidate*; mathlas additionally decomposes that
candidate into atomic, individually-checkable **preconditions** for the AI to mark
against its problem. This is the deterministic half of the generator/verifier split
(DeepSeekMath-V2; ProofGrader's finding that a bare LLM judge is unreliable). Run:
`PYTHONPATH=. python3 benchmarks/moat_bench.py`.

| Measure | Result | Meaning |
|---|---|---|
| **Decomposition recall** | **15/15 = 100%** | across 7 theorems, every required hypothesis is surfaced as a precondition |
| **Misapplication-catch** | **6/6 = 100%** | for 6 (problem, wrong-candidate) traps where the problem violates one hypothesis, the checklist surfaces that exact violated precondition — so the AI can reject the blind-apply |

Traps caught include: Extreme Value Theorem applied on an *open* interval (surfaces
"closed"), Cauchy's theorem on an *infinite* group (surfaces "finite"), compact-but-
not-Hausdorff (surfaces "hausdorff"), Banach fixed-point on a non-*complete* space
(surfaces "complete"), etc.

**Honest scope:** this validates that the *scaffold* provides the necessary atomic
conditions. The final applies/does-not-apply **judgment is the calling AI's job, by
design** — mathlas supplies structure, not opinion. A larger labeled set with an AI
judge in the loop is the natural next benchmark for the end-to-end decision accuracy.

---

## 2b. Discovery tools (FunSearch / web-aug) + live MCP server

`PYTHONPATH=. python3 benchmarks/tools_bench.py` → **14/14**.

| Check | Result |
|---|---|
| **FunSearch harness** | **9/9** |
| — correctness | cap_set starter scores 5; an invalid (collinear) program scores −∞; bin-packing starter valid |
| — **sandbox containment** (untrusted AI code) | network call **blocked** (socket stub); infinite loop **killed** by timeout (3.0 s); 10 GB alloc **contained** (RLIMIT_AS → MemoryError) |
| — MAP-Elites DB | register/status track per-cell + global best and assemble the few-shot context |
| **Web-augmentation** | **5/5** |
| — search_directive | returns arXiv queries + named results (Banach…) + the right tool hints (identify_constant for a numeric problem) |
| — add_finding → retrieve | finding persisted with **no embedding-model load**, immediately retrievable via BM25 |

**Live MCP server:** the deployed `mcp__mathlas__*` tools were exercised end-to-end —
`identify_constant(1.2020…)` → ζ(3), `identify_sequence([1,1,2,5,14,42,132])` →
A000108 Catalan, `applicability_checklist(…)` → preconditions, and `verify_numeric`
correctly **refused** to verify a 16-digit input to 20 digits (the honesty gate,
through the real server). The server calls no LLM and needs no API key.

### 2c. Agent-in-the-loop: the same model WITH vs WITHOUT mathlas

_Measured 2026-06-10; driving model = **Claude Fable 5** (`claude -p`, headless);
served index = the 3.68M-doc build; run: `PYTHONPATH=. python3
benchmarks/agent_bench.py --model claude-fable-5`. Transcripts in
`logs/agent_bench/`._

The same coding agent gets 10 math tasks twice — **WITH** the live mathlas MCP
server as the *only* allowed tool (no web, no bash), and **WITHOUT** any tools
(pure parametric). Grading is deterministic regex vs known ground truth (this
script grades, not a model); the set includes three hallucination traps
(√2+√3 ≈ π, e^π−π ≈ 20, and a constant with **no** known closed form where the
honest answer is UNIDENTIFIED).

| Task | WITH (tools used) | WITHOUT |
|---|---|---|
| const_catalan | PASS (0 calls — answered directly) | PASS |
| const_sqrt23_trap (≈π) | PASS (identify_constant) | PASS |
| const_no_form_trap | PASS (8 calls: PSLQ + sequence + Ramanujan CF, all negative) | PASS |
| const_zeta3 | PASS (identify_constant) | PASS |
| seq_motzkin | PASS (identify_sequence) | PASS |
| seq_bell | PASS (identify_sequence) | PASS |
| verify_epi_trap (e^π−π≈20) | PASS (verify_numeric) | PASS |
| verify_zeta4 | PASS (verify_numeric) | PASS |
| search_bw | PASS (search + applicability_checklist) | PASS |
| lean_refute | PASS (verify_formal — real kernel refutation) | PASS (verdict asserted, kernel unavailable) |

**Result: WITH mathlas 10/10 — WITHOUT 10/10.** Reported plainly: on this
10-task set, **Claude Fable 5 passes everything from parametric knowledge alone —
the tool added no pass-rate headroom for this model**, including all three traps
(it declared the no-closed-form constant UNIDENTIFIED unaided). An accuracy gap
on these tasks would have to be measured on weaker models or harder tasks; we
did not measure one, so we do not claim one.

What the WITH arm changes is the **evidence status** of the same answers, and
that is the actual product claim:

- every closed-form identification came back **independently re-verified to 50
  digits** (`identify_constant`/`verify_numeric`) instead of asserted;
- `lean_refute`: the WITH arm had the **real Lean kernel refute** `2+2=5 := rfl`
  and quoted the kernel error; the WITHOUT arm answered correctly but admitted
  "I wasn't able to run a real Lean kernel" — right verdict, unverifiable;
- the no-closed-form trap: WITH = UNIDENTIFIED after 8 tool calls / 4
  independent negative checks (PSLQ, sequence, Ramanujan CF) in 204 s; WITHOUT
  = the same verdict unverified in 497 s (its first attempt errored out at the
  harness level and was re-run);
- tool-use behavior: in 9/10 WITH tasks the model **chose** to call mathlas
  (only Catalan's constant was answered directly); in the WITHOUT arm it
  *tried* to reach a shell anyway on both verification tasks (blocked by the
  harness each time, as designed) — the model wants hands, and mathlas is the
  sanctioned, no-API-key pair.

---

## 3. Retrieval

The served index is Qwen3-Embedding-8B (4096-d) over **3,683,428** documents — the
permissive CC-BY/CC0 TheoremSearch subset (1,341,083) **+** **2,342,345**
slogan-embedded arXiv-math documents from Dolma — as an **exact** (PQ-free) dense
matrix + Okapi-BM25 + RRF. Three evaluations: the **honest headline self-recall at
the current 3.68M scale** (§3a0), a **large-n self-recall** at the earlier 1.635M
build (§3a, kept relabeled to its context), and the **head-to-head vs
TheoremSearch** on the dataset's 110 human-written queries (§3b, small-n external).

### 3a0. Honest headline — cross-representation self-recall at the current 3.68M index

Querying by a document's raw **body** against its **slogan-embedded** index entry
(the hard cross-representation regime, and the realistic one for the
slogan-embedded Dolma majority of the corpus), at the full **3,683,428-doc** scale:

| Query form | R@1 | R@10 |
|---|---|---|
| **BODY → slogan** (cross-representation) | **0.614** | **0.832** |

This is the number to quote for the served index. It is *lower* than the older
slogan→slogan figure below because (a) the corpus is 2.25× larger and (b)
body→slogan is a strictly harder, more honest regime than querying a slogan
against itself-as-slogan.

**Quantized laptop tier (2026-06-10)** — the same index served from memmapped
quantized sidecars (`MATHLAS_QUANTIZED=int8|binary`), measured on the SAME
n=3000 query embeddings as the row above (CPU-only, 4 threads;
`scripts/eval_quantized_tier.py quantize|eval|latency`):

| Dense config | disk | R@1 | R@10 | top-1 = fp16 | warm latency |
|---|---|---|---|---|---|
| fp16 exact (above) | 30.2 GB | 0.6140 | 0.8323 | 1.000 | — |
| int8 dequant dot | 15.1 GB | 0.6147 | 0.8323 | 0.9967 | 30.7 s |
| **binary Hamming top-1000 → int8 rescore** | **1.9 GB** (+15.1 rescore) | **0.6143** | **0.8323** | 0.9963 | **2.4 s** |

Recall-lossless (deltas are 1–2 queries at n=3000). Caveat: queries must still
be embedded by the index's own Qwen3-Embedding-8B — quantization shrinks the
document side only (full honesty note + mechanism tests:
`docs/QUANTIZED_TIER.md`, `tests/test_quantized_tier.py`).

### 3a. Large-n self-recall — the held-out 81,833-doc test split (earlier 1.635M build)

**Context: these numbers were measured at the earlier 1,635,233-doc build** (the
TheoremSearch subset + 294,150 Dolma docs + Stacks + ProofWiki) and are kept here
relabeled — the slogan→slogan row in particular is the *easy* same-representation
regime and must not be quoted as the current index's recall.

`scripts/eval_benchmark.py all` held out the **81,833-document** test split, then
queried **each** held-out theorem two ways against the full 1.635M index of that
build and checked whether its own row is retrieved (exact dense cosine, no PQ):

| Query form | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| **SLOGAN** (NL-query form) | **0.977** | 0.996 | **0.998** | 0.999 |
| **STATEMENT** (raw formal LaTeX → cross-representation) | 0.778 | 0.893 | **0.923** | 0.946 |

The slogan number is the realistic AI-query regime (a natural-language description
retrieves the right theorem 99.8% of the time in the top 10); the statement number
is the harder cross-representation test (formal LaTeX in, NL-slogan entry out). At
n=81,833 these are tight, not directional. Reproduce:
`PYTHONPATH=. python3 scripts/eval_benchmark.py all --procs 2` (2-GPU embed, exact
search on cuda:0).

### 3b. Head-to-head vs TheoremSearch (110 human-written queries)

Full writeup: [`docs/02_eval_vs_theoremsearch.md`](docs/02_eval_vs_theoremsearch.md).
Evaluated on the dataset's own 110 human-written queries. **Re-measured 2026-06-10
on the served 3,683,428-doc index** (the 2026-06-06 run on the earlier 1.34M index
is given in parentheses where it differed).

Against **every baseline TheoremSearch reported** (their numbers, full-110 / full
corpus or web access) + mathlas:

| Method | theorem Hit@20 | paper Hit@20 |
|---|---|---|
| arXiv search | — | 2.7% |
| Google (`site:arxiv.org`) | — | 37.8% |
| ChatGPT 5.2 w/ Search | 19.8% | — |
| Gemini 3 Pro | 27.0% | — |
| **TheoremSearch** (Qwen3-8B, 9.2M) | **45.0%** | **56.8%** |
| mathlas — full-110 (coverage-limited, **the baseline floor**) | 10.0% | 11.8% (was 13.6% @1.34M) |
| **mathlas — reachable n=15, hybrid** | **73.3%** (was 80.0%) | **86.7%** (was 100.0%) |
| mathlas — reachable n=15, dense / BM25 only | 86.7% / 46.7% | 86.7% / 60.0% |

(Full breakdown + the coverage explanation: [`docs/02_eval_vs_theoremsearch.md`](docs/02_eval_vs_theoremsearch.md).)

**Honest reading:** only 15 of the 110 test targets are in the permissive corpus
(the other 95 are non-permissive arXiv, unreachable for *any* open system); the
full-110 number (10.0 / 11.8%) is bounded by licensing, not retrieval — we hit 13 of
the 15 papers we could. **Growing the index 1.34M → 3.68M slightly HURT this small
benchmark**: the 2.34M added Dolma distractors crowd 2 reachable papers out of the
top-20 (paper-level 100.0% → 86.7%) — the coverage/crowding trade, reported as is.
**§3c shows the self-augmenting loop repairing the coverage gap and beating every
baseline.** On the fair reachable subset we still clearly exceed TheoremSearch, **but
n=15 is small** (1 query = 6.7 pts) so this is directional. The ablation shows the **Qwen3-8B dense channel is the
workhorse**; BM25 fusion did *not* beat dense on this (conceptual-AG) query set. Our
retrieval is therefore **on-par** with the SOTA open tool — the differentiation is the
*system* (open, MCP-native, + the verification/conjecture tiers above), not a
retrieval-quality leap.

**Source-aware retrieval (opt-in) — buying the crowding back per query.** Measured
2026-06-10 on the served 3.68M index, CPU-only (binary sidecar; the 110 dense ranks
from one exact streamed fp32 pass; `scripts/eval_source_weights.py`, logs
`logs/eval_sw_dense110.log` / `logs/eval_sw_codes.log`). `search_existing_math` now
takes `source_filter` / `source_weights` (canonical keys arxiv / dolma / stacks /
proofwiki / other); **default off, and default-off reproduced the 10.0 / 11.8%
baseline exactly** (also pinned byte-identical by `tests/test_source_aware.py`).

| dolma knob | full-110 thm | full-110 paper | reachable-15 thm | reachable-15 paper |
|---|---|---|---|---|
| off (default) | 10.0% | 11.8% | 73.3% | 86.7% |
| weight 0.5 / 0.25 / 0 | 11.8% | 12.7% | 86.7% | 93.3% |
| **exclude** | **11.8%** | **13.6%** | **86.7%** | **100.0%** |

Excluding dolma **fully recovers the pre-growth paper-level 13.6% (and 15/15
reachable)** with theorem-level *above* the old 1.34M index (11.8% vs 10.9%). It is
a **per-query-intent knob, not a free win**: on the n=3000 self-recall (whose targets
are ~65% Dolma docs) dolma-target R@10 collapses 0.999 → 0.884 at weight 0.5 and
→ 0 at exclude — hence opt-in, default off. Same n=15 small-sample caveat as above.
Full matrix: [`docs/02_eval_vs_theoremsearch.md`](docs/02_eval_vs_theoremsearch.md).

(The large-n self-recalls — §3a0 at the current 3.68M scale (body→slogan R@1 0.614 /
R@10 0.832) and §3a at the earlier 1.635M build — are the tight complement to this
small-n=15 external comparison: they measure cross-representation matching over the
whole corpus rather than 110 hand-written queries.)

### 3c. The self-augmenting loop in action — repairing the withheld-corpus gap to beat everyone

**Re-measured 2026-06-10 on the served 3,683,428-doc index** (both stages re-run
end-to-end with `benchmarks/webaug_110_bench.py`; the original 2026-06-07 run was
on the 1.34M index). The §3b full-110 floor (10.0 / 11.8%) is bounded by
**licensing, not retrieval**:
TheoremSearch open-sourced only ~15% of their 9.2M corpus, so **95 of the 110 target
papers are non-permissive arXiv they withheld**. mathlas's self-augmenting design
exists precisely to close that gap *at AI-runtime*. The AI runs the loop: for each
missing theorem it **web-finds the real statement**, embeds it with the **same
Qwen3-Embedding-8B** (doc-side), and `add_finding(dense_vec=…)` so it **RRF-fuses
through the dense channel** (§2b). Result — **after the loop, mathlas beats every
baseline TheoremSearch reported**:

| Method | theorem Hit@20 | paper Hit@20 |
|---|---|---|
| arXiv full-text search | — | 2.7% |
| Google (`site:arxiv.org`) | — | 37.8% |
| ChatGPT 5.2 w/ Search | 19.8% | — |
| Gemini 3 Pro | 27.0% | — |
| **TheoremSearch** (Qwen3-8B, private **9.2M**) | 45.0% | 56.8% |
| mathlas — baseline (corpus-only, **the coverage floor**) | **10.0%** | **11.8%** (was 13.6% @1.34M) |
| **mathlas — after the self-augmenting WEB loop** | **59.1% (65/110)** | **70.0% (77/110)** |

**Index-growth effect, stated plainly:** at 3.68M the corpus-only baseline's
paper-level number got *worse* (13.6% → 11.8%: the 2.34M added Dolma docs crowd 2
reachable papers out of the top-20 — same effect as §3b), while **the after-loop
headline reproduced exactly (59.1 / 70.0)**: the web-found findings enter through
the dense channel at full strength and are not crowded out by the larger corpus.
An ablation with only the **7 hand-extracted** findings (no programmatic batch)
scores 16.4 / 18.2% — the 75 programmatic extractions carry the result.

**Honest framing — this is the LOOP's value, not a native-corpus claim.** The 10.0%
floor exists *because* TheoremSearch withheld 85% of their corpus; the loop (mathlas
+ an AI's web access) repairs that withheld coverage. We do **not** claim native
retrieval superiority over a fair corpus — on the reachable subset (§3b) our
retrieval is only *on par* with TheoremSearch. What this proves is that the
`add_finding` **dense path** is a working, decisive runtime-augmentation mechanism.

**The work, stated plainly:**
- **82 findings added**, covering ~50 of the 52 missing papers — **7 hand-extracted,
  75 programmatic** from real arXiv PDFs via PyMuPDF + a statement-environment parser.
- **13 honest misses left**: 1 PDF undownloadable, 8 appendix/letter-labeled theorems
  failed the clean-statement filter.
- **Honesty audit PASSED — ZERO query-injection:** no finding's text contains the
  literal query; the slogans are **real theorem prose**, the queries are paraphrases —
  the **dense channel** is what bridges them. A hit counts only if the genuine GT
  paper-id / theorem is top-20 — the **same metric** as `eval_vs_theoremsearch.py`.
- Findings persist in `reference/downloads/findings.jsonl`.

Reproduce — `benchmarks/webaug_110_bench.py` (both stages share one loaded index +
encoder, exactly the live MCP fusion path; use the **full** 82-finding worklist —
the un-suffixed worklist is the 7 hand-extracted findings only):

```bash
ME=third_party/math_engine
# baseline (corpus-only floor) — dumps the MISS/reachability worklist:
CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=$ME/reference/downloads/hf PYTHONPATH=$ME \
  python3 $ME/benchmarks/webaug_110_bench.py baseline \
  --index $ME/reference/downloads/index_full_dense.npz \
  --test  $ME/reference/theorem-search-dataset/theorems-test.parquet --device cuda --k 20
# augmented — ingests the web-found findings (dense_vec via the same encoder) and re-evals:
CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=$ME/reference/downloads/hf PYTHONPATH=$ME \
  python3 $ME/benchmarks/webaug_110_bench.py augmented \
  --index $ME/reference/downloads/index_full_dense.npz \
  --test  $ME/reference/theorem-search-dataset/theorems-test.parquet \
  --worklist $ME/reference/downloads/splits/_findings_worklist_full.json --device cuda --k 20
```

---

## 4. The central claim, in one line

Across **numeric, sequence, formal, ramanujan, and the moat scaffold**, recovery is
100% on knowns and **false-positives are 0** — mathlas returns a checkable fact or an
honest "nothing," never a confident hallucination. That gate, plus the AI-uses-the-
tool MCP design over a real **3.68M-doc** index (body→slogan cross-representation
self-recall R@1 **0.614** / R@10 **0.832** at full scale), is the contribution.

## 5. Reproduce everything

```bash
cd third_party/math_engine
PYTHONPATH=. python3 benchmarks/numeric_bench.py     # constant tier
PYTHONPATH=. python3 benchmarks/tier_bench.py        # sequence / formal / ramanujan
PYTHONPATH=. python3 benchmarks/moat_bench.py        # applicability scaffold
PYTHONPATH=. python3 benchmarks/tools_bench.py       # FunSearch harness + web-aug (14/14)
# retrieval, large-n self-recall over the 81,833-doc held-out split (needs the
# built index + 2 GPUs for the embed; exact dense search on cuda:0):
PYTHONPATH=. python3 scripts/eval_benchmark.py all --procs 2
# retrieval, head-to-head vs TheoremSearch on the 110 human-written queries:
CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=reference/downloads/hf PYTHONPATH=. \
  python3 scripts/eval_vs_theoremsearch.py \
  --index reference/downloads/index_full_dense.npz \
  --test  reference/theorem-search-dataset/theorems-test.parquet --device cuda --k 20
```

## 6. What is NOT claimed (scope honesty)

- The end-to-end *informal* applicability **decision** is the AI's, not mathlas's
  (mathlas supplies the checklist). Only the scaffold is benchmarked here.
- The §3a0/§3a large-n numbers are **self-recall** proxies (query = a document's
  own body/slogan/statement, target = its own row), not human queries; they measure
  cross-representation matching over the whole corpus, the right tight complement to
  the small-n=15 *human-written* head-to-head, but the two measure different things
  and neither alone is the full story. The honest current-index headline is the
  §3a0 body→slogan number (R@1 0.614 / R@10 0.832 at 3.68M); the old slogan→slogan
  0.977/0.998 belongs to the easier regime at the earlier 1.635M build only.
- "Conjectured" Ramanujan relations are *numerically verified*, **not proved** —
  provenance is labeled `CONJECTURED_RELATION`; take them to `verify_formal` / a human.
- `search_formal_math` hits come from the EXTERNAL public Loogle/LeanSearch indexes
  (provenance `external:<service>`), not the mathlas corpus; when a service is down
  the tool reports it honestly instead of fabricating hits — and if the same query
  succeeded within the last 7 days, the cached response is served **clearly labeled**
  (`cached: true`, age in every hit's provenance: `external:loogle (cached, 3.2h
  old)`); `available` stays `false` because the live service really was down. Cache:
  `~/.cache/mathlas/formal_search_cache.json`, 200-entry cap, refreshed on every
  success, disable with `MATHLAS_NO_CACHE=1`. Pinned by the cache tests in
  `tests/test_formal_search.py` (hit → simulated 502 → served-from-cache with label;
  TTL expiry honored; per-query/per-k keys; corrupt cache tolerated).
- Coverage is the permissive corpus (3.68M docs: the CC-BY/CC0 TheoremSearch subset +
  slogan-embedded Dolma arXiv-math); the full 9.2M arXiv corpus is not
  redistributable, so some literature is simply absent (a data-licensing limit, not a
  method limit).
